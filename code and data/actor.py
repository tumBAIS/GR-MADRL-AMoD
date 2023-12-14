"""Multi-agent actor including post-processing. All computations are made across a mini-batch and agents in parallel."""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.keras.layers import Dense, Activation, Multiply
from tensorflow.keras.regularizers import L2
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching
from scipy.optimize import linear_sum_assignment
from joblib import Parallel, delayed


class Actor(tf.keras.Model):
    def __init__(self, args, env):
        super().__init__(name="Actor")
        
        self.request_embedding = RequestEmbedding(args)
        self.vehicle_embedding = VehicleEmbedding(args)
        self.requests_context = RequestsContext(args)
        self.vehicles_context = VehiclesContext(args)
        
        reg_coef = args["regularization_coefficient"]
        inner_layers = []
        for layer_size in args["inner_units"]:
            layer = Dense(layer_size, activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(reg_coef))
            inner_layers.append(layer)
        self.inner_layers = inner_layers

        self.output_logits = Dense(2, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
        self.activation = Activation('softmax', dtype='float32')
        
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.post_processing_mode = args["post_processing"]

        # For greedy start
        self.episode_max_steps = int(args["episode_length"] / args["time_step_size"])
        self.max_waiting_time = int(args["max_waiting_time"] / args["time_step_size"])
        self.cost_parameter = args["cost_parameter"]
        self.data_dir = args["data_dir"]
        self.graph = pd.read_csv(self.data_dir + f'/graph(dt_{args["time_step_size"]}s).csv', index_col=[0,1], usecols=[0,1,2,3,5])
        zone_IDs = self.graph.index.unique(level=0).tolist()
        idx = pd.MultiIndex.from_tuples([(i,i) for i in zone_IDs], names=["origin_ID", "destination_ID"])
        additional_entries = pd.DataFrame([[0,0] for i in range(len(zone_IDs))], idx, columns=["distance", "travel_time"])
        self.graph = pd.concat([self.graph, additional_entries])
        self.max_time = self.graph.travel_time.max()

        # For myopic policy rvaluation
        self.zones = pd.read_csv(self.data_dir + '/zones.csv', header=0, index_col=0)
        
        self.env = env
    
    def call(self, state, hvs, test, request_masks):
        probs = self.compute_prob(state, hvs, request_masks["s"])
        act = self.post_process(probs, test, hvs, request_masks["l"])
        return tf.squeeze(act, axis=[0])

    def get_random_action(self, state, hvs, request_mask_l):
        probs = tf.ones((1, self.n_veh*self.n_req_max, 2)) / 2
        act = self.post_process(probs, tf.constant(False), tf.expand_dims(hvs, axis=0), request_mask_l)
        return tf.squeeze(act, axis=[0])
    
    def get_struct_analysis(self, state, hvs, rl_action):
        ## First part: Calculation of the per-agent profits
        # get the vehicle and request states
        requests_state = state["requests_state"]
        requests_state = tf.expand_dims(requests_state, axis=0)
        vehicles_state = state["vehicles_state"]
        vehicles_state = tf.expand_dims(vehicles_state, axis=0)
        # get distance of the request
        distance1 = requests_state[:,:,4] * self.env.max_distance
        distance1 = tf.repeat(distance1, repeats=self.env.n_veh, axis=1)
        # check if request is within max waiting time
        within_max_waiting_time = self.env.get_flag_within_max_waiting_time(vehicles_state, hvs, requests_state)
        within_max_waiting_time = tf.reduce_sum(within_max_waiting_time, axis=2)
        # get location of each vehicle (which is the position = end of last request)
        location = tf.where(hvs[:,:,7] != -1, hvs[:,:,7], tf.where(hvs[:,:,4] != -1, hvs[:,:,4], hvs[:,:,0]))
        location = tf.cast(tf.tile(location, multiples=tf.constant([1,self.env.n_req_max])), dtype=tf.int32)
        # get pickup location for each request
        pickup_horizontal = tf.cast(tf.math.round(requests_state[:,:,0] * self.env.max_horizontal_idx), tf.int32)
        pickup_vertical = tf.cast(tf.math.round(requests_state[:,:,1] * self.env.max_vertical_idx), tf.int32)
        pickup_1 = self.env.zone_mapping_table[self.env.get_keys(pickup_horizontal, pickup_vertical)]
        pickup = tf.repeat(pickup_1, repeats=self.env.n_veh, axis=1)
        pickup_numpy = tf.squeeze(pickup, axis=0).numpy()
        idx = tf.where(location == pickup, (pickup + 1) % (self.env.nodes_count - 1), location)
        # get destination location for each request
        destination_horizontal = tf.cast(tf.math.round(requests_state[:,:,2] * self.env.max_horizontal_idx), tf.int32)
        destination_vertical = tf.cast(tf.math.round(requests_state[:,:,3] * self.env.max_vertical_idx), tf.int32)
        destination_1 = self.env.zone_mapping_table[self.env.get_keys(destination_horizontal, destination_vertical)]
        destination = tf.repeat(destination_1, repeats=self.env.n_veh, axis=1)
        destination_numpy = tf.squeeze(destination, axis=0).numpy()
        #get the distance from the vehicle to the pickup location
        distance2 = tf.where(location != pickup, self.env.distance_table[self.env.get_keys(idx, pickup)], 0)
        distance2 = tf.where(distance1 == 0, 0, distance2) # set distance for dummy requests to 0
        # calculate the fares
        fares = []
        for i in range(len(destination_numpy)):
            if destination_numpy[i] - pickup_numpy[i] == 0: fares.append(0)
            else: fares.append(self.graph.loc[(pickup_numpy[i], destination_numpy[i]), "fare"])
        fares = tf.expand_dims(tf.cast(fares, dtype=tf.float32), axis=0)
        # calculate the profit
        profit = fares + self.cost_parameter * (tf.cast(distance1, dtype=tf.float32) + tf.cast(distance2, dtype=tf.float32))
        profit = tf.where(profit <= 0, 0, profit)
        profit = tf.where(within_max_waiting_time==1, profit, 0)

        ## Second part: Calculation of the test metrics
        # rejection rate
        rejections = tf.where(rl_action == -1, 1, 0)
        profitable_requests = tf.where(tf.reduce_sum(tf.reshape(profit, [self.n_req_max, self.n_veh]), axis=1) > 0, 1, 0)
        rejections = rejections * profitable_requests
        # conditional rejection rates
        locations = tf.squeeze(tf.where(hvs[:,:,7] != -1, hvs[:,:,7], tf.where(hvs[:,:,4] != -1, hvs[:,:,4], hvs[:,:,0])), axis=0)
        zones_vehicles = np.zeros(len(self.zones), dtype = np.int32)
        for i in range(len(zones_vehicles)):
            zones_vehicles[i] = tf.reduce_sum(tf.where(locations == i, 1, 0)).numpy()
        cond_rejection_rates = self.cond_rejection_rates(zones_vehicles, destination_1, profitable_requests, rejections)
        # Profits
        profit = tf.reduce_sum(tf.reshape(profit, [self.n_req_max, self.n_veh]), axis=1)
        return profitable_requests, rejections, cond_rejection_rates, profit
    
    def cond_rejection_rates(self, zones_vehicles, destination_1, profitable_requests, rejections):
        # vector initialization
        cond_rejection_rates = []
        
        # rejection rate for 0 vehicles in destination
        profitable_req_cond = 0
        rejections_cond = 0
        for i in range(len(zones_vehicles)):
            if zones_vehicles[i] == 0:
                requests_cond = tf.where(tf.squeeze(destination_1, axis=0).numpy()==i, 1, 0)
                profitable_req_cond += tf.reduce_sum(profitable_requests*requests_cond).numpy()
                rejections_cond += tf.reduce_sum(rejections*requests_cond).numpy()
        cond_rejection_rates.append([rejections_cond, profitable_req_cond])

        # rejection rate for >2 vehicles in destination
        profitable_req_cond = 0
        rejections_cond = 0
        for i in range(len(zones_vehicles)):
            if zones_vehicles[i] >2:
                requests_cond = tf.where(tf.squeeze(destination_1, axis=0).numpy()==i, 1, 0)
                profitable_req_cond += tf.reduce_sum(profitable_requests*requests_cond).numpy()
                rejections_cond += tf.reduce_sum(rejections*requests_cond).numpy()
        cond_rejection_rates.append([rejections_cond, profitable_req_cond])

        return cond_rejection_rates
        
    @tf.function
    def compute_prob(self, state, hvs, request_mask_s):
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]

        request_embedding = self.request_embedding(requests_state)
        vehicle_embedding = self.vehicle_embedding(vehicles_state)
        requests_context = self.requests_context(request_embedding, request_mask_s)
        vehicles_context = self.vehicles_context(vehicle_embedding)
        
        context = tf.concat([requests_context, vehicles_context], axis=1)
        context = tf.repeat(tf.expand_dims(context,axis=1), repeats=self.n_veh, axis=1)
        misc_state = tf.repeat(tf.expand_dims(tf.cast(state["misc_state"], tf.float16), axis=1), repeats=self.n_veh, axis=1)
        combined_input = tf.concat([misc_state, context, vehicle_embedding], axis=2)
        combined_input = tf.tile(combined_input, multiples=[1,self.n_req_max,1])
        
        request_embedding = tf.repeat(request_embedding, repeats=self.n_veh, axis=1)
        
        within_max_waiting_time = self.env.get_flag_within_max_waiting_time(vehicles_state, hvs, requests_state)
        
        features = tf.concat([combined_input, within_max_waiting_time, request_embedding], axis=2)
        
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(features, paddings, constant_values=0.)
        
        for layer in self.inner_layers:
            features = layer(features)
                
        return self.activation(self.output_logits(features))

    def post_process(self, probs, test, hvs, request_mask_l):
        batch_size = tf.shape(probs)[0]

        probs = tf.where(probs == 0, 1e-16*tf.ones_like(probs), probs)
        
        probs = self.mask_probs(probs, hvs, request_mask_l)
        
        if self.post_processing_mode == "matching":
            act = self.get_action_from_probs(probs, test)
            act = self.reshape_transpose(act, batch_size, self.n_veh)
            act = act.numpy()
            action_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.matching)(act[i,:,:]) for i in range(batch_size))
            act = tf.constant(action_list)
        
        if self.post_processing_mode == "weighted_matching":
            act = probs[:,:,1]
            sampled_action = self.get_action_from_probs(probs, test)
            act = act * tf.cast(sampled_action, tf.float32) # set score to zero if decision is reject
            act = self.reshape_transpose(act, batch_size, self.n_veh)
            act = act.numpy()
            action_list = Parallel(n_jobs=2, prefer="threads")(delayed(self.weighted_matching)(act[i,:,:]) for i in range(batch_size))
            act = tf.constant(action_list)
          
        return act
    
    @tf.function
    def mask_probs(self, probs, hvs, request_mask_l):
        mask = hvs[:,:,6] == -1 # Set probability to zero if vehicle already serves 2 requests
        mask = tf.tile(mask, multiples=tf.constant([1,self.n_req_max]))
        mask = mask & tf.cast(request_mask_l, tf.bool) # Set probability to zero if request is a dummy request
        mask = tf.expand_dims(mask, axis=2)
        dummy_mask = tf.ones((hvs.shape[0], self.n_veh*self.n_req_max, 1), dtype=tf.bool)
        mask = tf.concat([dummy_mask, mask], axis=2)
        
        probs = probs * tf.cast(mask, tf.float32)
        probs /= tf.reduce_sum(probs, axis=2, keepdims=True)
        
        return probs
    
    @tf.function
    def get_action_from_probs(self, probs, test):
        if test:
            return tf.argmax(probs, axis=2, output_type=tf.int32)
        else:
            return tfp.distributions.Categorical(probs=probs).sample()
    
    @tf.function
    def reshape_transpose(self, act, batch_size, n_veh):
        act = tf.reshape(act, [batch_size, self.n_req_max, n_veh])
        return tf.transpose(act, perm=[0,2,1])
    
    def matching(self, x):
        return maximum_bipartite_matching(csr_matrix(x))
    
    def weighted_matching(self, x):
        try:
            matched_veh, matched_req = linear_sum_assignment(x, maximize=True) # weighted matching
        except ValueError:
            print(x)
        
        matched_weights = x[matched_veh, matched_req]
        matched_veh = np.where(matched_weights == 0., -1, matched_veh) # if weight is zero, correct matching decision to reject decision
        
        action = -np.ones(self.n_req_max, int)
        action[matched_req] = matched_veh
        
        return action


class RequestEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestEmbedding")
        
        self.embedding_layer = Dense(args["req_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    @tf.function
    def call(self, requests_state):
        paddings = tf.constant([[0,0],[0,0],[0,3]])
        features = tf.pad(requests_state, paddings, constant_values=0.)
        
        return self.embedding_layer(features)


class VehicleEmbedding(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehicleEmbedding")
        
        self.embedding_layer = Dense(args["veh_embedding_dim"], activation="relu", kernel_initializer='he_uniform', kernel_regularizer=L2(args["regularization_coefficient"]))

    @tf.function
    def call(self, vehicles_state):
        paddings = tf.constant([[0,0],[0,0],[0,4]])
        features = tf.pad(vehicles_state, paddings, constant_values=0.)
        
        return self.embedding_layer(features)


class RequestsContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="RequestsContext")

        self.attention = args["attention"]
        reg_coef = args["regularization_coefficient"]

        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["req_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, requests_embeddings, request_mask_s):
        if self.attention:
            betas = Multiply()([self.w(self.W(requests_embeddings)), tf.expand_dims(request_mask_s, axis=2)])
        else:
            betas = tf.expand_dims(tf.cast(request_mask_s, tf.float16), axis=2)
        
        return tf.reduce_sum(betas * requests_embeddings, axis=1) / tf.reduce_sum(tf.cast(request_mask_s, tf.float16), axis=1, keepdims=True)


class VehiclesContext(tf.keras.layers.Layer):
    def __init__(self, args):
        super().__init__(name="VehiclesContext")
        
        self.attention = args["attention"]
        reg_coef = args["regularization_coefficient"]

        if self.attention:
            self.w = Dense(1, activation="sigmoid", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))
            self.W = Dense(args["veh_context_dim"], activation="tanh", use_bias=False, kernel_initializer='glorot_uniform', kernel_regularizer=L2(reg_coef))

    @tf.function
    def call(self, vehicles_embeddings):
        if self.attention:
            betas = self.w(self.W(vehicles_embeddings))
            return tf.reduce_mean(betas * vehicles_embeddings, axis=1)
        else:
            return tf.reduce_mean(vehicles_embeddings, axis=1)
