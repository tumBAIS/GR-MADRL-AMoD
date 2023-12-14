"""Training loop incl. validation and testing"""

import os
import copy
import pandas as pd
import numpy as np
import tensorflow as tf

from replay_buffer import ReplayBuffer

np.random.seed(0)


class Trainer:
    def __init__(self, policy, env, args):
        self.policy = policy
        self.env = env
        
        self.episode_max_steps = int(args["episode_length"]/args["time_step_size"]) # no. of steps per episode
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.max_steps = args["max_steps"]
        self.min_steps = args["min_steps"]
        self.random_steps = args["random_steps"]
        self.update_interval = args["update_interval"]
        self.validation_interval = args["validation_interval"]
        self.tracking_interval = args["tracking_interval"]
        self.profile_interval = args["profile_interval"]
        self.rb_size = args["rb_size"]        
        self.batch_size = args["batch_size"]
        self.scheduled_discount = args["scheduled_discount"]
        self.scheduled_discount_values = args["scheduled_discount_values"]
        self.scheduled_discount_steps = args["scheduled_discount_steps"]
        self.normalized_rews = args["normalized_rews"]
        self.data_dir = args["data_dir"]
        self.results_dir = args["results_dir"]
        self.validation_episodes = len(pd.read_csv(self.data_dir + '/validation_dates.csv').validation_dates.tolist())

        self.struct_analysis = args["struct_analysis"]

        # save arguments and environment variables
        with open(self.results_dir + '/args.txt', 'w') as f: f.write(str(args))
        with open(self.results_dir + '/environ.txt', 'w') as f: f.write(str(dict(os.environ)))

        # initialize model saving and potentially restore saved model
        self.set_check_point(args["model_dir"])

        # prepare TensorBoard output
        self.writer = tf.summary.create_file_writer(self.results_dir)
        self.writer.set_as_default()

        # For myopic policy rvaluation
        self.zones = pd.read_csv(self.data_dir + '/zones.csv', header=0, index_col=0)
        self.cost_parameter = args["cost_parameter"]

    # initialize model saving and potentially restore saved model
    def set_check_point(self, model_dir):
        self.checkpoint = tf.train.Checkpoint(policy=self.policy)
        self.checkpoint_manager = tf.train.CheckpointManager(self.checkpoint, directory=self.results_dir, max_to_keep=1)

        if model_dir is not None:
            assert os.path.isdir(model_dir)
            latest_path_ckpt = tf.train.latest_checkpoint(model_dir)
            self.checkpoint.restore(latest_path_ckpt)

    def __call__(self):   
        total_steps = 0
        episode_steps = 0
        episode_reward = 0.
        self.validation_rewards = []
        self.ckpt_id = 0
        if self.scheduled_discount: discount_iter = 0
        profiling_start_step = self.profile_interval

        replay_buffer = ReplayBuffer(self.rb_size, self.normalized_rews, self.n_veh, self.n_req_max)

        state, hvs = self.env.reset()

        while total_steps < self.max_steps:
            if (total_steps + 1) % 100 == 0: tf.print("Started step", total_steps+1)
            
            if self.scheduled_discount:
                if self.scheduled_discount_steps[discount_iter] == total_steps:
                    self.policy.discount.assign(self.scheduled_discount_values[discount_iter])
                    discount_iter += 1
                    if discount_iter == len(self.scheduled_discount_values):
                        discount_iter = 0
            
            if (total_steps + 1) % self.profile_interval == 0:
                profiling_start_step = total_steps
                tf.profiler.experimental.start(self.results_dir)
            
            with tf.profiler.experimental.Trace('train', step_num=total_steps, _r=1):
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                elif total_steps < self.random_steps:
                    request_masks = self.policy.get_masks(tf.expand_dims(state["requests_state"], axis=0))
                    action = self.policy.actor.get_random_action(state, hvs, request_masks["l"])
                else:
                    action = self.policy.get_action(state, hvs)
                
                next_state, reward, next_hvs = self.env.step(action)
    
                if ~tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    next_state_adapted = copy.deepcopy(next_state)
                    next_state_adapted["requests_state"] = state["requests_state"] # replace new requests with old requests (solution for dimensionality problem in critic loss)
                    if self.normalized_rews:
                        rew_mask = tf.squeeze(self.policy.get_masks(tf.expand_dims(state["requests_state"], axis=0))["l"], axis=0)
                        replay_buffer.add(obs=state, hvs=hvs, act=action, rew=reward, next_obs=next_state_adapted, next_hvs=next_hvs, mask=rew_mask)
                    else:
                        replay_buffer.add(obs=state, hvs=hvs, act=action, rew=reward, next_obs=next_state_adapted, next_hvs=next_hvs)
    
                state = next_state
                hvs = next_hvs
                total_steps += 1
                episode_steps += 1
                episode_reward += tf.reduce_sum(reward).numpy()
                
                tf.summary.experimental.set_step(total_steps)            
    
                if total_steps >= self.min_steps and total_steps % self.update_interval == 0:
                    states, hvses, acts, rews, next_states, next_hvses, nonzero_obs = replay_buffer.sample(self.batch_size)
                    with tf.summary.record_if(total_steps % self.tracking_interval == 0):
                        self.policy.train(states, hvses, acts, rews, next_states, next_hvses, nonzero_obs)

            if total_steps % self.validation_interval == 0:
                avg_validation_reward = self.validate_policy()
                tf.summary.scalar(name="validation_reward", data=avg_validation_reward)
                self.validation_rewards.append(avg_validation_reward)
                
                if avg_validation_reward == tf.reduce_max(self.validation_rewards).numpy():
                    self.ckpt_id += 1
                    self.checkpoint_manager.save()

            if episode_steps == self.episode_max_steps:
                tf.summary.scalar(name="training_reward", data=episode_reward)
                episode_steps = 0
                episode_reward = 0.
                state, hvs = self.env.reset()

            if total_steps == profiling_start_step + 5: tf.profiler.experimental.stop() # stop profiling 5 steps after start

        tf.summary.flush()
                
        self.test_policy()
        
        tf.print("Finished")

    # compute average reward per validation episode achieved by current policy
    def validate_policy(self):
        validation_reward = 0.
        
        for i in range(self.validation_episodes):
            state, hvs = self.env.reset(validation=True)
            
            for j in range(self.episode_max_steps):
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                else:
                    action = self.policy.get_action(state, hvs, test=tf.constant(True))
                
                next_state, reward, hvs = self.env.step(action)
                
                validation_reward += tf.reduce_sum(reward).numpy()
                
                state = next_state
        
        avg_validation_reward = validation_reward / self.validation_episodes
        
        self.env.remaining_validation_dates = copy.deepcopy(self.env.validation_dates) # reset list of remaining validation dates

        return avg_validation_reward

    # compute rewards per test episode with best policy
    def test_policy(self):

        self.checkpoint.restore(self.results_dir + f"/ckpt-{self.ckpt_id}")
        
        test_dates = pd.read_csv(self.data_dir + '/test_dates.csv').test_dates.tolist()
        
        test_rewards = []
        
        if self.struct_analysis:
            rej_rates_to = []
            cond_rejections_to = []
            ratio_op_to = []
        
        for i in range(len(test_dates)):

            test_reward = 0.
            
            state, hvs = self.env.reset(testing=True)

            if self.struct_analysis:
                profit_req_ep = []
                rejections_ep = []
                cond_rejections_ep = []
                forecasting_steps = 10
                self.forecasting_buffer = np.zeros((forecasting_steps+1, 3, len(self.zones)), dtype=np.float32) # 3 columns: rejection indicator, original profit, theoretical profit
                rej_op_ep = []
                acc_op_ep = []
            
            for j in range(self.episode_max_steps):
                if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                    action = -tf.ones(self.n_req_max, tf.int32)
                else:
                    action = self.policy.get_action(state, hvs, test=tf.constant(True))
                hvs_old = hvs
                
                next_state, reward, hvs = self.env.step(action)
                
                if self.struct_analysis:
                    # get raw measures for the structural analysis
                    if tf.reduce_all(state["requests_state"] == tf.zeros([self.n_req_max,5])):
                        profitable_requests, rejections, cond_rejection_rates, profit = tf.zeros(self.n_req_max, dtype=tf.int32), tf.zeros(self.n_req_max, dtype=tf.int32), [[0., 0.]]*2, tf.zeros(self.n_req_max, dtype=tf.float32)
                    else:
                        profitable_requests, rejections, cond_rejection_rates, profit = self.policy.actor.get_struct_analysis(state, tf.expand_dims(hvs_old, axis=0), action)
                    
                    # get measures for testing prediction power
                    rejected_overprofit, accepted_overprofit = self.overprofit_calc(j, forecasting_steps+1, rejections, profit, state)

                    # calculate measures for the current step
                    profit_req_ep.append(tf.reduce_sum(profitable_requests).numpy())
                    rejections_ep.append(tf.reduce_sum(rejections).numpy())
                    cond_rejections_ep.append(cond_rejection_rates)
                    rej_op_ep.append(rejected_overprofit)
                    acc_op_ep.append(accepted_overprofit)
                
                test_reward += tf.reduce_sum(reward).numpy()
                
                state = next_state
            
            # calculate measures for the curent episode
            if self.struct_analysis:
                rej_rates_to.append(np.nansum(rejections_ep) / np.nansum(profit_req_ep))
                aggregate_rejections = np.nansum(cond_rejections_ep, axis=0)
                cond_rejections_to.append(list(aggregate_rejections[:,0] / aggregate_rejections[:,1]))
                if np.nansum(acc_op_ep) == 0: ratio_op_to.append(np.nan)
                else: ratio_op_to.append(np.nanmean(rej_op_ep) / np.nanmean(acc_op_ep))
            
            test_rewards.append(test_reward)
                
        pd.DataFrame({"test_rewards_RL": test_rewards}, index=test_dates).to_csv(self.results_dir + "/test_rewards.csv")
        with open(self.results_dir + "/avg_test_reward.txt", 'w') as f: f.write(str(np.mean(test_rewards)))
        
        # summatize and store measures of structural analysis
        if self.struct_analysis:
            analysis_results = [] # initialize empty array
            analysis_results.append(np.nanmean(rej_rates_to)) # mean rejection rate
            analysis_results.append(list(np.nanmean(cond_rejections_to, axis=0))) # mean rejection rates depending on number of vehicles in destination
            analysis_results.append(np.nanmean(ratio_op_to)) # mean ratio of overperformance after rejected versus accepted requests

            pd.DataFrame({"structural analysis test results": analysis_results}).to_csv(self.results_dir + "/structural_analsysis_test_results.csv")
    
    def overprofit_calc(self, j, fs, rejections, profit, state):
        
        # Calculate overprofits after rejection and acceptance
        if j < fs:
            rejected_overprofit, accepted_overprofit = np.nan, np.nan
        else:
            considered_forecast = self.forecasting_buffer[j%fs,:,:]
            rejected_overprofit = np.sum(considered_forecast[0,:] * considered_forecast[2,:]) / np.sum(considered_forecast[0,:]) if np.sum(considered_forecast[0,:]) != 0 else np.nan
            accepted_overprofit = np.where(considered_forecast[1,:] > 0, 1, 0) * (1-considered_forecast[0,:])
            accepted_overprofit = np.sum(accepted_overprofit * considered_forecast[2,:]) / np.sum(accepted_overprofit) if np.sum(accepted_overprofit) != 0 else np.nan
        
        # Calculate theoretical profits
        requests_state = state["requests_state"]
        req_distance = requests_state[:,4] * self.env.max_distance
        pickup_horizontal = tf.cast(tf.math.round(requests_state[:,0] * self.env.max_horizontal_idx), tf.int32)
        pickup_vertical = tf.cast(tf.math.round(requests_state[:,1] * self.env.max_vertical_idx), tf.int32)
        req_pickup = self.env.zone_mapping_table[self.env.get_keys(pickup_horizontal, pickup_vertical)]
        req_pickup = req_pickup.numpy()
        destination_horizontal = tf.cast(tf.math.round(requests_state[:,2] * self.env.max_horizontal_idx), tf.int32)
        destination_vertical = tf.cast(tf.math.round(requests_state[:,3] * self.env.max_vertical_idx), tf.int32)
        req_destination = self.env.zone_mapping_table[self.env.get_keys(destination_horizontal, destination_vertical)]
        req_destination = req_destination.numpy()
        fares = []
        for i in range(self.n_req_max):
            if req_destination[i] - req_pickup[i] == 0: fares.append(0)
            else: fares.append(self.env.graph.loc[(req_pickup[i], req_destination[i]), "fare"])
        theoretical_profit = fares + self.cost_parameter * req_distance.numpy()

        # Add theoretical profits to forecasting buffer
        self.forecasting_buffer[j%fs,:,:] = 0
        for i in range(len(self.zones)):
            requests_cond = np.where(req_pickup==i, 1., 0.)
            self.forecasting_buffer[j%fs,0,i] = tf.reduce_max(requests_cond*tf.cast(rejections, dtype=tf.float32))
            self.forecasting_buffer[j%fs,1,i] = tf.reduce_sum(requests_cond*profit)
            for k in range(fs):
                overprofit = np.sum(requests_cond * theoretical_profit) - self.forecasting_buffer[k,1,i]
                self.forecasting_buffer[k,2,i] += overprofit if overprofit > 0 else 0
            self.forecasting_buffer[j%fs,2,i] = 0
        
        return rejected_overprofit, accepted_overprofit