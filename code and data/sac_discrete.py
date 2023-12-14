"""Set up networks and define one training iteration"""

import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers.schedules import PolynomialDecay
from tensorflow.keras.mixed_precision import LossScaleOptimizer

from actor import Actor
from critic import Critic

class SACDiscrete(tf.keras.Model):
    def __init__(self, args, env):
        super().__init__()
        
        self.n_veh = args["veh_count"]
        self.n_req_max = args["max_req_count"]
        self.batch_size = args["batch_size"]
        self.alpha = tf.exp(tf.constant(args["log_alpha"]))
        self.tau = tf.constant(args["tau"])
        self.huber_delta = tf.constant(args["huber_delta"])
        self.gradient_clipping = tf.constant(args["gradient_clipping"])
        self.clip_norm = tf.constant(args["clip_norm"])
        self.discount = tf.Variable(args["discount"], dtype=tf.float32)

        # For COMA^adj and COMA^scd
        self.beta_param = args["beta_param"]
        if self.beta_param < 0:
            self.beta_param = 0
            print("Beta parameter must be positive or zero.")
        max_steps = args["max_steps"]
        min_steps = args["min_steps"]
        update_interval = args["update_interval"]
        update_steps_max = (max_steps-min_steps)/update_interval + 1
        self.scal_fac = 1/(update_steps_max**self.beta_param)
        self.update_step = 1 # Count number of training episodes

        self.algorithm_type = args["algorithm_type"]
        self.xi = args["xi"] # For target actor network algorithms (COMA^tgt, COMA^adj, COMA^scd)
        self.avg_rew = args["avg_rew"]
        self.share_glob_rew = args["share_glob_rew"] # For LRGA
        if self.share_glob_rew < 0:
            self.share_glob_rew = 0
            print("Share of global rewards must be in range [0,1].")
        if self.share_glob_rew > 1:
            self.share_glob_rew = 1
            print("Share of global rewards must be in range [0,1].")

        self.actor = Actor(args, env)
        if self.algorithm_type in {"COMA^tgt", "COMA^adj", "COMA^scd"}: self.policy_target = Actor(args, env) # For models with target actor networks (COMA^tgt, COMA^adj, COMA^scd)
        self.qf1 = Critic(args, env, name="qf1")
        self.qf2 = Critic(args, env, name="qf2")
        self.qf1_target = Critic(args, env, name="qf1_target")
        self.qf2_target = Critic(args, env, name="qf2_target")

        lr_a = args["lr_a"]
        lr_c = args["lr_c"]
        self.actor_optimizer = LossScaleOptimizer(Adam(lr_a))
        self.qf1_optimizer = LossScaleOptimizer(Adam(lr_c))
        self.qf2_optimizer = LossScaleOptimizer(Adam(lr_c))
        
        self.q1_update = tf.function(self.q_update)
        self.q2_update = tf.function(self.q_update)

        # For COMA^scd
        if self.algorithm_type == "COMA^scd":
            self.qf1_ego = Critic(args, env, name="qf1_ego")
            self.qf2_ego = Critic(args, env, name="qf2_ego")
            self.qf1_ego_target = Critic(args, env, name="qf1_ego_target")
            self.qf2_ego_target = Critic(args, env, name="qf2_ego_target")
            self.q1_ego_update = tf.function(self.q_update)
            self.q2_ego_update = tf.function(self.q_update)
            self.qf1_ego_optimizer = LossScaleOptimizer(Adam(lr_c))
            self.qf2_ego_optimizer = LossScaleOptimizer(Adam(lr_c))
            self.kappa_exponent = args["kappa_exponent"]
            if self.kappa_exponent < 0:
                self.kappa_exponent = 0
                print("Kappa exponent must be positive or zero.")
            self.kappa_jump = args["kappa_jump"]
            if self.kappa_jump < 0:
                self.kappa_jump = 0
                print("Kappa jump must be in range [0,1]")
            if self.kappa_jump > 1:
                self.kappa_jump = 1
                print("Kappa jump must be in range [0,1]")
            if self.kappa_jump == 0: # COMA^scd with linear or power function schedule
                self.kappa_factor = 1/(update_steps_max**self.kappa_exponent) # Use update_steps_max from above
            else: # COMA^scd with jump schedule
                self.kappa_jump_step = self.kappa_jump*(update_steps_max) # Use update_steps_max from above

    # get action from actor network for state input without batch dim
    def get_action(self, state, hvs, test=tf.constant(False)):
        state, request_masks = self.get_action_body(state)
        return self.actor(state, tf.expand_dims(hvs, axis=0), test, request_masks)
    
    @tf.function
    def get_action_body(self, state):
        requests_state = state["requests_state"]
        vehicles_state = state["vehicles_state"]
        misc_state = state["misc_state"]
        
        requests_state = tf.expand_dims(requests_state, axis=0)
        vehicles_state = tf.expand_dims(vehicles_state, axis=0)
        misc_state = tf.expand_dims(misc_state, axis=0)
        
        state = {"requests_state": requests_state,
                 "vehicles_state": vehicles_state,
                 "misc_state": misc_state}
                
        return state, self.get_masks(requests_state)

    # request masks of shape (batch size, n_req_max) and (batch size, n_req_max * n_veh)
    @tf.function
    def get_masks(self, requests_state):
        request_mask_s = tf.cast(tf.reduce_sum(requests_state, axis=2) > 0, tf.float32)
        request_mask_l = tf.repeat(request_mask_s, repeats=self.n_veh, axis=1)
        
        request_masks = {"s": tf.stop_gradient(request_mask_s),
                         "l": tf.stop_gradient(request_mask_l)}
        
        return request_masks

    # define one training iteration for a batch of experience
    def train(self, states, hvses, actions, rewards, next_states, next_hvses, nonzero_obs):
        request_masks = self.get_masks(states["requests_state"])
        
        if self.algorithm_type == "LRGA": # For static local-global mix
            if self.avg_rew: rewards_glob = rewards/nonzero_obs
            else: rewards_glob = rewards
            rewards_glob = tf.reduce_sum(rewards_glob, axis=1, keepdims=True) # Sum rewards over agents per batch to get global rewards
            rewards = rewards*(1-self.share_glob_rew) + rewards_glob*self.share_glob_rew
        elif self.algorithm_type == "COMA^scd": # For scheduled COMA
            rewards_ego = tf.where(tf.math.is_nan(rewards), 0, rewards) # Get ego rewards and NaN eliminator
            if self.avg_rew: rewards /= nonzero_obs
            rewards = tf.reduce_sum(rewards, axis=1, keepdims=True)
        elif self.algorithm_type != "LRA": # For all other global rewards-based methods
            if self.avg_rew:
                rewards /= nonzero_obs
                rewards = tf.reduce_sum(rewards, axis=1, keepdims=True)
            else:
                rewards = tf.reduce_sum(rewards, axis=1, keepdims=True)
        
        if tf.math.abs(tf.math.reduce_max(rewards)) > 10000:
            print(f'extreme reward: reward of {rewards}')
            print(f'nonzero obs of: {nonzero_obs}')

        rewards = tf.where(tf.math.is_nan(rewards), 0, rewards) # NaN eliminator
                
        cur_act_prob = self.actor.compute_prob(states, hvses, request_masks["s"])
        actions_current_policy = self.actor.post_process(cur_act_prob, tf.constant(False), hvses, request_masks["l"])
        
        target_q = self.target_Qs(rewards, next_states, request_masks, next_hvses)

        if self.algorithm_type == "COMA^scd":
            target_q_ego = self.target_Qs_scd(rewards_ego, next_states, request_masks, next_hvses)
        else:
            target_q_ego = 0
        
        q1_loss, q2_loss, policy_loss, mean_ent, cur_act_logp = self.train_body(states, hvses, actions, target_q, request_masks, actions_current_policy, target_q_ego)

        # Logs for algorithm testing        
        cur_q1 = self.qf1(states, actions_current_policy, hvses, request_masks["s"])
        cur_q2 = self.qf2(states, actions_current_policy, hvses, request_masks["s"])
        if tf.math.count_nonzero(rewards) == 0: avg_reward = tf.reduce_sum(rewards).numpy()
        else: avg_reward = tf.reduce_sum(rewards).numpy()/tf.math.count_nonzero(rewards).numpy()
        tf.summary.scalar(name="critic_loss", data=(q1_loss + q2_loss) / 2.)
        tf.summary.scalar(name="actor_loss", data=policy_loss)
        tf.summary.scalar(name="mean_ent", data=mean_ent)
        tf.summary.scalar(name="logp_mean", data=tf.reduce_mean(cur_act_logp))
        tf.summary.scalar(name="avg_nonzero_training_reward", data=avg_reward)
        tf.summary.scalar(name="average_Qs", data=tf.reduce_mean(tf.minimum(cur_q1, cur_q2)).numpy())

    @tf.function
    def train_body(self, states, hvses, actions, target_q, request_masks, actions_current_policy, target_q_ego):
        cur_q1 = self.qf1(states, actions_current_policy, hvses, request_masks["s"])
        cur_q2 = self.qf2(states, actions_current_policy, hvses, request_masks["s"])
        
        indices = tf.one_hot(actions, depth=self.n_veh, dtype=tf.int32) # indices that can be used to get Q(s,a) for correct a from Q(s), which is a vector with Q(s,a) for all possible a
        indices = tf.stop_gradient(tf.reshape(indices, shape=(self.batch_size, self.n_veh*self.n_req_max)))
        
        q1_loss = self.q1_update(states, hvses, actions, indices, target_q, self.qf1, self.qf1_optimizer, self.qf1_target, request_masks)
        q2_loss = self.q2_update(states, hvses, actions, indices, target_q, self.qf2, self.qf2_optimizer, self.qf2_target, request_masks)

        if self.algorithm_type == "COMA^scd":
            cur_q1_ego = self.qf1_ego(states, actions_current_policy, hvses, request_masks["s"])
            cur_q2_ego = self.qf2_ego(states, actions_current_policy, hvses, request_masks["s"])
            self.q1_ego_update(states, hvses, actions, indices, target_q_ego, self.qf1_ego, self.qf1_ego_optimizer, self.qf1_ego_target, request_masks)
            self.q2_ego_update(states, hvses, actions, indices, target_q_ego, self.qf2_ego, self.qf2_ego_optimizer, self.qf2_ego_target, request_masks)
        else:
            cur_q1_ego = 0
            cur_q2_ego = 0

        policy_loss, cur_act_prob, cur_act_logp = self.actor_update(states, hvses, indices, request_masks, cur_q1, cur_q2, cur_q1_ego, cur_q2_ego)

        mean_ent = self.compute_mean_ent(cur_act_prob, cur_act_logp, request_masks["l"]) # Mean entropy (info for summary output, not needed for algorithm)
        
        return q1_loss, q2_loss, policy_loss, mean_ent, cur_act_logp

    def target_Qs(self, rewards, next_states, request_masks, next_hvses):
        next_act_prob = self.actor.compute_prob(next_states, next_hvses, request_masks["s"])
        next_actions = self.actor.post_process(next_act_prob, tf.constant(False), next_hvses, request_masks["l"])
        return self.target_Qs_body(rewards, next_states, next_hvses, next_actions, next_act_prob, request_masks["s"])
    
    @tf.function
    def target_Qs_body(self, rewards, next_states, next_hvses, next_actions, next_act_prob, request_mask_s):
        next_q1_target = self.qf1_target(next_states, next_actions, next_hvses, request_mask_s)
        next_q2_target = self.qf2_target(next_states, next_actions, next_hvses, request_mask_s)
        next_q = tf.minimum(next_q1_target, next_q2_target)
        
        next_action_logp = tf.math.log(next_act_prob + 1e-8)
        target_q = tf.einsum('ijk,ijk->ij', next_act_prob, next_q - self.alpha * next_action_logp)
        return tf.stop_gradient(rewards + self.discount * target_q)        
    
    def target_Qs_scd(self, rewards, next_states, request_masks, next_hvses):
        next_act_prob = self.actor.compute_prob(next_states, next_hvses, request_masks["s"])
        next_actions = self.actor.post_process(next_act_prob, tf.constant(False), next_hvses, request_masks["l"])
        return self.target_Qs_body_scd(rewards, next_states, next_hvses, next_actions, next_act_prob, request_masks["s"])
    
    @tf.function
    def target_Qs_body_scd(self, rewards, next_states, next_hvses, next_actions, next_act_prob, request_mask_s):
        next_q1_target = self.qf1_ego_target(next_states, next_actions, next_hvses, request_mask_s)
        next_q2_target = self.qf2_ego_target(next_states, next_actions, next_hvses, request_mask_s)
        next_q = tf.minimum(next_q1_target, next_q2_target)
        
        next_action_logp = tf.math.log(next_act_prob + 1e-8)
        target_q = tf.einsum('ijk,ijk->ij', next_act_prob, next_q - self.alpha * next_action_logp)
        return tf.stop_gradient(rewards + self.discount * target_q)   

    def q_update(self, states, hvses, actions, indices, target_q, qf, qf_optimizer, qf_target, request_masks):
        with tf.GradientTape() as tape:
            cur_q = qf(states, actions, hvses, request_masks["s"]) # gives Q(s) for all a, not Q(s,a) for one a
            cur_q_selected = tf.gather_nd(cur_q, tf.expand_dims(indices, axis=2), batch_dims=2) # get correct Q(s,a) from Q(s)

            q_loss = self.huber_loss(target_q - cur_q_selected, self.huber_delta)
            q_loss = q_loss * request_masks["l"]
            q_loss = tf.reduce_mean(tf.reduce_sum(q_loss, axis=1)) # sum over agents and expectation over batch

            regularization_loss = tf.reduce_sum(qf.losses)
            scaled_q_loss = qf_optimizer.get_scaled_loss(q_loss + regularization_loss)
        
        scaled_gradients = tape.gradient(scaled_q_loss, qf.trainable_weights)
        gradients = qf_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        qf_optimizer.apply_gradients(zip(gradients, qf.trainable_weights))
        
        for target_var, source_var in zip(qf_target.weights, qf.weights):
            target_var.assign(self.tau * source_var + (1. - self.tau) * target_var)
        
        return q_loss

    @tf.function
    def huber_loss(self, x, delta):
        delta = tf.ones_like(x) * delta
        less_than_max = 0.5 * tf.square(x) # MSE
        greater_than_max = delta * (tf.abs(x) - 0.5 * delta) # linear
        return tf.where(tf.abs(x)<=delta, x=less_than_max, y=greater_than_max) # MSE for -delta < x < delta, linear otherwise

    @tf.function
    def actor_update(self, states, hvses, indices, request_masks, cur_q1, cur_q2, cur_q1_ego, cur_q2_ego): #[COMA] Contains baseline and loss function calculations
        with tf.GradientTape() as tape:
            cur_act_prob = self.actor.compute_prob(states, hvses, request_masks["s"])
            cur_act_logp = tf.math.log(cur_act_prob + 1e-8)

            min_q = tf.stop_gradient(tf.minimum(cur_q1, cur_q2)) # calculate vector for minimum Q-values (helpful for baselines and loss functions)

            # COMA baselines
            if self.algorithm_type == "COMA^nve": # naive COMA
                baseline = tf.tile(tf.expand_dims(tf.einsum('ijk,ijk->ij', cur_act_prob, min_q), 2), [1,1,2])
            elif self.algorithm_type == "COMA^equ": # COMA with equally-weighted baseline
                baseline = tf.tile(tf.expand_dims(tf.reduce_mean(min_q, axis=2), 2), [1,1,2])
            elif self.algorithm_type == "COMA^tgt": # COMA with target actor network baseline
                target_prob = self.policy_target.compute_prob(states, hvses, request_masks["s"])
                baseline = tf.tile(tf.expand_dims(tf.einsum('ijk,ijk->ij', target_prob, min_q), 2), [1,1,2])
            elif self.algorithm_type in {"COMA^adj", "COMA^scd"}: # adjusted COMA and scheduled COMA
                target_prob = self.policy_target.compute_prob(states, hvses, request_masks["s"])
                baseline_target = tf.tile(tf.expand_dims(tf.einsum('ijk,ijk->ij', target_prob, min_q), 2), [1,1,2])
                baseline_equal = tf.tile(tf.expand_dims(tf.reduce_mean(min_q, axis=2), 2), [1,1,2])
                weight_target = self.scal_fac*(self.update_step**self.beta_param)
                baseline = (1-weight_target)*baseline_equal + weight_target*baseline_target

            # Loss functions
            if self.algorithm_type in {"LRA", "GRA", "LRGA"}: # local rewards, global rewards without baseline, static local-global mix
                policy_loss = tf.einsum('ijk,ijk->ij', cur_act_prob, self.alpha * cur_act_logp - min_q)
            else: # COMA versions
                policy_loss = tf.einsum('ijk,ijk->ij', cur_act_prob, self.alpha * cur_act_logp - min_q + baseline)
            
            # Loss function of COMA^scd
            if self.algorithm_type == "COMA^scd":
                min_q_ego = tf.stop_gradient(tf.minimum(cur_q1_ego, cur_q2_ego))
                ego_policy_loss = tf.einsum('ijk,ijk->ij', cur_act_prob, self.alpha * cur_act_logp - min_q_ego)
                if self.kappa_jump == 0: # COMA^scd with linear or power function schedule
                    weight_COMA = self.kappa_factor*(self.update_step**self.kappa_exponent)
                    policy_loss = (1-weight_COMA)*ego_policy_loss + weight_COMA*policy_loss
                else: # COMA^scd with jump schedule
                    if self.update_step < self.kappa_jump_step: policy_loss = ego_policy_loss
            
            self.update_step += 1 # For COMA^adj and COMA^scd, counts number of training episodes
            
            policy_loss = policy_loss * request_masks["l"]
            policy_loss = tf.reduce_mean(tf.reduce_sum(policy_loss, axis=1)) # sum over agents and expectation over batch

            regularization_loss = tf.reduce_sum(self.actor.losses)
            scaled_loss = self.actor_optimizer.get_scaled_loss(policy_loss + regularization_loss)

        scaled_gradients = tape.gradient(scaled_loss, self.actor.trainable_weights)
        gradients = self.actor_optimizer.get_unscaled_gradients(scaled_gradients)
        if self.gradient_clipping:
            gradients, _ = tf.clip_by_global_norm(gradients, self.clip_norm)
        self.actor_optimizer.apply_gradients(zip(gradients, self.actor.trainable_weights))

        # Target actor network update (for COMA^tgt, COMA^adj, COMA^scd)
        if self.algorithm_type in {"COMA^tgt", "COMA^adj", "COMA^scd"}:
            for target_var, source_var in zip(self.policy_target.weights, self.actor.weights):
                target_var.assign(self.xi * source_var + (1. - self.xi) * target_var)
        
        return policy_loss, cur_act_prob, cur_act_logp

    @tf.function
    def compute_mean_ent(self, cur_act_prob, cur_act_logp, request_mask_l):
        mean_ent = -tf.einsum('ijk,ijk->ij', cur_act_prob, cur_act_logp)
        mean_ent = mean_ent * request_mask_l
        return tf.reduce_sum(mean_ent) / tf.reduce_sum(request_mask_l) # mean over agents and batch

