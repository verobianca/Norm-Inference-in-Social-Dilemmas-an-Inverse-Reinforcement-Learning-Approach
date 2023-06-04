import numpy as np
from collections import deque
from tqdm import tqdm
import os
import pickle
import tensorflow as tf
from tensorflow.keras import optimizers, layers, models, Sequential, losses, Model
import environment
import tensorflow_probability as tfp

"""
Soft Actor-Critic algorithm to train policies on a given reward function
"""


class Soft_AC:
    def __init__(self, folder, print="none", norm=False, agent=False, agent_folder=None, lr_policy=1e-4, lr_qnet=1e-4,
                 bs=32, cv=10.0, df=0.999, C=10000, buf_s=100000, filters=None, neurons=None):
        if neurons is None:
            self.neurons = [128, 64, 32]
        if filters is None:
            self.filters = [3, 6, 9]
        self.number = 2
        self.input_shape = (5, 5, 1)
        self.n_outputs = 7  # ==env.action_space.n
        self.batch_size = bs
        self.discount_factor = df
        self.optimizer_q2_net = optimizers.Adam(lr=lr_qnet, clipvalue=cv, weight_decay=0.05)
        self.optimizer_q1_net = optimizers.Adam(lr=lr_qnet, clipvalue=cv, weight_decay=0.05)
        self.optimizer_policy = optimizers.Adam(lr=lr_policy, clipvalue=cv, weight_decay=0.05)
        self.mse_loss = losses.mean_squared_error
        self.C = C  # update step for target net
        self._log_alpha = tf.Variable(0.0)
        self._alpha = tfp.util.DeferredTensor(self._log_alpha, tf.exp)
        self.optimizer_alpha = tf.optimizers.Adam(3e-4, name='alpha_optimizer')
        self.update = 0
        self.print = print
        self.folder = folder
        self.norm = norm
        self.env = environment.Environment(self.number, self.print, self.norm)
        self.target_entropy = 0.99 * -np.log(1/7)
        self.policy, self.q_net1, self.q_net2, self.t_net1, self.t_net2, self.replay_buffer = self.initialize_nets(buf_s)
        if agent:
            self.policy = self.load_agent()
            self.agent_folder = agent_folder

    def load_agent(self):
        checkpoint_path = self.agent_folder
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_dir)

        return model

    def get_policy(self):
        return self.policy

    def initialize_nets(self, buf_s):
        state1 = layers.Input(shape=self.input_shape)
        x1 = layers.Conv2D(filters=self.filters[0], kernel_size=3, padding='same', activation="relu")(state1)
        for i in self.filters[1:]:
            x1 = layers.Conv2D(filters=i, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Flatten()(x1)
        x1 = layers.Dense(units=self.neurons[0], input_dim=2, activation="relu")(x1)
        for i in self.neurons[1:]:
            x1 = layers.Dense(units=i, activation="relu")(x1)
        out1 = layers.Dense(self.n_outputs)(x1)
        q_net1 = Model(inputs=state1, outputs=out1, name='forward')

        state2 = layers.Input(shape=self.input_shape)
        x2 = layers.Conv2D(filters=self.filters[0], kernel_size=3, padding='same', activation="relu")(state2)
        for i in self.filters[1:]:
            x2 = layers.Conv2D(filters=i, kernel_size=3, padding='same', activation="relu")(x2)
        x2 = layers.Flatten()(x2)
        x2 = layers.Dense(units=self.neurons[0], input_dim=2, activation="relu")(x2)
        for i in self.neurons[1:]:
            x2 = layers.Dense(units=i, activation="relu")(x2)
        out2 = layers.Dense(self.n_outputs)(x2)
        q_net2 = Model(inputs=state2, outputs=out2, name='forward')

        state3 = layers.Input(shape=self.input_shape)
        x3 = layers.Conv2D(filters=self.filters[0], kernel_size=3, padding='same', activation="relu")(state3)
        for i in self.filters[1:]:
            x3 = layers.Conv2D(filters=i, kernel_size=3, padding='same', activation="relu")(x3)
        x3 = layers.Flatten()(x3)
        x3 = layers.Dense(units=self.neurons[0], input_dim=2, activation="relu")(x3)
        for i in self.neurons[1:]:
            x3 = layers.Dense(units=i, activation="relu")(x3)
        out3 = layers.Dense(self.n_outputs)(x3)
        policy = Model(inputs=state3, outputs=out3, name='forward')
        policy.summary()

        target_model1 = models.clone_model(q_net1)
        target_model1.set_weights(q_net1.get_weights())

        target_model2 = models.clone_model(q_net2)
        target_model2.set_weights(q_net2.get_weights())

        replay_buffer = deque(maxlen=buf_s)

        return policy, q_net1, q_net2, target_model1, target_model2, replay_buffer

    def one_hot_encoding(self, acts):
        actions = np.array([np.zeros(self.n_outputs) for _ in acts])
        for i in range(len(acts)):
            actions[i][acts[i]] = 1
        return actions

    # calculate the rewards of the two agents from the reward function in a given timestep
    def reward_func(self, states, acts, next_states, prob_act):
        states = np.array(states)
        actions = np.array(self.one_hot_encoding(acts))
        next_states = np.array(next_states)

        g_out = self.g([states, actions])[:, 0]
        h_next_state = self.h(next_states)[:, 0]
        h_states = self.h(states)[:, 0]

        e1 = tf.math.divide(tf.math.exp(g_out + 0.99 * h_next_state - h_states),
                            (tf.math.exp(g_out + 0.99 * h_next_state - h_states) + prob_act))

        loss_1 = tf.math.log(np.maximum(1e-7, (e1 - 1e-7)))
        loss_2 = tf.math.log(1 - (e1 - 1e-7))
        reward = loss_1 - loss_2

        return reward

    def add_to_replay_buffer(self, states, actions, rewards, next_state, done, next_poss_action):
        for i in range(self.number):
            self.replay_buffer.append([states[i], actions[i], rewards[i], next_state[i], next_poss_action[i], done])

    # start training
    def start(self, ep=100, steps=200, start_training=50, save=10, reward=False, g=None, h=None):
        self.steps = steps
        self.reward = reward
        self.g = g
        self.h = h

        for episode in tqdm(range(1, ep+1)):
            if episode % save == 0:
                self.save_model(episode)

            self.env.reset()
            observations = self.env.get_observation()
            poss_actions = self.env.possible_actions()

            for step in range(self.steps):
                observations, poss_actions, done = self.play_one_step(observations, poss_actions, step)

                if self.print == "map":
                    self.env.render(episode)

                if done:
                    break

                if episode > start_training:
                    loss = self.training_step()
                    if episode % 10 == 0:
                        print(loss)

        return self.policy

    def play_one_step(self, states, poss_actions, step):
        actions = list()
        probs_act = list()
        for i in range(self.number):
            state = states[i]
            poss_act = poss_actions[i]

            action_logits = self.policy(state[np.newaxis])
            poss_logits = [action_logits[0][i] for i in poss_act]
            a = tf.random.categorical([poss_logits], 1)[0, 0]
            action = poss_act[a]

            actions.append(action)
            probs = tf.math.softmax(action_logits)[0]
            prob_act = probs[action]
            probs_act.append(prob_act)

        next_states, next_poss_action, rewards, done = self.env.step(actions)

        if self.reward is True:
            rewards = self.reward_func(states, actions, next_states, probs_act)

        if step == self.steps - 1:
            done = True

        self.add_to_replay_buffer(states, actions, rewards, next_states, done, next_poss_action)
        return next_states, next_poss_action, done

    def training_step(self):
        entropy_scale = tf.convert_to_tensor(self._alpha)
        batch = self.sample_experience()
        states, actions, rewards, next_states, next_poss_actions, dones = [
            np.array([experience[field_index] for experience in batch]) for field_index in range(6)]

        logits = self.policy(next_states)
        log_probs = tf.math.log_softmax(logits)
        probs = tf.math.softmax(logits)
        next_entropy_term = - entropy_scale * log_probs

        next_q_t1 = self.t_net1(next_states)
        next_q_t2 = self.t_net2(next_states)
        target_q_values = rewards + (1 - dones) * self.discount_factor * tf.reduce_sum(probs * (np.minimum(next_q_t1, next_q_t2) + next_entropy_term), 1)

        # update q_nets
        with tf.GradientTape() as tape1:
            q1_values = self.q_net1(states)
            q1_values = tf.convert_to_tensor([q1_values[i, actions[i]] for i in range(len(actions))])
            q1_loss = tf.reduce_mean(self.mse_loss(q1_values, target_q_values))
        q1_grads = tape1.gradient(q1_loss, self.q_net1.trainable_variables)
        self.optimizer_q1_net.apply_gradients(zip(q1_grads, self.q_net1.trainable_variables))

        with tf.GradientTape() as tape2:
            q2_values = self.q_net2(states)
            q2_values = tf.convert_to_tensor([q2_values[i, actions[i]] for i in range(len(actions))])
            q2_loss = tf.reduce_mean(self.mse_loss(q2_values, target_q_values))
        q2_grads = tape2.gradient(q2_loss, self.q_net2.trainable_variables)
        self.optimizer_q2_net.apply_gradients(zip(q2_grads, self.q_net2.trainable_variables))

        # update policy
        with tf.GradientTape() as tape3:
            logits = self.policy(states)
            probs = tf.math.softmax(logits)
            log_probs = tf.math.log_softmax(logits)
            entropy_term = - entropy_scale * log_probs

            q1_val = self.q_net1(states)
            q2_val = self.q_net2(states)

            p_loss = - tf.reduce_mean(tf.reduce_sum(probs * (np.minimum(q1_val, q2_val) + entropy_term), 1))

        p_grads = tape3.gradient(p_loss, self.policy.trainable_variables)
        self.optimizer_policy.apply_gradients(zip(p_grads, self.policy.trainable_variables))

        # update entropy_term
        with tf.GradientTape() as tape4:
            e_loss = - tf.reduce_mean(self._alpha * (log_probs + self.target_entropy))

        e_grads = tape4.gradient(e_loss, [self._log_alpha])
        self.optimizer_alpha.apply_gradients(zip(e_grads, [self._log_alpha]))

        # update targets
        self.update += 1
        if self.update == self.C:
            self.update_target_net(self.q_net1, self.t_net1)
            self.update_target_net(self.q_net2, self.t_net2)

        return p_loss

    def sample_experience(self):
        indices = np.random.randint(len(self.replay_buffer), size=self.batch_size)
        batch = [self.replay_buffer[index] for index in indices]
        return batch

    def update_target_net(self, model, target_model):
        target_model.set_weights(model.get_weights())

    def save_model(self, episode):
        checkpoint_path = "./{}/agent_{}/training_{}/cp.ckpt".format(self.folder, self.number, episode)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        self.policy.save(checkpoint_dir)



