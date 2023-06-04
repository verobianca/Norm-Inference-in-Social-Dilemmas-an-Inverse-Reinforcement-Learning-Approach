import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm
import os
from tensorflow.keras import optimizers, layers, models, Sequential, losses, Model
import pickle
import environment
import sac

"""
AIRL algorithm. Only the reward function (discriminator) is trained.
"""
class AIRL:
    def __init__(self, save_folder, expert_folder=None, agent=False, agent_folder=None):
        self.env_norm = environment.Environment(2, 'none', True)
        self.env = environment.Environment(2, 'none', False)
        self.sac = sac.Soft_AC('none', 'none', False)
        self.n_outputs = 7
        self.input_shape = (5, 5, 1)

        self.optimizer_policy = keras.optimizers.Adam(lr=1e-3, clipvalue=10.0)
        self.optimizer_cost = keras.optimizers.Adam(lr=1e-3, clipvalue=10.0)

        self.save_folder = save_folder
        self.expert_folder = expert_folder
        self.agent = agent
        self.agent_folder = agent_folder
        if self.agent:
            self.policy = self.load_model(self.agent_folder)
        else:
            self.policy = self.policy_model()

        self.policy = self.policy_model() #self.sac.get_policy()
        self.g = self.g_function()
        self.h = self.h_function()
        self.num_demo = 1
        self.expert_data = [[], [], []]
        self.train_data = [[], [], []]
        self.losses = []

    def load_model(self, checkpoint_path):
        checkpoint_dir = os.path.dirname(checkpoint_path)
        model = tf.keras.models.load_model(checkpoint_dir)

        return model

    def policy_model(self):
        state1 = layers.Input(shape=self.input_shape)
        x1 = layers.Conv2D(filters=3, kernel_size=3, padding='same', activation="relu")(state1)
        x1 = layers.Conv2D(filters=6, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Conv2D(filters=9, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Flatten()(x1)

        x1 = layers.Dense(units=128, input_dim=2, activation="relu")(x1)
        x1 = layers.Dense(units=64, activation="relu")(x1)
        x1 = layers.Dense(units=32, activation="relu")(x1)
        out1 = layers.Dense(self.n_outputs)(x1)

        model = Model(inputs=state1, outputs=out1, name='policy')
        return model

    def g_function(self):
        state1 = layers.Input(shape=self.input_shape)
        ohe_prev_act1 = layers.Input(shape=(self.n_outputs,))
        x1 = layers.Conv2D(filters=3, kernel_size=3, padding='same', activation="relu")(state1)
        x1 = layers.Conv2D(filters=6, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Conv2D(filters=9, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Flatten()(x1)
        x1 = layers.Concatenate(axis=1)([x1, ohe_prev_act1])

        #x1 = layers.Dense(units=128, input_dim=2, activation="relu")(x1)
        x1 = layers.Dense(units=64, activation="relu")(x1)
        x1 = layers.Dense(units=32, activation="relu")(x1)
        x1 = layers.Dense(units=16, activation="relu")(x1)
        out1 = layers.Dense(1)(x1)

        model = Model(inputs=[state1, ohe_prev_act1], outputs=out1, name='cost')

        return model

    def h_function(self):
        state1 = layers.Input(shape=self.input_shape)
        x1 = layers.Conv2D(filters=3, kernel_size=3, padding='same', activation="relu")(state1)
        x1 = layers.Conv2D(filters=6, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Conv2D(filters=9, kernel_size=3, padding='same', activation="relu")(x1)
        x1 = layers.Flatten()(x1)

        #x1 = layers.Dense(units=128, input_dim=2, activation="relu")(x1)
        x1 = layers.Dense(units=64, activation="relu")(x1)
        x1 = layers.Dense(units=32, activation="relu")(x1)
        x1 = layers.Dense(units=16, activation="relu")(x1)
        out1 = layers.Dense(1)(x1)

        model = Model(inputs=state1, outputs=out1, name='cost')

        return model

    def clear_data(self, s, a):
        still = [i for i in range(len(a)-1, -1, -1) if a[i] == 6]
        print(still)
        for elem in still:
            del s[elem]
            del a[elem]

        return s, a

    #  generate expert trajectories (real samples)
    def load_expert(self):
        model = self.load_model(self.expert_folder)
        steps = 1000
        states = []
        actions = []
        next_states = []

        for _ in tqdm(range(50)):
            s1, s2, a1, a2, ns1, ns2 = self.generate_session(model, True, steps)
            states.extend(s1)
            states.extend(s2)
            actions.extend(a1)
            actions.extend(a2)
            next_states.extend(ns1)
            next_states.extend(ns2)

        expert_data = [states, actions, next_states]
        return expert_data

    #  generate external agent or random trajectories (fake sample from the generator)
    def load_train_data(self):
        model = self.policy
        steps = 1000
        states = []
        actions = []
        next_states = []
        for _ in tqdm(range(50)):
            s1, s2, a1, a2, ns1, ns2 = self.generate_session(model, False, steps)
            states.extend(s1)
            states.extend(s2)
            actions.extend(a1)
            actions.extend(a2)
            next_states.extend(ns1)
            next_states.extend(ns2)

        samples = [states, actions, next_states]

        return samples

    def one_hot_encoding(self, acts):
        actions = np.array([np.zeros(self.n_outputs) for _ in acts])
        for i in range(len(acts)):
            actions[i][acts[i]] = 1
        return actions

    #  generate a trajectories
    def generate_session(self, model, expert=False, steps=10, prnt=None):
        if prnt == 'map':
            env = environment.Environment(2, prnt, expert)
        else:
            if expert:
                env = self.env_norm
            else:
                env = self.env

        states, actions, next_states, previous_actions = [], [], [], []
        env.reset()

        sts = env.get_observation()
        poss_acts = env.possible_actions()
        for step in range(steps):
            acts = list()
            for i in range(2):
                state = sts[i]
                poss_act = poss_acts[i]
                action_logits = model(state[np.newaxis])

                if expert:
                    poss_logits = [action_logits[0][i] for i in poss_act]
                    a = tf.random.categorical([poss_logits], 1)[0, 0]
                    action = poss_act[a]
                else:
                    poss_logits = [action_logits[0][i] for i in poss_act]
                    a = tf.random.categorical([poss_logits], 1)[0, 0]
                    action = poss_act[a]

                acts.append(action)

            next_sts, poss_acts, _, done = env.step(acts)
            states.append(sts)
            actions.append(acts)
            next_states.append(next_sts)
            sts = next_sts

            if prnt == 'map':
                env.render(0)

            if done:
                break
        if prnt == 'map':
            env.close_plot()

        return [s[0] for s in states], [s[1] for s in states], [a[0] for a in actions],  [a[1] for a in actions], \
               [ns[0] for ns in next_states], [ns[1] for ns in next_states]

    #  gradient descent step
    def learn(self, train_data, expert_data):
        # calculate loss, here are some hyper-param
        with tf.GradientTape(persistent=True) as tape:
            states, acts, next_states = train_data
            states_exp, acts_exp, next_states_exp = expert_data

            states = np.array(states)
            actions = np.array(self.one_hot_encoding(acts))
            next_states = np.array(next_states)

            states_exp = np.array(states_exp)
            actions_exp = np.array(self.one_hot_encoding(acts_exp))
            next_states_exp = np.array(next_states_exp)

            g_out_exp = self.g([states_exp, actions_exp])
            h_next_exp = self.h(next_states_exp)
            h_exp = self.h(states_exp)
            logits_exp = self.policy(states_exp)
            probs_exp = tf.math.softmax(logits_exp)

            prob_act_exp = [[probs_exp[i][acts_exp[i]] + 1e-8] for i in range(len(acts_exp))]

            g_out = self.g([states, actions])
            h_next = self.h(next_states)
            h = self.h(states)

            logits = self.policy(states)
            probs = tf.math.softmax(logits)
            prob_act = [[probs[i][acts[i]] + 1e-8] for i in range(len(acts))]

            e1 = tf.math.divide(tf.math.exp(g_out_exp + 0.99 * h_next_exp - h_exp),
                                (tf.math.exp(g_out_exp + 0.99 * h_next_exp - h_exp) + prob_act_exp))
            loss_1 = tf.math.log(e1)

            e2 = tf.math.divide(tf.math.exp(g_out + 0.99 * h_next - h),
                                (tf.math.exp(g_out + 0.99 * h_next - h) + prob_act))
            loss_2 = tf.math.log(1 - e2)
            loss = -(tf.math.reduce_mean(loss_1 + loss_2))

        print(loss)
        self.optimizer_cost.build(list(self.g.trainable_variables) + list(self.h.trainable_variables))
        grads_g = tape.gradient(loss, self.g.trainable_variables)
        self.optimizer_cost.apply_gradients(zip(grads_g, self.g.trainable_variables))

        grads_h = tape.gradient(loss, self.h.trainable_variables)
        self.optimizer_cost.apply_gradients(zip(grads_h, self.h.trainable_variables))
        del tape

        return loss

    # calculate the rewards of the two agents from the reward function in a given timestep
    def reward(self, states, acts, next_states, prob_act):
        states = np.array(states)
        actions = np.array(self.one_hot_encoding(acts))
        next_states = np.array(next_states)

        g_out = self.g([states, actions])
        h_next_state = self.h(next_states)
        h_states = self.h(states)

        e1 = tf.math.divide(tf.math.exp(g_out + 0.99 * h_next_state - h_states),
                            (tf.math.exp(g_out + 0.99 * h_next_state - h_states) + prob_act))
        loss_1 = tf.math.log(e1)
        loss_2 = tf.math.log(1 - e1)
        loss = loss_1 - loss_2

        return loss

    def train(self):
        self.expert_data = self.load_expert()
        for ep in tqdm(range(15)):
            loss_ep = []
            new_train = self.load_train_data()
            self.train_data[0].extend(new_train[0])
            self.train_data[1].extend(new_train[1])
            self.train_data[2].extend(new_train[2])

            data = self.train_data
            for _ in tqdm(range(500)):
                batch_size = 32

                demo_trajs_ids = np.random.choice(range(len(self.expert_data[0])), batch_size)

                sample_expert_data = [np.array(self.expert_data[0], dtype=object)[demo_trajs_ids].tolist(), np.array(self.expert_data[1], dtype=object)[demo_trajs_ids].tolist(),
                                      np.array(self.expert_data[2], dtype=object)[demo_trajs_ids].tolist()]

                sampled_trajs_ids = np.random.choice(range(len(data[0])), batch_size)

                sample_train_data = [np.array(data[0], dtype=object)[sampled_trajs_ids].tolist(), np.array(data[1], dtype=object)[sampled_trajs_ids].tolist(),
                                     np.array(data[2], dtype=object)[sampled_trajs_ids].tolist()]

                loss = self.learn(sample_train_data, sample_expert_data)
                loss_ep.append(loss)

            self.losses.append(np.mean(loss_ep))
            self.save_model(ep)

        loss_file = open(r'AIRL_reward_good2/loss/loss.pkl', 'wb')
        pickle.dump(self.losses, loss_file)
        loss_file.close()

    def save_model(self, ep):
        checkpoint_path_g = './AIRL_reward_good2/training_{}/g/cp.ckpt'.format(ep)
        checkpoint_dir_g = os.path.dirname(checkpoint_path_g)
        self.g.save(checkpoint_dir_g)

        checkpoint_path_h = './AIRL_reward_good2/training_{}/h/cp.ckpt'.format(ep)
        checkpoint_dir_h = os.path.dirname(checkpoint_path_h)
        self.h.save(checkpoint_dir_h)


if __name__ == '__main__':
    save_folder = "save_folder"
    expert_folder = "./good/expert2/training_360/"
    agent = True
    agent_folder = "good/external_agent/agent_2/training_410/"
    gcl = AIRL(save_folder, expert_folder, agent, agent_folder)
    gcl.train()






