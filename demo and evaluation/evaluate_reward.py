import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import pickle
import environment
import seaborn as sns
import matplotlib.pyplot as plt

"""
Evaluation of the performance of the retrieved reward function.
The output plots are explained in Fig. 4.4 of the dissertation.
"""


def load_reward(ep, expert=True):
    if expert:
        checkpoint_path = 'good/AIRL_reward_not_sustainable/training_{}/g/cp.ckpt'.format(ep)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cost = tf.keras.models.load_model(checkpoint_dir)
        checkpoint_path = 'good/AIRL_reward_not_sustainable/training_{}/h/cp.ckpt'.format(ep)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        h = tf.keras.models.load_model(checkpoint_dir)
    else:
        checkpoint_path = 'AIRL_reward_no_norm_not_random/training_{}/g/cp.ckpt'.format(ep)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cost = tf.keras.models.load_model(checkpoint_dir)
        checkpoint_path = 'AIRL_reward_no_norm_not_random/training_{}/h/cp.ckpt'.format(ep)
        checkpoint_dir = os.path.dirname(checkpoint_path)
        h = tf.keras.models.load_model(checkpoint_dir)
    return cost, h


def load_agent():
    checkpoint_path = "./good/real_external_agent/agent_2/training_410/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)

    return model


# calculate the rewards of the two agents from the reward function in a given timestep
def reward_func(states, acts, next_states, prob_act, g, h):
    states = np.array(states)
    actions = np.array(one_hot_encoding(acts))
    next_states = np.array(next_states)
    g_out = g([states, actions])[:, 0]
    h_next_state = h(next_states)[:, 0]
    h_states = h(states)[:, 0]

    e1 = tf.math.divide(tf.math.exp(g_out + 0.99 * h_next_state - h_states),
                        (tf.math.exp(g_out + 0.99 * h_next_state - h_states) + prob_act))
    loss_1 = tf.math.log(e1)

    loss_2 = tf.math.log(1 - e1)
    reward = loss_1 - loss_2

    return reward


# generate a full episode
def generate_session(model, steps, env, g, h, random):
    states, actions, next_states, agent_pos, rewards = [], [], [], [], []
    env.reset()
    sts = env.get_observation()
    poss_acts = env.possible_actions()

    for step in range(steps):
        acts = list()
        positions = list()
        for i in range(2):
            poss_act = poss_acts[i]

            if random:
                action = np.random.choice(poss_act, 1)
            else:
                action_logits = model(sts[i][np.newaxis])

                poss_logits = [action_logits[0][i] for i in poss_act]
                a = tf.random.categorical([poss_logits], 1)[0, 0]
                action = poss_act[a]
            acts.append(action)
            pos = env.get_pos(i)

            positions.append(pos)

        next_sts, poss_acts, _, done = env.step(acts)
        reward = reward_func(sts, acts, next_sts, 1 / 7, g, h)

        states.append(sts)
        actions.append(acts)
        next_states.append(next_sts)
        agent_pos.append(positions)

        rewards.append(reward)
        sts = next_sts

        if done:
            break

    return [s[0] for s in states], [s[1] for s in states], [a[0] for a in actions],  [a[1] for a in actions], \
           [ns[0] for ns in next_states], [ns[1] for ns in next_states], [ap[0] for ap in agent_pos], [ap[1] for ap in agent_pos],\
            [r[0] for r in rewards], [r[1] for r in rewards]


def generate_comparison(model, steps, env, g1, h1, g2, h2, random):
    states, actions, next_states, agent_pos, rewards1, rewards2 = [], [], [], [], [], []
    env.reset()
    sts = env.get_observation()
    poss_acts = env.possible_actions()

    for step in range(steps):
        acts = list()
        positions = list()
        for i in range(2):
            poss_act = poss_acts[i]
            if random:
                action = np.random.choice(poss_act, 1)
            else:
                action_logits = model(sts[i][np.newaxis])

                poss_logits = [action_logits[0][i] for i in poss_act]
                a = tf.random.categorical([poss_logits], 1)[0, 0]
                action = poss_act[a]

            acts.append(action)

        next_sts, poss_acts, _, done = env.step(acts)
        for i in range(2):
            pos = env.get_pos(i)
            if i == 1:
                pos = (pos[0], env.size[1] - pos[1] - 1)
            positions.append(pos)
        reward1 = reward_func(sts, acts, next_sts, 1 / 7, g1, h1)
        reward2 = reward_func(sts, acts, next_sts, 1 / 7, g2, h2)
        states.append(sts)
        actions.append(acts)
        next_states.append(next_sts)
        agent_pos.append(positions)

        rewards1.append(reward1)
        rewards2.append(reward2)
        sts = next_sts

        if done:
            break

    return [s[0] for s in states], [s[1] for s in states], [a[0] for a in actions],  [a[1] for a in actions], \
           [ns[0] for ns in next_states], [ns[1] for ns in next_states], [ap[0] for ap in agent_pos], [ap[1] for ap in agent_pos],\
            [r1[0] for r1 in rewards1], [r1[1] for r1 in rewards1], [r2[0] for r2 in rewards2], [r2[1] for r2 in rewards2]


def one_hot_encoding(acts):
    actions = np.array([np.zeros(7) for _ in acts])
    for i in range(len(acts)):
        actions[i][acts[i]] = 1

    return actions


def load_trajs(env, g, h):
    steps = 1000
    states_left, states_right = [], []
    actions_left, actions_right = [], []
    next_states_left, next_states_right = [], []
    positions_left, positions_right = [], []
    rewards_left, rewards_right = [], []
    model = load_agent()
    for i in tqdm(range(20)):
        s1, s2, a1, a2, ns1, ns2, ap1, ap2, r1, r2 = generate_session(model, steps, env, g, h, False)
        states_left.extend(s1)
        states_right.extend(s2)
        actions_left.extend(a1)
        actions_right.extend(a2)
        next_states_left.extend(ns1)
        next_states_right.extend(ns2)
        positions_left.extend(ap1)
        positions_right.extend(ap2)
        rewards_left.extend(r1)
        rewards_right.extend(r2)
        print("end".format(i))

    trajs_left = [states_left, actions_left, next_states_left, positions_left, rewards_left]
    trajs_right = [states_right, actions_right, next_states_right, positions_right, rewards_right]
    return trajs_left, trajs_right


def load_comparison(env, g1, h1, g2, h2, random=False):
    steps = 1000
    states = []
    actions = []
    next_states = []
    positions = []
    rewards1, rewards2 = [], []
    model = load_agent()
    for i in tqdm(range(20)):
        s1, s2, a1, a2, ns1, ns2, ap1, ap2, r1_1, r1_2, r2_1, r2_2 = generate_comparison(model, steps, env, g1, h1, g2, h2, random)
        states.extend(s1)
        states.extend(s2)
        actions.extend(a1)
        actions.extend(a2)
        next_states.extend(ns1)
        next_states.extend(ns2)
        positions.extend(ap1)
        positions.extend(ap2)
        rewards1.extend(r1_1)
        rewards1.extend(r1_2)
        rewards2.extend(r2_1)
        rewards2.extend(r2_2)
        print("end".format(i))

    trajs = [states, actions, next_states, positions, rewards1, rewards2]

    return trajs


# delete double state action pairs
def clean_comparison(trajs):
    states, actions, next_states, positions, rewards1, rewards2 = trajs
    s, a, ns, p, r1, r2 = [], [], [], [], [], []
    sdict = dict()
    adict = dict()
    for i in range(np.shape(trajs)[1]):
        if not any(np.array_equal(states[i], v) for v in list(sdict.values())):
            sdict[i] = states[i]
            adict[i] = [actions[i]]
            s.append(states[i])
            a.append(actions[i])
            ns.append(next_states[i])
            p.append(positions[i])
            r1.append(rewards1[i])
            r2.append(rewards2[i])

        else:
            k = [x for x in sdict if np.array_equal(states[i], sdict[x])][0]
            print(k)
            if actions[i] not in list(adict.get(k)):
                adict[k].append(actions[i])
                s.append(states[i])
                a.append(actions[i])
                ns.append(next_states[i])
                p.append(positions[i])
                r1.append(rewards1[i])
                r2.append(rewards2[i])

    return [s, a, ns, p, r1, r2]


# delete double state action pairs
def clean_trajs(trajs):
    states, actions, next_states, positions, rewards = trajs
    s, a, ns, p, r = [], [], [], [], []
    sdict = dict()
    adict = dict()
    for i in range(np.shape(trajs)[1]):

        if not any(np.array_equal(states[i], v) for v in list(sdict.values())):
            sdict[i] = states[i]
            adict[i] = [actions[i]]
            s.append(states[i])
            a.append(actions[i])
            ns.append(next_states[i])
            p.append(positions[i])
            r.append(rewards[i])
        else:
            k = [x for x in sdict if np.array_equal(states[i], sdict[x])][0]
            print(k)
            if actions[i] not in list(adict.get(k)):
                adict[k].append(actions[i])
                s.append(states[i])
                a.append(actions[i])
                ns.append(next_states[i])
                p.append(positions[i])
                r.append(rewards[i])

    return [s, a, ns, p, r]


# separate the rewards obtained from the two different areas
def two_areas(positions, rewards1, rewards2, env):
    left = list()
    right = list()
    for pos, rew1, rew2 in tqdm(zip(positions, rewards1, rewards2)):
        if pos[1] >= env.size[1] // 2:
            right.append((rew2 - rew1))
        else:
            left.append((rew2 - rew1))
    return left, right


# analyse the retrieved reward function
def analyse_rewards(trajs_left, trajs_right):
    _, _, _, _, rewards_left = trajs_left
    _, _, _, _, rewards_right = trajs_right

    print(np.mean(rewards_left), np.mean(rewards_right))
    plot_distributions(rewards_left, rewards_right)


# compare the retrieved reward function with the one of the external agent
def compare(trajs, env, ep, save=False):
    states, actions, next_states, positions, rewards1, rewards2 = trajs
    if save:
        norm_data = open(r'./norm_data/ep{}.pkl'.format(ep), 'wb')
        data = [states, actions, [np.abs(r1-r2) for r1, r2 in zip(rewards1, rewards2)]]
        pickle.dump(data, norm_data)
        norm_data.close()
    left, right = two_areas(positions, rewards1, rewards2, env)

    print(np.mean(left), np.mean(right))
    plot_distributions(left, right)

    left.extend(right)
    left_file = open(r'./left_data/left.pkl'.format(ep), 'wb')
    pickle.dump(left, left_file)
    left_file.close()
    """
    ax = plt.axes()
    sns.histplot(np.array(left), ax=ax, bins=50, element="poly")
    ax.legend()
    plt.show()
    """


def plot_distributions(left, right):
    print("plot")
    ax = plt.axes()
    print(left, right)
    sns.histplot(np.array(left), ax=ax, bins=50, element="poly", label='left area')
    sns.histplot(np.array(right), ax=ax, bins=50, element="poly", label='right area')

    plt.legend()
    plt.show()


def start_analysis(ep):
    env = environment.Environment(2, "none", True, True)
    g, h = load_reward(ep)
    print("traj")
    trajs_left, trajs_right = load_trajs(env, g, h)
    trajs_right = clean_trajs(trajs_right)
    trajs_left = clean_trajs(trajs_left)
    print("study")
    analyse_rewards(trajs_left, trajs_right)

def start_comparison(ep, random=False):
    env = environment.Environment(2, "not_random", "none", False)
    g1, h1 = load_reward(ep) #expert
    g2, h2 = load_reward(8, False) #external agent
    print("traj")
    trajs = load_comparison(env, g1, h1, g2, h2, random)
    print(np.shape(trajs))
    trajs = clean_comparison(trajs)
    print(np.shape(trajs))
    print("study")
    compare(trajs, env, ep, True)

if __name__ == '__main__':

    for ep in range(5, 6):
        start_analysis(ep)
        # start_comparison(ep, False)

    """
    with open('./left_data/left.pkl', "rb") as f:
        left = pickle.load(f)
        
    ax = plt.axes()
    sns.histplot(np.array(left), ax=ax, bins=50, element="poly")
    ax.legend()
    plt.axvline(x=4.2, color="black", linestyle="--")
    plt.show()
    """