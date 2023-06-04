import numpy as np
import tensorflow as tf
import environment
import os
import matplotlib.pyplot as plt

"""
Evaluation of the retrieved policies.
The output plots are explained in Fig. 4.2 of the dissertation.
"""


# generate a full episode
def generate_episode(episode, steps):
    env.reset()
    states = env.get_observation()
    poss_actions = env.possible_actions()

    for timestep in range(1, steps):
        if prints == 'map':
            env.render(episode)

        actions = []
        for i in range(number):
            norm_action = dict()
            flag = True
            while flag:
                if not poss_actions[i]:
                    val = np.min(list(norm_action.values()))
                    print(val)
                    print(norm_action)
                    action = [x for x in norm_action if np.array_equal(val, norm_action[x])][0]
                    print(action)
                else:
                    action_logits = model(states[i][np.newaxis])
                    poss_logits = [action_logits[0][a] for a in poss_actions[i]]
                    a = tf.random.categorical([poss_logits], 1)[0, 0]
                    action = poss_actions[i][a]
                    ac = np.zeros(7)
                    ac[action] = 1
                    ac = tf.expand_dims(ac, 0)
                    norm_value = norm_model([states[i][np.newaxis], ac])
                    if not (norm_value > 0.7):
                        flag = False
                    else:
                        poss_actions[i].remove(action)
                        norm_action[action] = norm[0][0]
                        print('true')

            act = tf.convert_to_tensor(action)
            actions.append(act)

        states, poss_actions, rewards, done = env.step(actions)

        if done:
            ep_st.append(timestep)
            env.close_plot()
            break
        if timestep == (steps - 1):
            ep_st.append(timestep)
            env.close_plot()
    episode_rewards.append(np.sum(env.rewards))
    episode_errors.append(env.errors)



def load_norm_model():
    checkpoint_path = './norm_model/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)

    return model


def load_model(episode):
    checkpoint_path = "./{}/agent_2/training_{}/cp.ckpt".format(folder, episode)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)

    return model


def plot_results(last_train):
    figure, axis = plt.subplots(1, 1)
    print(all_rewards)
    axis.plot([i for i in range(0, last_train, 10)], all_rewards)
    axis.plot([i for i in range(0, last_train, 10)], all_errors)
    axis.plot([i for i in range(0, last_train, 10)], all_steps)
    axis.set_ylim(bottom=0, top=150)
    axis.set_xlabel("Episodes")
    plt.show()


if __name__ == '__main__':
    #folder = 'AIRL_policy/norm8'
    folder = "new/not_random_no_norm"
    number = 2
    prints = "map"
    steps = 1000
    first_ep = 210
    last_ep = 210

    norm = False
    all_rewards = list()
    all_errors = list()
    all_steps = list()

    for episode in range(first_ep, last_ep+1, 10):
        episode_rewards = list()
        episode_errors = list()
        episode_private_area = list()
        ep_st = list()

        env = environment.Environment(number, prints, norm)
        for _ in range(1):
            model = load_model(episode)
            norm_model = load_norm_model()
            generate_episode(episode, steps)

        all_rewards.append(np.mean(episode_rewards))
        all_steps.append(np.mean(ep_st))
        all_errors.append(np.mean(episode_errors))

    plot_results(last_ep-first_ep+1)


