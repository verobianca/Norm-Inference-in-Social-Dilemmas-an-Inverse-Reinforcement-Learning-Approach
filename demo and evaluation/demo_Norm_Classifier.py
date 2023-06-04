import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import environment

"""
Evaluation of the performance of a random agent that moves accordingly to the prediction of the Norm Classifier.
The comparison between its performance and the one of a simple random policy is explained in Fig. 4.6 (b) of the dissertation.
"""


#  generate a full episode. If norm is False it generate a simple random episode
def generate_episode(steps, norm=True):
    env.reset()
    states = env.get_observation()
    poss_actions = env.possible_actions()

    for timestep in range(1, steps):
        if prints == 'map':
            env.render(episode)

        actions = []
        for i in range(number):
            if norm:
                norm_value = dict()
                while True:
                    if not poss_actions[i]:
                        action = min(norm_value, key=norm_value.get)
                        break
                    else:
                        action = np.random.choice(poss_actions[i])
                        act = np.zeros(7)
                        act[action] = 1
                        norm = model_norm([states[i][np.newaxis], act[np.newaxis]])
                        if norm > 0.5:
                            poss_actions[i].remove(action)
                            norm_value[action] = norm
                        else:
                            break
            else:
                action = np.random.choice(poss_actions[i])

            act = tf.convert_to_tensor(action)
            actions.append(act)

        states, poss_actions, rewards, done = env.step(actions)

        if done:
            ep_st.append(timestep)
            if prints == 'map':
                env.close_plot()
            break

        if timestep == (steps - 1):
            ep_st.append(timestep + 1)
            if prints == 'map':
                env.close_plot()

    episode_rewards.append(np.sum(env.rewards))
    episode_errors.append(env.errors)


def load_model(folder):
    checkpoint_path = "./{}/cp.ckpt".format(folder)

    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)
    return model


def plot_results():
    N = 2
    all_rewards = [155, 142]
    all_errors =[155-71, 8]
    print((all_rewards[0] - all_errors[0], all_rewards[1] - all_errors[1]))
    print(all_errors[0], all_errors[1])
    good_apples = (all_rewards[0] - all_errors[0], all_rewards[1] - all_errors[1])
    violations = (all_errors[0], all_errors[1])
    ind = np.arange(N)  # the x locations for the groups
    width = 0.35
    figure, ax = plt.subplots(1, 1)
    ax.bar(ind, good_apples, width, color='r')
    ax.bar(ind, violations, width, bottom=good_apples, color='b')
    ax.set_xticks(ind, ('without norm', 'with norm'))
    ax.set_yticks(np.arange(0, 201, 10))
    ax.legend(labels=['non-violation', 'violation'])
    plt.show()



"""
This is the DEMO of the random agent which moves accordingly to the NormClassifier predictions
To start the DEMO set the following variable:
-folder = the name of the folder were the Norm Classifier is saved
-print = "map" if you want to look at the whole map
         "observations" if you want to look at the observations in input to the model (of the first agent)
         "none" if you want to run the episodes to generate the results' plot
-steps = how many steps for episode should be showed
"""


if __name__ == '__main__':
    folder_norm = "output_files/norm_classifier"  # the Norm Classifier
    prints = "none"
    steps = 1000

    number = 2
    all_rewards = list()
    all_errors = list()
    all_steps = list()

    for episode in range(2):  # one episode is random, the other follows the Norm Classifier predictions
        episode_rewards = list()
        episode_errors = list()
        episode_private_area = list()
        ep_st = list()

        env = environment.Environment(number, prints)
        model_norm = load_model(folder_norm)

        for _ in range(5):
            if episode == 0:
                generate_episode(steps, False)
            else:
                generate_episode(steps)
        all_rewards.append(np.mean(episode_rewards))
        all_steps.append(np.mean(ep_st))

        all_errors.append(np.mean(episode_errors))

    print(all_rewards)
    print(all_steps)
    print(all_errors)

    plot_results()



