import numpy as np
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import environment
import imageio
"""
Demo of the trained policies.
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

            action_logits = model(states[i][np.newaxis])
            poss_logits = [action_logits[0][a] for a in poss_actions[i]]
            a = tf.random.categorical([poss_logits], 1)[0, 0]
            # a = np.argmax(poss_logits) # if you want to use a greedy policy
            action = poss_actions[i][a]
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


def load_model(episode):
    checkpoint_path = "./{}/training_{}/cp.ckpt".format(folder, episode)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)

    return model


def plot_results(last_train):
    plt.plot([i for i in range(0, last_train, 2)], all_rewards, label="rewards")
    plt.plot([i for i in range(0, last_train, 2)], all_steps, label="steps")
    plt.plot([i for i in range(0, last_train, 2)], all_errors, label="errors")
    plt.legend(loc='best')
    plt.ylim(bottom=0, top=1100)
    plt.xlabel("episodes")
    plt.show()


"""
To start the DEMO set the following variable:
-folder = the name of the folder were the models you want to load are saved
-print = "map" if you want to look at the whole map
         "observations" if you want to look at the observations in input to the model (of the first agent)
         "none" if you want to run the episodes to generate the results' plot
-steps = how many steps for episode should be showed
-first_ep and last_ep = from which to which saved model you want to be showed 
-norm = True if you want the agent to move only in their own areas, False otherwise
"""


if __name__ == '__main__':
    folder = 'output_files/external_agent'
    prints = "map"
    steps = 1000
    first_ep = 410
    last_ep = 420
    norm = False

    number = 2
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
            generate_episode(episode, steps)
            imageio.mimsave('./external_agent.gif', env.images)

        all_rewards.append(np.mean(episode_rewards))
        all_steps.append(np.mean(ep_st))

        all_errors.append(np.mean(episode_errors))

    print(all_rewards)
    print(all_steps)
    print(all_errors)

    plot_results(last_ep-first_ep+1)



