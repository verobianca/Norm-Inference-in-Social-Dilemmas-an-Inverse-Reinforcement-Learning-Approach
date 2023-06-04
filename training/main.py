import tensorflow as tf
import os
import sac

"""
Main to train a policy on a given reward function with SAC
"""


# load the reward function
def load_reward(ep):
    checkpoint_path = 'output_files/AIRL_reward_not_sustainable/training_{}/g/cp.ckpt'.format(ep)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cost = tf.keras.models.load_model(checkpoint_dir)
    checkpoint_path = 'output_files/AIRL_reward_not_sustainable/training_{}/h/cp.ckpt'.format(ep)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    h = tf.keras.models.load_model(checkpoint_dir)

    return cost, h


if __name__ == '__main__':
    """
    for i in range(0, 15):
        sac = sac.Soft_AC('folder'.format(i), 'map', False, False)
        g, h = load_reward(i)
        sac.start(150, 1000, 50, 2, True, g, h)

    """
    # train on the reward function of the environment (the one of the external agent)
    sac = sac.Soft_AC('expert', 'map', False)
    sac.start(3000, 300, 10)
