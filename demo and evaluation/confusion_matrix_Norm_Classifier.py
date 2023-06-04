import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
import environment
import matplotlib.pyplot as plt

"""
This file plots confusionmatrix of the Norm Classifier.
The output plot is explained in Fig 4.6 (a) of the dissertation.
"""


def load_agent():
    checkpoint_path = "./output_files/external_agent/training_410/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)

    return model


def generate_episode(model, steps, env, random):
    env.reset()
    sts = env.get_observation()
    poss_acts = env.possible_actions()
    fp, fn, tp, tn = 0, 0, 0, 0
    apple_fp = 0  # apples gathered in the right area classified wrong
    apple_tn = 0  # apples gathered in the right area classified right
    for step in range(steps):
        acts = list()

        for i in range(2):
            if random:
                action = np.random.choice(poss_acts[i], 1)
            else:
                action_logits = model(sts[i][np.newaxis])
                poss_logits = [action_logits[0][i] for i in poss_acts[i]]
                a = tf.random.categorical([poss_logits], 1)[0, 0]
                action = poss_acts[i][a]
            acts.append(action)
        next_sts, poss_acts, rew, done = env.step(acts)

        for i in range(2):
            norm = False
            act = np.zeros(7)
            act[acts[i]] = 1
            norm_value = model_norm([sts[i][np.newaxis], act[np.newaxis]])
            if norm_value > 0.5:
                norm = True
            if not norm and rew[i] != 0 and \
            ((env.get_pos(i)[1] > env.size[0]//2 and i == 0) or (env.get_pos(i)[1] < env.size[0]//2 and i == 1)):
                fn += 1
            elif norm and rew[i] != 0 and ((env.get_pos(i)[1] > env.size[0]//2 and i == 0) or (env.get_pos(i)[1] < env.size[0]//2 and i == 1)):
                tp += 1
            elif norm and ((env.get_pos(i)[1] < env.size[0]//2 and i == 0) or (env.get_pos(i)[1] > env.size[0]//2 and i == 1)):
                fp += 1
                if rew[i] != 0:
                    apple_fp += 1
            elif not norm and ((env.get_pos(i)[1] < env.size[0]//2 and i == 0) or (env.get_pos(i)[1] > env.size[0]//2 and i == 1)):
                tn += 1
                if rew[i] != 0:
                    apple_tn += 1
        sts = next_sts

        if done:
            break

    return tp, tn, fp, fn, apple_fp, apple_tn


def load_values( random=False):
    env = environment.Environment(2, "none", False)
    steps = 1000
    all_tp, all_tn, all_fp, all_fn, all_apples_fp, all_apples_tn = [], [], [], [], [], []
    model = load_agent()
    n = 10
    for i in tqdm(range(n)):
        tp, tn, fp, fn, apple_fp, apple_tn = generate_episode(model, steps, env, random)
        all_tp.append(np.array(tp))
        all_tn.append(np.array(tn))
        all_fp.append(np.array(fp))
        all_fn.append(np.array(fn))
        all_apples_fp.append((np.array(apple_fp)))
        all_apples_tn.append((np.array(apple_tn)))
        print("end".format(i))

    return np.sum(all_tp)//n, np.sum(all_tn)//n, np.sum(all_fp)//n, np.sum(all_fn)//n, np.sum(all_apples_fp)//n, np.sum(all_apples_tn)//n


def load_model(folder):
    checkpoint_path = "./{}/cp.ckpt".format(folder)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model = tf.keras.models.load_model(checkpoint_dir)
    return model


def plot_cm(tp, tn, fp, fn, apple_fp, apple_tn):
    # Create the confusion matrix
    # conf_matrix1 = np.array([[0, fp], [0, tn]])
    conf_matrix2 = np.array([[tp, apple_fp], [fn, apple_tn]])

    # Plot the confusion matrix
    # plt.imshow(conf_matrix1)
    plt.imshow(conf_matrix2, cmap='tab20')

    # Add labels to the plot
    plt.xticks([0, 1], ['Positive', 'Negative'])
    plt.yticks([0, 1], ['Positive', 'Negative'])
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    # Add values to the cells of the matrix

    plt.text(0, 0, '{}\n-'.format(conf_matrix2[0, 0]), ha='center', va='center', color='white')
    plt.text(0, 1, '{}\n{}'.format(conf_matrix2[0, 1], fp), ha='center', va='center', color='white')
    plt.text(1, 0, '{}\n-'.format(conf_matrix2[1, 0]), ha='center', va='center', color='white')
    plt.text(1, 1, '{}\n{}'.format(conf_matrix2[1, 1], tn), ha='center', va='center', color='white')

    # Show the plot
    plt.show()


if __name__ == '__main__':
    folder_norm = "output_files/norm_classifier"
    model_norm = load_model(folder_norm)
    tp, tn, fp, fn, apple_fp, apple_tn = load_values(False)
    plot_cm(tp, tn, fp, fn, apple_fp, apple_tn)