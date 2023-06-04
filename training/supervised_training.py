import tensorflow as tf
import os
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

"""
Training of the Norm Classifier.
"""


def one_hot_encoding(acts):
    actions = np.array([np.zeros(7) for _ in acts])
    for i in range(len(acts)):
        actions[i][acts[i]] = 1

    return actions


def load_data(ep):
    with open('./norm_data/ep{}.pkl'.format(ep), "rb") as f:
        a = pickle.load(f)
    st, act, diff = a[0], a[1], a[2]
    return [[s, a] for s, a in zip(st, one_hot_encoding(act))], [0 if d < 5.25 else 1 for d in diff]


def model():
    state1 = tf.keras.layers.Input(shape=(5, 5, 1))
    ohe_prev_act1 = tf.keras.layers.Input(shape=(7,))
    x1 = tf.keras.layers.Conv2D(filters=3, kernel_size=3, padding='same', activation="relu")(state1)
    x1 = tf.keras.layers.Conv2D(filters=6, kernel_size=3, padding='same', activation="relu")(x1)
    x1 = tf.keras.layers.Flatten()(x1)
    x1 = tf.keras.layers.Concatenate(axis=1)([x1, ohe_prev_act1])

    x1 = tf.keras.layers.Dense(units=64, input_dim=2, activation="relu")(x1)
    x1 = tf.keras.layers.Dense(units=32, activation="relu")(x1)
    out1 = tf.keras.layers.Dense(1, activation="sigmoid")(x1)
    model = tf.keras.Model(inputs=[state1, ohe_prev_act1], outputs=out1, name='forward')

    return model


def process_data(xx, yy):
    state = [x[0] for x in xx]
    action = [x[1] for x in xx]
    x = [np.array(state), np.array(action)]
    y = np.array(yy)

    return x, y


def save_model(model):
    checkpoint_path = './norm_classifier/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)
    model.save(checkpoint_dir)


if __name__ == '__main__':
    ep = 5
    x_data, y_data = load_data(ep)
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.33)

    x, y = process_data(x_train, y_train)

    model = model()
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    model.fit(x, y, batch_size=32, epochs=100)

    x, y = process_data(x_test, y_test)
    model.evaluate(x, y)

    save_model(model)

