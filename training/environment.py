import numpy as np
import matplotlib.pyplot as plt
import imageio
"""
Environment of the system.
"""

class Environment:
    def __init__(self, number=2, print="map", norm=False, eval=False):
        self.eval = eval
        self.number = number
        self.map = map
        self.size = (17, 17)
        self.print = print
        self.norm = norm
        self.size_ob = 5
        self.size_obs = (5, 5)

    def set_map(self):
        # set size and private areas
        self.grid = np.zeros(self.size)
        self.grid[:, 1: (self.size[0]) // 2] = 2
        self.grid[:, self.size[0] // 2:-1] = 3

        # walls
        self.grid[:, self.size[0] // 2] = 0
        self.grid[0, :] = 1
        self.grid[-1, :] = 1
        self.grid[:, 0] = 1
        self.grid[:, -1] = 1

        # apples
        self.apples = {(2, 5): 1, (2, 11): 1, (5, 2): 1, (5, 14): 1, (8, 5): 1, (8, 11): 1,
                       (11, 2): 1, (11, 14): 1, (14, 5): 1, (14, 11): 1}

        keys = list(self.apples.keys())
        for a in keys:
            i, j = a
            self.apples[i + 1, j] = 1
            self.apples[i, j + 1] = 1
            self.apples[i - 1, j] = 1
            self.apples[i, j - 1] = 1

        for i in self.apples:
            self.grid[i] = 4

    def set_agents(self):
        self.agents = list()
        self.agents.append([(self.size[0] // 2, 1), 1, 5])
        self.grid[self.size[0] // 2, 1] = 5

        if self.number > 1:
            self.agents.append([(self.size[0] // 2, self.size[0] - 2), 3, 6])  # pos, orientation, color
            self.grid[self.size[0] // 2, -2] = 6

        self.agents = np.array(self.agents, dtype=object)
        self.rewards = [0 for _ in range(self.number)]
        self.errors = 0

    def add_orientation(self):
        grid_copy = self.grid.copy()

        for i in range(self.number):
            pos = self.agents[i][0]
            orientation = self.agents[i][1]

            if orientation == 0:
                or_pos = (pos[0] - 1, pos[1])
            elif orientation == 1:
                or_pos = (pos[0], pos[1] + 1)
            elif orientation == 2:
                or_pos = (pos[0] + 1, pos[1])
            else:
                or_pos = (pos[0], pos[1] - 1)

            if self.grid[or_pos] in [2, 3]:
                grid_copy[or_pos] = 7

        return grid_copy

    def render(self, ep):
        if self.number == 2:
            if self.print != "none":
                self.ax.title.set_text(
                    'Reward: {}  Step {}  Reward: {}'.format(self.rewards[0], self.steps, self.rewards[1]))
        # grid = self.add_orientation()
        grid = self.grid
        self.myplot.set_data(grid.astype('uint8'))
        plt.savefig('./map.png')
        im = imageio.v2.imread('./map.png')
        self.images.append(im.copy())

        self.fig.canvas.draw_idle()
        plt.pause(0.01)
        self.steps += 1

    def reset(self):
        self.steps = 0
        self.images = []
        self.set_map()
        self.set_agents()
        if self.print == "map":
            self.fig, self.ax = plt.subplots(1, 1)
            # grid = self.add_orientation()
            grid = self.grid
            self.myplot = self.ax.imshow(grid.astype('uint8'))


            #plt.savefig('./map.pdf')
            plt.pause(0.1)

        elif self.print == "observations":
            window = np.zeros(self.size_obs)
            window[0, 0] = 6
            self.fig, self.ax = plt.subplots(1, 1)
            self.myplot = self.ax.imshow(window.astype('uint8'))

        self.total_reward = 0

    def close_plot(self):
        plt.close(self.fig)

    def get_observation(self):
        observations = list()
        for i in range(self.number):
            pos = self.agents[i][0]
            orientation = self.agents[i][1]
            grid_copy = self.grid.copy()

            # if you are running generate_session in evaluate_reward.py you don't want to swap colors
            if i == 1 and not self.eval:
                other_pos = self.agents[0][0]
                grid_copy[other_pos] = 6
                grid_copy[pos] = 5
                grid_copy[self.grid == 2] = 3
                grid_copy[self.grid == 3] = 2
            pad = 5
            grid_copy = np.pad(grid_copy, pad)

            pos_0 = pos[0] + pad
            pos_1 = pos[1] + pad

            if orientation == 0:  # up
                window = grid_copy[pos_0 - 4: pos_0 + 1, pos_1 - 2: pos_1 + 2 + 1].copy()
            elif orientation == 1:  # right
                window = grid_copy[pos_0 - 2: pos_0 + 2 + 1, pos_1: pos_1 + 5].copy()
                window = np.rot90(window)
            elif orientation == 2:  # down
                window = grid_copy[pos_0: pos_0 + 5, pos_1 - 2: pos_1 + 2 + 1].copy()
                window = np.rot90(window, 2)
            else:  # left
                window = grid_copy[pos_0 - 2: pos_0 + 2 + 1, pos_1 - 4: pos_1 + 1].copy()
                window = np.rot90(window, 3)

            if self.print == "observations" and i == 0:
                self.print_observation(window)

            window = np.around(window / 6, 2)
            observations.append(window)

        return observations

    def print_observation(self, window):
        self.myplot.set_data(window.astype('uint8'))
        self.fig.canvas.draw_idle()
        plt.pause(0.5)

    # check which are the legal action in a given state
    def possible_actions(self):
        # 0 step forward
        # 1 step right
        # 2 step backward
        # 3 step left
        # 4 rotate right
        # 5 rotate left
        # 6 stay still

        poss_actions = list()
        for i in range(self.number):
            col = self.agents[i][2]
            poss_action = [4, 5, 6]
            pos = self.agents[i][0]
            orientation = self.agents[i][1]
            # up
            if self.grid[pos[0] - 1, pos[1]] != 1:
                if orientation == 0:
                    poss_action.append(0)
                elif orientation == 1:
                    poss_action.append(3)
                elif orientation == 2:
                    poss_action.append(2)
                elif orientation == 3:
                    poss_action.append(1)
            # right
            if self.grid[pos[0], pos[1] + 1] != 1:
                if self.norm and pos[1] + 1 >= self.size[0] // 2 and col == 5:
                    pass
                else:
                    if orientation == 0:
                        poss_action.append(1)
                    elif orientation == 1:
                        poss_action.append(0)
                    elif orientation == 2:
                        poss_action.append(3)
                    elif orientation == 3:
                        poss_action.append(2)
            # down
            if self.grid[pos[0] + 1, pos[1]] != 1:
                if orientation == 0:
                    poss_action.append(2)
                elif orientation == 1:
                    poss_action.append(1)
                elif orientation == 2:
                    poss_action.append(0)
                elif orientation == 3:
                    poss_action.append(3)
            # left
            if self.grid[pos[0], pos[1] - 1] != 1:
                if self.norm and pos[1] - 1 <= self.size[0] // 2 and col == 6:
                    pass
                else:
                    if orientation == 0:
                        poss_action.append(3)
                    elif orientation == 1:
                        poss_action.append(2)
                    elif orientation == 2:
                        poss_action.append(1)
                    elif orientation == 3:
                        poss_action.append(0)
            poss_actions.append(poss_action)

        return poss_actions

    # do a full step for each agent, get the rewards, change the environment accordingly and get the new states
    def step(self, actions):
        done = False
        rewards = list()
        for i in range(self.number):
            col = self.agents[i][2]
            reward = 0
            pos = self.agents[i][0]
            orientation = self.agents[i][1]
            action = actions[i]
            new_pos = pos
            if action in [0, 1, 2, 3]:
                if action == 0:
                    if orientation == 0:
                        new_pos = (pos[0] - 1, pos[1])
                    elif orientation == 1:
                        new_pos = (pos[0], pos[1] + 1)
                    elif orientation == 2:
                        new_pos = (pos[0] + 1, pos[1])
                    else:
                        new_pos = (pos[0], pos[1] - 1)

                elif action == 1:
                    if orientation == 0:
                        new_pos = (pos[0], pos[1] + 1)
                    elif orientation == 1:
                        new_pos = (pos[0] + 1, pos[1])
                    elif orientation == 2:
                        new_pos = (pos[0], pos[1] - 1)
                    else:
                        new_pos = (pos[0] - 1, pos[1])

                elif action == 2:
                    if orientation == 0:
                        new_pos = (pos[0] + 1, pos[1])
                    elif orientation == 1:
                        new_pos = (pos[0], pos[1] - 1)
                    elif orientation == 2:
                        new_pos = (pos[0] - 1, pos[1])
                    else:
                        new_pos = (pos[0], pos[1] + 1)

                elif action == 3:
                    if orientation == 0:
                        new_pos = (pos[0], pos[1] - 1)
                    elif orientation == 1:
                        new_pos = (pos[0] - 1, pos[1])
                    elif orientation == 2:
                        new_pos = (pos[0], pos[1] + 1)
                    else:
                        new_pos = (pos[0] + 1, pos[1])

                self.agents[i][0] = new_pos
                if pos[1] < self.size[0] // 2:
                    self.grid[pos] = 2
                elif pos[1] > self.size[0] // 2:
                    self.grid[pos] = 3
                else:
                    self.grid[pos] = 0

                if self.grid[new_pos] == 4:
                    self.rewards[i] += 1
                    reward = 1

                    self.apples[new_pos] = 0
                    if (new_pos[1] > self.size[0]//2 and col == 5) or (new_pos[1] < self.size[0]//2 and col == 6):
                        self.errors += 1

                self.grid[new_pos] = col

            elif action == 4:
                self.agents[i][1] = (self.agents[i][1] + 1) % 4

            elif action == 5:
                self.agents[i][1] = (self.agents[i][1] + 3) % 4

            rewards.append(reward)

        self.total_reward += np.sum(rewards)

        # if there aren't any apples left stop the episode
        if 1 not in list(self.apples.values()):
            done = True

        next_states = self.get_observation()
        poss_next_actions = self.possible_actions()
        self.check_apples()

        return next_states, poss_next_actions, rewards, done

    # check if apple can respawn
    def check_apples(self):
        for key in [k for k, v in self.apples.items() if
                    v == 0 and k != self.agents[0][0] and (self.number == 2 and k != self.agents[1][0])]:
            n = 0
            for j in range(key[0] - 2, key[0] + 3):
                for i in range(key[1] - 2, key[1] + 3):
                    try:
                        if self.grid[j, i] == 4:
                            n += 1
                    except:
                        pass
            if n == 0:
                p = 0
            elif n in [1, 2]:
                p = 0.035
                # p = 0.03
            elif n in [3]:
                p = 0.065
                # p = 0.06
            else:
                p = 0.1
            if np.random.rand() < p:
                self.apples[key] = 1
                self.grid[key] = 4

    def get_pos(self, num):
        return self.agents[num][0]
