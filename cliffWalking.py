#!/usr/bin/env python

"""
Trabalho Prático #2 - Q-Learning - Cliff Walking

@author: Leonardo Pezenatto da Silva
@email: leonardo.pezenatto@posgrad.ufsc.br
@date: June 24, 2023
"""

import numpy as np
import matplotlib.pyplot as plt

print(__doc__)

class CliffWalking:
    """
    Class Gambler.
    Have all methods needed to implement the Gambler's problem
    with value iteration.
    """
    
    def __init__(self, rows, columns, episodes, alpha=0.5, discount=1, epsilon=0.1):
        self.reward_recorded = []
        self.policy_recorded = []
        self.rows = rows
        self.columns = columns
        self.episodes = episodes
        self.alpha = alpha
        self.discount = discount
        self.epsilon = epsilon
        
    def setup(self):
    
        self.Q = np.zeros((self.rows, self.columns, 4))
                
        self.reward = np.full((self.rows, self.columns),-1) 
        self.reward[self.rows-1][self.columns-1] = -1
        
        for column in range(1,self.columns-1):
            self.reward[self.rows-1][column] = -100 
        
        self.start_position = [self.rows-1, 0]
        self.end_position = [self.rows-1, self.columns-1]
        self.cliff_position = [[self.rows-1, i] for i in range(1, self.columns-1)]


    def compute_next_state(self, state):
        """
        Use epsilon-greedy method to choose the action.
        Update the next state.
        """
        if np.random.uniform(0, 1) > self.epsilon:
            # Exploitation
            action = np.argmax(self.Q[state[0], state[1], :])
        else:
            # Exploration
            action = np.random.randint(4)

        if action == 0: # Up
            new_row = max(0, state[0] - 1)
            new_columns = state[1]
        elif action == 1: # Down
            new_row = min(self.rows - 1, state[0] + 1)
            new_columns = state[1]
        elif action == 2: # Left
            new_row = state[0]
            new_columns = max(0, state[1] - 1)
        elif action == 3: # Right
            new_row = state[0]
            new_columns = min(self.columns - 1, state[1] + 1)
        if [new_row, new_columns] in self.cliff_position:
            new_row, new_columns = self.start_position
        return [new_row, new_columns], action

    def q_learning(self):
        """
        Compute Q Learning
        """
        for _ in range(self.episodes):
            reward_sum = 0
            state = self.start_position
            while state != self.end_position:
                next_state, action = self.compute_next_state(state)
                self.Q[state[0], state[1], action] += self.alpha * (self.reward[next_state[0], next_state[1]] 
                                                    + self.discount * np.max(self.Q[next_state[0], next_state[1], action]) 
                                                    -self.Q[state[0], state[1], action])
                state = next_state
                reward_sum += self.reward[next_state[0], next_state[1]]
            self.reward_recorded.append(reward_sum)
                

    def compute(self):
        """
        Compute Q Learning and plot results
        """
        self.setup()
        self.q_learning()
        self.plot_results()

    def plot_results(self):
        """
        Plot on a graph the value matrix and the policy matrix
        """
        plt.plot(self.reward_recorded)
        plt.title("Q-Learning Performance")
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.show()

        # arrows = ['↑', '↓', '←', '→']
        # fig, (ax1, ax2) = plt.subplots(1, 2)
        # fig.suptitle('Value and Policy k = '+ str(k), fontsize=16)

        # ax1.axis('off')
        # ax2.axis('off')

        # value_grid = np.around(self.value_grid, decimals=2)
        # value_table = ax1.table(cellText=value_grid, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        # value_table.auto_set_font_size(False)
        # value_table.set_fontsize(14)
        # value_table[0, 0].set_facecolor('lightgreen')
        # value_table[self.grid_rows-1, self.grid_columns-1].set_facecolor('lightgreen')

        # policy_table = ax2.table(cellText=self.policy, loc='center', cellLoc='center', bbox=[0, 0, 1, 1])
        # policy_table.auto_set_font_size(False)
        # policy_table.set_fontsize(14)
        # policy_table[0, 0].set_facecolor('lightgreen')
        # policy_table[self.grid_rows-1, self.grid_columns-1].set_facecolor('lightgreen')
        
        # for i in range(self.policy.shape[0]):
        #     for j in range(self.policy.shape[1]):
        #         if [i,j] != [0,0] and [i,j] != [self.grid_rows-1, self.grid_columns-1]:
        #             value = self.policy[i, j]
        #             arrow_symbol = arrows[value]
        #             policy_table[i, j].get_text().set_text(f'{value} {arrow_symbol}')
        #             policy_table[i, j].get_text().set_fontsize(14)
        #         else:
        #             policy_table[i, j].get_text().set_text('')
        # plt.show()


if __name__ == "__main__":
    cw = CliffWalking(4, 12, 500) 
    cw.compute()