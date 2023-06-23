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
    Class CliffWalking.
    Have all methods needed to implement the Cliff Walking
    with Q-Learning.
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
        """
        Initialize the reward matrix, Q value and set the terminal states.
        """
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

        # if it is on cliff than oes back to the start position
        if [new_row, new_columns] in self.cliff_position:
            new_row, new_columns = self.start_position
        return [new_row, new_columns], action

    def q_learning(self):
        """
        Compute Q Learning
        """
        for _ in range(self.episodes):
            reward_sum = 0

            # The first state is on the start position
            state = self.start_position

            # Stay in loop until get to the end position
            while state != self.end_position:

                # Compute the next state and action using epsilon greedy method
                next_state, action = self.compute_next_state(state)

                # Compute Q(s,a)
                self.Q[state[0], state[1], action] += self.alpha * (self.reward[next_state[0], next_state[1]] 
                                                    + self.discount * np.max(self.Q[next_state[0], next_state[1], action]) 
                                                    -self.Q[state[0], state[1], action])
                state = next_state
                reward_sum += self.reward[next_state[0], next_state[1]]

            # Store the total reward earned for each episode    
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
        Plot on a graph the value matrix with the optimal trajectory
        Plot the graph Episode x Reward
        """
        plt.subplot(2, 1, 1)
        plt.plot(self.reward_recorded)
        plt.title("Q-Learning Performance")
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')

        arrows = ['↑', '↓', '←', '→']
        ax = plt.subplot(2, 1, 2)
        ax.axis('off')

        value = np.zeros((self.rows, self.columns))
        for i in range(self.rows):
            for j in range(self.columns):
                value[i,j] = np.max(self.Q[i, j, :])

        value_grid = np.around(value, decimals=2)
        value_table = ax.table(cellText=value_grid, cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        value_table.auto_set_font_size(False)
        value_table.set_fontsize(12)

        for i in range(self.rows):
            for j in range(self.columns):
                action = np.argmax(self.Q[i, j, :])
                arrow_symbol = arrows[action]
                value_table[i, j].get_text().set_text(f'{action}{arrow_symbol}')
                value_table[i, j].get_text().set_fontsize(14)

        for i in range(1, self.columns-1):
            value_table[self.rows-1, i].get_text().set_text('C')   
            value_table[self.rows-1, i].set_facecolor('red')

        value_table[self.rows-1, 0].set_facecolor('grey')
        value_table[self.rows-1, 0].get_text().set_text('S')
        value_table[self.rows-1, self.columns-1].set_facecolor('lightgreen')
        value_table[self.rows-1, self.columns-1].get_text().set_text('E')        
        value_table[self.rows-1, 0].set_facecolor('grey')
        value_table[self.rows-1, 0].get_text().set_text('S')
        value_table[self.rows-1, self.columns-1].set_facecolor('lightgreen')
        value_table[self.rows-1, self.columns-1].get_text().set_text('E')

        plt.show()

if __name__ == "__main__":
    cw = CliffWalking(4, 12, 500, 0.4, 1, 0.6) 
    cw.compute()