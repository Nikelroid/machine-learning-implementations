from __future__ import print_function
import json
import numpy as np

class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        alpha = np.zeros((S, L))
        for i in range(S):
            alpha[i, 0] = self.pi[i] * self.B[i, O[0]]
        for t in range(1, L):
            for j in range(S):
                current_sum = 0.0
                for i in range(S):
                    current_sum += alpha[i, t-1] * self.A[i, j]
                alpha[j, t] = current_sum * self.B[j, O[t]]
        return alpha

    def backward(self, Osequence):
        S = len(self.pi)
        L = len(Osequence)
        O = self.find_item(Osequence)
        beta = np.zeros((S, L))
        for i in range(S):
            beta[i, L-1] = 1.0
        for t in range(L-2, -1, -1):
            for i in range(S):
                current_sum = 0.0
                for j in range(S):
                    current_sum += self.A[i, j] * self.B[j, O[t+1]] * beta[j, t+1]
                beta[i, t] = current_sum
        return beta

    def sequence_prob(self, Osequence):
        alpha_matrix = self.forward(Osequence)
        num_states = alpha_matrix.shape[0]
        num_observations = alpha_matrix.shape[1]
        prob_sum = 0.0
        if num_observations > 0:
            for i in range(num_states):
                prob_sum += alpha_matrix[i, num_observations-1]
        return prob_sum

    def posterior_prob(self, Osequence):
        S = len(self.pi)
        L = len(Osequence)

        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        gamma = np.zeros((S, L))
        
        for t in range(L):
            current_col_sum = 0.0
            for i in range(S):
                gamma[i, t] = alpha[i, t] * beta[i, t]
                current_col_sum += gamma[i,t]
            
            if current_col_sum != 0:
                for i in range(S):
                    gamma[i, t] = gamma[i, t] / current_col_sum
        return gamma

    def likelihood_prob(self, Osequence):
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        O = self.find_item(Osequence)
        P_X = np.sum(alpha[:, L-1])
        if P_X == 0:return prob
        for t in range(L - 1):
            for i in range(S):
                for j in range(S):prob[i, j, t] = (alpha[i, t] * self.A[i, j] * self.B[j, O[t + 1]] * beta[j, t + 1]) / P_X
        return prob

    def viterbi(self, Osequence):
        path = []
        obs_indices = self.find_item(Osequence)
        T = len(obs_indices)
        S = len(self.pi)

        delta = np.zeros((T, S))
        psi = np.zeros((T, S), dtype=int)

        for i in range(S):
            delta[0, i] = self.pi[i] * self.B[i, obs_indices[0]]
            psi[0, i] = 0
        
        for t in range(1, T):
            for j in range(S):
                max_prob = 0.0
                max_index = 0
                for i in range(S):
                    current_prob = delta[t-1, i] * self.A[i, j]
                    if current_prob > max_prob:
                        max_prob = current_prob
                        max_index = i
                delta[t, j] = max_prob * self.B[j, obs_indices[t]]
                psi[t, j] = max_index
        
        best_last_state = 0
        max_val = 0.0
        if T > 0 :
             for i in range(S):
                if delta[T-1, i] > max_val:
                    max_val = delta[T-1, i]
                    best_last_state = i

        best_path_indices = np.zeros(T, dtype=int)
        if T > 0:
            best_path_indices[T-1] = best_last_state
            for t in range(T-2, -1, -1):
                best_path_indices[t] = psi[t+1, best_path_indices[t+1]]
        
        for state_idx in best_path_indices:
            path.append(self.find_key(self.state_dict, state_idx))
        return path
    
    #DO NOT MODIFY CODE BELOW
    def find_key(self, obs_dict, idx):
        for item in obs_dict:
            if obs_dict[item] == idx:
                return item

    def find_item(self, Osequence):
        O = []
        for item in Osequence:
            O.append(self.obs_dict[item])
        return O
