'''
This script defines how the frequency of exploitation is related to mortality rate, birth rate and carrying capacity on a grid level
- D disturbance: higher exploitation frequency leads to increased mortality rate
- B disturbance: higher exploitation frequency leads to decreased birth rate
- K disturbance: higher exploitation frequency leads to decreased carrying capacity
- Disturbance has the range of [0,1]
Assume each species has different sensitivity to disturbance
- Sensitivity is a float number >= 0, larger means more sensitive
'''

import numpy as np
import matplotlib.pyplot as plt

class Disturbance:
    def __init__(self, effect_of_exploitation_mat, 
                 alpha=0.1, beta=0.05, gamma=0.02,
                 k=1.0, f0=10.0, D_mode="exponential", B_mode="exponential", K_mode="exponential"):
        """
        effect_of_exploitation_mat : 3D numpy array, effect of exploitation of each species (0 to ∞)
        alpha, beta, gamma : sensitivity parameters for exponential mode
        k, f0 : steepness and threshold parameters for sigmoid mode
        mode : "exponential" or "sigmoid"
        """
        self.f = effect_of_exploitation_mat
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.k = k
        self.f0 = f0
        self.D_mode = D_mode
        self.B_mode = B_mode
        self.K_mode = K_mode

    def _sigmoid(self, f):
        return 1 / (1 + np.exp(-self.k * (f - self.f0)))

    def transform_to_D_disturbance(self):
        """
        D-effect: mortality disturbance
        """
        if self.D_mode == "exponential":
            return 1 - np.exp(-self.alpha * self.f)
        elif self.D_mode == "sigmoid":
            return self._sigmoid(self.f)
        else:
            raise ValueError("D_mode must be 'exponential' or 'sigmoid'")

    def transform_to_B_disturbance(self):
        """
        B-effect: reduction in birth/growth rate
        """
        if self.B_mode == "exponential":
            return 1 / (1 + self.beta * self.f)
        elif self.B_mode == "sigmoid":
            return 1 - self._sigmoid(self.f)  # high f -> lower birth rate
        else:
            raise ValueError("B_mode must be 'exponential' or 'sigmoid'")

    def transform_to_K_disturbance(self):
        """
        K-effect: reduction in carrying capacity
        """
        if self.K_mode == "exponential":
            return np.exp(-self.gamma * self.f)
        elif self.K_mode == "sigmoid":
            return 1 - self._sigmoid(self.f)  # high f -> lower carrying capacity
        else:
            raise ValueError("K_mode must be 'exponential' or 'sigmoid'")

    def plot_disturbance_effects(self, max_f=5):
        """
        Visually plot self.f vs D, B, K disturbance effects.
        """
        f_range = np.linspace(0, max_f, 200)
        # Generate virtualfmatrix for plotting
        D = (1 - np.exp(-self.alpha * f_range)) if self.D_mode == "exponential" else 1 / (1 + np.exp(-self.k * (f_range - self.f0)))
        B = (1 / (1 + self.beta * f_range)) if self.B_mode == "exponential" else 1 - (1 / (1 + np.exp(-self.k * (f_range - self.f0))))
        K = (np.exp(-self.gamma * f_range)) if self.K_mode == "exponential" else 1 - (1 / (1 + np.exp(-self.k * (f_range - self.f0))))

        plt.figure(figsize=(8,5))
        plt.plot(f_range, D, label="D disturbance (mortality)", color='red')
        plt.plot(f_range, B, label="B disturbance (birth rate)", color='blue')
        plt.plot(f_range, K, label="K disturbance (carrying capacity)", color='green')
        plt.xlabel("Effect of Exploitation (f)")
        plt.ylabel("Disturbance Effect")
        plt.title("Disturbance Effects vs Exploitation")
        plt.legend()
        plt.grid(True)
        plt.show()