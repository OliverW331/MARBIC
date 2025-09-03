"""
Visualization script for CorporateBiodiversityEnv

This script demonstrates running the custom Gym environment
and plotting the initial and final state of the grid variables (H, D, T).

Usage:
    python visualize_env.py
"""

import matplotlib.pyplot as plt
import numpy as np
from gym_corp_biodiv_env import CorporateBiodiversityEnv


def plot_grids(H, D, T, grid_size, title_prefix=""):
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    im0 = axs[0].imshow(H.reshape((grid_size, grid_size)), cmap="Greens")
    axs[0].set_title(f"{title_prefix} Species Abundance H")
    plt.colorbar(im0, ax=axs[0])

    im1 = axs[1].imshow(D.reshape((grid_size, grid_size)), cmap="Reds", vmin=0, vmax=1)
    axs[1].set_title(f"{title_prefix} Disturbance D")
    plt.colorbar(im1, ax=axs[1])

    im2 = axs[2].imshow(T.reshape((grid_size, grid_size)), cmap="Blues", vmin=0, vmax=10)
    axs[2].set_title(f"{title_prefix} Climate Stress T")
    plt.colorbar(im2, ax=axs[2])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env = CorporateBiodiversityEnv(grid_size=6, Nc=3, Ni=2, seed=123, max_steps=50)
    obs = env.reset()

    # plot initial state
    plot_grids(obs['H'], obs['D'], obs['T'], env.grid_size, title_prefix="Initial")

    # run environment for full episode with random actions
    done = False
    while not done:
        action = {
            'corp_actions': env.action_space.spaces['corp_actions'].sample(),
            'corp_targets': env.action_space.spaces['corp_targets'].sample(),
            'investor_actions': env.action_space.spaces['investor_actions'].sample(),
        }
        obs, reward, done, info = env.step(action)

    # plot final state
    plot_grids(obs['H'], obs['D'], obs['T'], env.grid_size, title_prefix="Final")
