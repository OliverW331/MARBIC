#!/usr/bin/env python3
"""
Species Visualization Demo Script

Demonstrate how to usespecies_visualization.py functions to visualize species characteristics
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add project path
sys.path.append('/Users/muchenwang/Documents/GitHub/MARBIC')

from gym_marbic import CorporateBiodiversityEnv
from species_visualization import *

def demo_species_visualization():
    """Demonstrate species visualization functionality"""
    print("=" * 60)
    print("Species Characteristics Visualization Demo")
    print("=" * 60)
    
    # Create environment
    print("Creating environment...")
    env = CorporateBiodiversityEnv(
        grid_size=10,  # Use smaller grid for demonstration
        n_species=15,
        carrying_capacity=25,
        n_corporations=3,
        n_investors=2,
        seed=42
    )
    
    # Reset environment to initialize species
    print("Initializing species...")
    obs = env.reset()
    
    print(f"Environment created with {env.n_species} species in {env.grid_size}x{env.grid_size} grid")
    print(f"Total cells: {len(env.list_cells)}")
    
    # Check data
    species_dynamic = env.sdyn
    list_cells = env.list_cells
    
    print("\n" + "=" * 60)
    print("1. Species Disturbance Sensitivity Analysis")
    print("=" * 60)
    plot_species_disturbance_sensitivity(species_dynamic)
    
    print("\n" + "=" * 60)
    print("2. Species Maximum Age Distribution")
    print("=" * 60)
    plot_species_max_age_distribution(list_cells)
    
    print("\n" + "=" * 60)
    print("3. Species Age Structure (First 9 Species)")
    print("=" * 60)
    plot_species_age_structure(list_cells, species_ids=list(range(min(9, env.n_species))))
    
    print("\n" + "=" * 60)
    print("4. Species Abundance Patterns")
    print("=" * 60)
    plot_species_abundance_patterns(list_cells)
    
    print("\n" + "=" * 60)
    print("5. Species Sensitivity Correlation Analysis")
    print("=" * 60)
    plot_species_sensitivity_correlation(species_dynamic)
    
    print("\n" + "=" * 60)
    print("6. Comprehensive Species Report")
    print("=" * 60)
    df_report = create_species_summary_report(species_dynamic, list_cells, 
                                            save_path='species_characteristics_report.csv')
    
    return env, df_report

def demo_dynamic_changes():
    """demonstration在environmentchange后species特征如何can视化"""
    print("\n" + "=" * 60)
    print("Dynamic Changes Visualization Demo")
    print("=" * 60)
    
    # Create environment
    env = CorporateBiodiversityEnv(
        grid_size=10,
        n_species=12,
        carrying_capacity=30,
        seed=42
    )
    obs = env.reset()
    
    print("Simulating some environmental changes...")
    
    # Simulate some corporate actions
    for step in range(5):
        # randomselectaction
        actions = {}
        for corp_id in env.corp_ids:
            if np.random.random() < 0.7:  # 70%概率采取action
                action_type = np.random.choice([1, 2])  # exploit or restore
                cell_idx = np.random.randint(0, len(env.list_cells))
                actions[corp_id] = {"action_type": action_type, "cell": cell_idx}
        
        obs, rewards, dones, infos = env.step(actions)
        print(f"Step {step + 1}: {len(actions)} corporations took actions")
    
    print("\nVisualizing species characteristics after environmental changes...")
    
    # 再次can视化以查看change
    plot_species_abundance_patterns(env.list_cells)
    
    # display一些changestatisticsinformation
    total_disturbance = sum(cell.disturbance for cell in env.list_cells)
    mean_disturbance = total_disturbance / len(env.list_cells)
    
    print(f"\nEnvironmental Impact Summary:")
    print(f"Total disturbance: {total_disturbance:.2f}")
    print(f"Mean disturbance per cell: {mean_disturbance:.3f}")
    
    return env

def compare_species_characteristics():
    """compare不同environmentparameter下species特征"""
    print("\n" + "=" * 60)
    print("Species Characteristics Comparison")
    print("=" * 60)
    
    # create两个不同parameterenvironment
    env1 = CorporateBiodiversityEnv(
        grid_size=8, n_species=10, carrying_capacity=20,
        min_age=3, max_age=8, seed=42
    )
    
    env2 = CorporateBiodiversityEnv(
        grid_size=8, n_species=10, carrying_capacity=20,
        min_age=5, max_age=12, seed=42
    )
    
    obs1 = env1.reset()
    obs2 = env2.reset()
    
    print("Comparing maximum age distributions between two environments...")
    
    # 提取最大agedata
    def get_max_ages(list_cells):
        return [len(list_cells[0].sp_age_dict[i]) for i in range(len(list_cells[0].sp_age_dict))]
    
    max_ages1 = get_max_ages(env1.list_cells)
    max_ages2 = get_max_ages(env2.list_cells)
    
    # comparecan视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].bar(range(len(max_ages1)), max_ages1, color='lightblue', alpha=0.7, 
                edgecolor='blue', label='Env 1 (age 3-8)')
    axes[0].set_title('Environment 1: Max Age Distribution')
    axes[0].set_xlabel('Species ID')
    axes[0].set_ylabel('Maximum Age')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].bar(range(len(max_ages2)), max_ages2, color='lightcoral', alpha=0.7, 
                edgecolor='red', label='Env 2 (age 5-12)')
    axes[1].set_title('Environment 2: Max Age Distribution')
    axes[1].set_xlabel('Species ID')
    axes[1].set_ylabel('Maximum Age')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"Environment 1 - Mean max age: {np.mean(max_ages1):.2f}")
    print(f"Environment 2 - Mean max age: {np.mean(max_ages2):.2f}")

def main():
    """主function，runalldemonstration"""
    try:
        # 主要demonstration
        env, report = demo_species_visualization()
        
        # 动态changedemonstration
        demo_dynamic_changes()
        
        # comparedemonstration
        compare_species_characteristics()
        
        print("\n" + "=" * 60)
        print("Demo completed successfully!")
        print("All visualization functions are working properly.")
        print("Check 'species_characteristics_report.csv' for detailed species data.")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()