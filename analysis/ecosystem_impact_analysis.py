#!/usr/bin/env python3
"""
Corporate Action Impact Analysis

This script analyzes the impact of different corporate behaviors (exploit, restore, greenwash, no action)
on various ecosystem grid map indicators, including:
- Shannon diversity index
- Carrying capacity  
- Number of individuals
- Number of species
- blahblah

And records and visualizes the historical changes of these indicators during simulation.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import sys
import os
from typing import Dict, List, Any, Tuple

# Add project path
sys.path.append('/Users/muchenwang/Documents/GitHub/MARBIC')

from gym_marbic import CorporateBiodiversityEnv
from analysis.species_visualization import *

class EcosystemAnalyzer:
    """Ecosystem analyzer class"""
    
    def __init__(self, **env_params):
        """
        Initialize analyzer
        
        Parameters:
        -----------
        **env_params : dict
            Environment parameters
        """
        self.env_params = env_params
        self.grid_size = env_params.get('grid_size', 50)
        self.results = {}
        
    def create_environment(self):
        """Create new environment instance"""
        return CorporateBiodiversityEnv(**self.env_params)
    
    def extract_grid_metrics(self, list_cells):
        """
        Extract grid metrics from cell list
        
        Parameters:
        -----------
        list_cells : list
            List of Cell objects
            
        Returns:
        --------
        dict : Dictionary containing various indicators
        """
        n_cells = len(list_cells)
        grid_size = int(np.sqrt(n_cells))
        
        # Initialize indicator matrices
        shannon_matrix = np.zeros((grid_size, grid_size))
        carrying_capacity_matrix = np.zeros((grid_size, grid_size))
        individuals_matrix = np.zeros((grid_size, grid_size))
        species_count_matrix = np.zeros((grid_size, grid_size))
        disturbance_matrix = np.zeros((grid_size, grid_size))
        
        # Fill indicator matrices
        for i, cell in enumerate(list_cells):
            row = i // grid_size
            col = i % grid_size
            
            shannon_matrix[row, col] = cell.shannon_div_idx
            carrying_capacity_matrix[row, col] = cell.carrying_capacity
            individuals_matrix[row, col] = cell.n_individuals
            species_count_matrix[row, col] = cell.n_species
            disturbance_matrix[row, col] = cell.disturbance
        
        return {
            'shannon': shannon_matrix,
            'carrying_capacity': carrying_capacity_matrix,
            'individuals': individuals_matrix,
            'species_count': species_count_matrix,
            'disturbance': disturbance_matrix
        }
    
    def simulate_action_scenario(self, action_type: str, steps: int = 100):
        """
        Simulate specific behavior scenario
        
        Parameters:
        -----------
        action_type : str
            Behavior type: 'nothing', 'exploit', 'restore', 'greenwash', 'random'
        steps : int
            Number of simulation steps
            
        Returns:
        --------
        dict : Dictionary containing simulation results
        """
        print(f"\n=== Simulating scenario: {action_type.upper()} ===")
        
        # Create environment
        env = self.create_environment()
        obs = env.reset()
        
        # Record initial state
        initial_metrics = self.extract_grid_metrics(env.list_cells)
        
        # History record
        history = {
            'mean_shannon': [],
            'mean_carrying_capacity': [],
            'mean_individuals': [],
            'mean_species_count': [],
            'mean_disturbance': [],
            'total_individuals': [],
            'corp_capital': [[] for _ in range(env.n_corporations)],
            'corp_biodiv': [[] for _ in range(env.n_corporations)],
            'corp_resilience': [[] for _ in range(env.n_corporations)]
        }
        
        # Simulation loop
        for t in range(steps):
            # Generate actions
            actions = self._generate_actions(env, action_type)
            
            # Execute step
            obs, rewards, dones, infos = env.step(actions)
            
            # Record indicators
            current_metrics = self.extract_grid_metrics(env.list_cells)
            
            # Update history
            history['mean_shannon'].append(np.mean(current_metrics['shannon']))
            history['mean_carrying_capacity'].append(np.mean(current_metrics['carrying_capacity']))
            history['mean_individuals'].append(np.mean(current_metrics['individuals']))
            history['mean_species_count'].append(np.mean(current_metrics['species_count']))
            history['mean_disturbance'].append(np.mean(current_metrics['disturbance']))
            history['total_individuals'].append(np.sum(current_metrics['individuals']))
            
            # Record corporate status
            for i, corp in enumerate(env.corporations):
                history['corp_capital'][i].append(corp.capital)
                history['corp_biodiv'][i].append(corp.biodiversity_score)
                history['corp_resilience'][i].append(corp.resilience)
            
            # Print progress
            if (t + 1) % 50 == 0:
                print(f"  Step {t+1}/{steps} completed")
            
            if all(dones.values()):
                break
        
        # Record final state
        final_metrics = self.extract_grid_metrics(env.list_cells)
        
        return {
            'action_type': action_type,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'history': history,
            'steps_completed': t + 1,
            'env': env  # Save final environment state for analysis
        }
    
    def _generate_actions(self, env, action_type: str):
        """Generate actions of specified type"""
        actions = {}
        
        # Corporate actions
        for aid in env.corp_ids:
            if action_type == 'nothing':
                actions[aid] = {
                    "action_type": 0,  # NO_ACTION
                    "cell": np.random.randint(env.n_cells)
                }
            elif action_type == 'exploit':
                actions[aid] = {
                    "action_type": 1,  # EXPLOIT
                    "cell": np.random.randint(env.n_cells)
                }
            elif action_type == 'restore':
                actions[aid] = {
                    "action_type": 2,  # RESTORE
                    "cell": np.random.randint(env.n_cells)
                }
            elif action_type == 'greenwash':
                actions[aid] = {
                    "action_type": 3,  # GREEN_WASH
                    "cell": np.random.randint(env.n_cells)
                }
            elif action_type == 'random':
                actions[aid] = env.action_spaces[aid].sample()
        
        # Investors take no action
        for aid in env.inv_ids:
            actions[aid] = {
                "action_type": 0, 
                "weights": np.zeros(env.n_corporations)
            }
        
        return actions
    
    def simulate_mixed_strategy(self, strategy_name: str, action_probabilities: Dict[str, float], 
                              steps: int = 100, seed: int = 42):
        """
        Simulate mixed strategy with specified action probabilities
        
        Parameters:
        -----------
        strategy_name : str
            Name of the strategy
        action_probabilities : dict
            Probabilities for each action type (must sum to 1.0)
        steps : int
            Number of simulation steps
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Dictionary containing simulation results
        """
        print(f"\n=== Simulating mixed strategy: {strategy_name} ===")
        
        # Validate probabilities
        if abs(sum(action_probabilities.values()) - 1.0) > 1e-6:
            raise ValueError("Action probabilities must sum to 1.0")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create environment
        env = self.create_environment()
        obs = env.reset()
        
        # Record initial state
        initial_metrics = self.extract_grid_metrics(env.list_cells)
        
        # History record
        history = {
            'mean_shannon': [],
            'mean_carrying_capacity': [],
            'mean_individuals': [],
            'mean_species_count': [],
            'mean_disturbance': [],
            'total_individuals': [],
            'action_counts': {action: 0 for action in action_probabilities.keys()},
            'corp_capital': [[] for _ in range(env.n_corporations)],
            'corp_biodiv': [[] for _ in range(env.n_corporations)],
            'corp_resilience': [[] for _ in range(env.n_corporations)]
        }
        
        # Action mapping
        action_map = {
            'nothing': 0,    # NO_ACTION
            'exploit': 1,    # EXPLOIT
            'restore': 2,    # RESTORE
            'greenwash': 3,  # GREEN_WASH
        }
        
        # Simulation loop
        for t in range(steps):
            # Generate mixed strategy actions
            actions = {}
            
            # Corporate actions based on probabilities
            for aid in env.corp_ids:
                # Choose action based on probabilities
                action_types = list(action_probabilities.keys())
                probabilities = list(action_probabilities.values())
                chosen_action = np.random.choice(action_types, p=probabilities)
                
                actions[aid] = {
                    "action_type": action_map[chosen_action],
                    "cell": np.random.randint(env.n_cells)
                }
                
                # Count actions
                history['action_counts'][chosen_action] += 1
            
            # Investors take no action
            for aid in env.inv_ids:
                actions[aid] = {
                    "action_type": 0, 
                    "weights": np.zeros(env.n_corporations)
                }
            
            # Execute step
            obs, rewards, dones, infos = env.step(actions)
            
            # Record indicators
            current_metrics = self.extract_grid_metrics(env.list_cells)
            
            # Update history
            history['mean_shannon'].append(np.mean(current_metrics['shannon']))
            history['mean_carrying_capacity'].append(np.mean(current_metrics['carrying_capacity']))
            history['mean_individuals'].append(np.mean(current_metrics['individuals']))
            history['mean_species_count'].append(np.mean(current_metrics['species_count']))
            history['mean_disturbance'].append(np.mean(current_metrics['disturbance']))
            history['total_individuals'].append(np.sum(current_metrics['individuals']))
            
            # Record corporate status
            for i, corp in enumerate(env.corporations):
                history['corp_capital'][i].append(corp.capital)
                history['corp_biodiv'][i].append(corp.biodiversity_score)
                history['corp_resilience'][i].append(corp.resilience)
            
            # Print progress
            if (t + 1) % 50 == 0:
                print(f"  Step {t+1}/{steps} completed")
            
            if all(dones.values()):
                break
        
        # Record final state
        final_metrics = self.extract_grid_metrics(env.list_cells)
        
        return {
            'strategy_name': strategy_name,
            'initial_metrics': initial_metrics,
            'final_metrics': final_metrics,
            'history': history,
            'steps_completed': t + 1,
            'action_probabilities': action_probabilities,
            'env': env  # Save final environment state for analysis
        }
    
    def run_mixed_strategy_comparison(self, strategy_configs: List[Dict], steps: int = 100, seed: int = 42):
        """
        Run comparison analysis for multiple mixed strategies
        
        Parameters:
        -----------
        strategy_configs : list
            List of dictionaries with 'name' and 'probabilities' keys
        steps : int
            Number of simulation steps
        seed : int
            Random seed for reproducibility
            
        Returns:
        --------
        dict : Dictionary containing all mixed strategy results
        """
        mixed_results = {}
        
        print("Starting mixed strategy comparison analysis...")
        print(f"Will compare {len(strategy_configs)} mixed strategies")
        
        for config in strategy_configs:
            result = self.simulate_mixed_strategy(
                strategy_name=config['name'],
                action_probabilities=config['probabilities'],
                steps=steps,
                seed=seed
            )
            mixed_results[config['name']] = result
        
        print("\nMixed strategy comparison analysis completed!")
        return mixed_results
    
    def visualize_mixed_strategy_comparison(self, mixed_results: Dict, save_plots: bool = False):
        """
        Visualize comparison between mixed strategies
        
        Parameters:
        -----------
        mixed_results : dict
            Results from run_mixed_strategy_comparison
        save_plots : bool
            Whether to save plots to files
        """
        if not mixed_results:
            print("No mixed strategy results to visualize!")
            return
        
        results_list = list(mixed_results.values())
        colors = ['darkgreen', 'darkorange', 'darkblue', 'darkred', 'purple']
        line_styles = ['-', '--', '-.', ':', '-']
        
        # 1. Ecological indicators over time
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        metrics = ['mean_shannon', 'mean_carrying_capacity', 'mean_individuals', 
                  'mean_species_count', 'mean_disturbance', 'total_individuals']
        
        metric_titles = [
            'Shannon Diversity Index',
            'Mean Carrying Capacity', 
            'Mean Individuals per Cell',
            'Mean Species Count per Cell',
            'Mean Disturbance Level',
            'Total Individuals'
        ]
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row = i // 3
            col = i % 3
            
            for j, result in enumerate(results_list):
                history = result['history']
                steps = len(history[metric])
                
                axes[row, col].plot(range(steps), history[metric], 
                                  label=result['strategy_name'], 
                                  color=colors[j % len(colors)], 
                                  linestyle=line_styles[j % len(line_styles)],
                                  linewidth=3, 
                                  alpha=0.8)
            
            axes[row, col].set_title(title, fontweight='bold', fontsize=12)
            axes[row, col].set_xlabel('Steps')
            axes[row, col].set_ylabel(title.replace('Mean ', '').replace('Total ', ''))
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.suptitle('Mixed Strategy Comparison: Key Ecological Indicators Over Time', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('mixed_strategy_ecological_indicators.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Corporate performance comparison
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        corp_metrics = ['corp_capital', 'corp_biodiv', 'corp_resilience']
        metric_titles = ['Corporate Capital', 'Biodiversity Score', 'Resilience']
        
        for i, (metric, title) in enumerate(zip(corp_metrics, metric_titles)):
            for j, result in enumerate(results_list):
                history = result['history']
                
                # Calculate average values for all corporations
                n_corps = len(history[metric])
                if n_corps > 0:
                    avg_values = []
                    steps = len(history[metric][0])
                    
                    for step in range(steps):
                        step_values = [history[metric][corp][step] for corp in range(n_corps)]
                        avg_values.append(np.mean(step_values))
                    
                    axes[i].plot(range(steps), avg_values,
                               label=result['strategy_name'],
                               color=colors[j % len(colors)],
                               linestyle=line_styles[j % len(line_styles)],
                               linewidth=3, 
                               alpha=0.8)
            
            axes[i].set_title(f'Average {title}', fontweight='bold')
            axes[i].set_xlabel('Steps')
            axes[i].set_ylabel(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.suptitle('Mixed Strategy Comparison: Corporate Performance Over Time', 
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        if save_plots:
            plt.savefig('mixed_strategy_corporate_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_mixed_strategy_report(self, mixed_results: Dict):
        """
        Generate summary report for mixed strategy comparison
        
        Parameters:
        -----------
        mixed_results : dict
            Results from run_mixed_strategy_comparison
            
        Returns:
        --------
        pd.DataFrame : Summary comparison table
        """
        if not mixed_results:
            print("No mixed strategy results to analyze!")
            return None
        
        print("\n" + "="*80)
        print("MIXED STRATEGY COMPARISON REPORT")
        print("="*80)
        
        # Create comparison table
        comparison_data = []
        
        for result in mixed_results.values():
            initial = result['initial_metrics']
            final = result['final_metrics']
            history = result['history']
            
            # Calculate changes
            shannon_change = np.mean(final['shannon']) - np.mean(initial['shannon'])
            capacity_change = np.mean(final['carrying_capacity']) - np.mean(initial['carrying_capacity'])
            individuals_change = np.sum(final['individuals']) - np.sum(initial['individuals'])
            species_change = np.mean(final['species_count']) - np.mean(initial['species_count'])
            disturbance_change = np.mean(final['disturbance']) - np.mean(initial['disturbance'])
            
            # Final corporate status
            final_capital = np.mean([history['corp_capital'][i][-1] for i in range(len(history['corp_capital']))])
            final_biodiv = np.mean([history['corp_biodiv'][i][-1] for i in range(len(history['corp_biodiv']))])
            final_resilience = np.mean([history['corp_resilience'][i][-1] for i in range(len(history['corp_resilience']))])
            
            # Calculate averages over entire simulation
            avg_shannon = np.mean(history['mean_shannon'])
            avg_disturbance = np.mean(history['mean_disturbance'])
            
            comparison_data.append({
                'Strategy': result['strategy_name'],
                'Shannon_Change': shannon_change,
                'Capacity_Change': capacity_change,
                'Individuals_Change': individuals_change,
                'Species_Change': species_change,
                'Disturbance_Change': disturbance_change,
                'Final_Corp_Capital': final_capital,
                'Final_Corp_Biodiv': final_biodiv,
                'Final_Corp_Resilience': final_resilience,
                'Avg_Shannon': avg_shannon,
                'Avg_Disturbance': avg_disturbance,
                'Steps_Completed': result['steps_completed']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        print(comparison_df.to_string(index=False, float_format='%.4f'))
        
        # Show action distribution
        print(f"\n{'Action Distribution:':<20}")
        print("=" * 80)
        for result in mixed_results.values():
            print(f"\n{result['strategy_name']}:")
            total_actions = sum(result['history']['action_counts'].values())
            for action, count in result['history']['action_counts'].items():
                percentage = (count / total_actions) * 100
                print(f"  {action.capitalize():<12}: {count:>6} actions ({percentage:>5.1f}%)")
        
        # Analysis conclusions
        print(f"\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        
        # Find best performers
        best_shannon = comparison_df.loc[comparison_df['Shannon_Change'].idxmax(), 'Strategy']
        best_individuals = comparison_df.loc[comparison_df['Individuals_Change'].idxmax(), 'Strategy']
        best_capital = comparison_df.loc[comparison_df['Final_Corp_Capital'].idxmax(), 'Strategy']
        lowest_disturbance = comparison_df.loc[comparison_df['Avg_Disturbance'].idxmin(), 'Strategy']
        
        print(f"🌟 Best for Shannon Diversity: {best_shannon}")
        print(f"📈 Best for Population Growth: {best_individuals}")
        print(f"💰 Best for Corporate Capital: {best_capital}")
        print(f"🌿 Lowest Environmental Disturbance: {lowest_disturbance}")
        
        return comparison_df
    
    def run_all_scenarios(self, steps: int = 100):
        """Run analysis for all scenarios"""
        scenarios = ['nothing', 'exploit', 'restore', 'greenwash', 'random']
        
        print("Starting analysis for all scenarios...")
        print(f"Parameter settings: grid_size={self.grid_size}, steps={steps}")
        
        for scenario in scenarios:
            result = self.simulate_action_scenario(scenario, steps)
            self.results[scenario] = result
        
        print("\nAll scenario analysis completed!")
        return self.results
    
    def visualize_grid_comparison(self, metric: str = 'shannon'):
        """
        Visualize grid indicator comparison across different scenarios
        
        Parameters:
        -----------
        metric : str
            Indicator to visualize: 'shannon', 'carrying_capacity', 'individuals', 'species_count', 'disturbance'
        """
        if not self.results:
            print("Please run analysis first!")
            return
        
        n_scenarios = len(self.results)
        fig, axes = plt.subplots(2, n_scenarios, figsize=(4*n_scenarios, 8))
        
        if n_scenarios == 1:
            axes = axes.reshape(-1, 1)
        
        metric_names = {
            'shannon': 'Shannon Diversity Index',
            'carrying_capacity': 'Carrying Capacity',
            'individuals': 'Number of Individuals',
            'species_count': 'Number of Species',
            'disturbance': 'Disturbance Level'
        }
        
        vmin_initial = float('inf')
        vmax_initial = float('-inf')
        vmin_final = float('inf')
        vmax_final = float('-inf')
        
        # Calculate global color range
        for scenario, result in self.results.items():
            initial_data = result['initial_metrics'][metric]
            final_data = result['final_metrics'][metric]
            
            vmin_initial = min(vmin_initial, np.min(initial_data))
            vmax_initial = max(vmax_initial, np.max(initial_data))
            vmin_final = min(vmin_final, np.min(final_data))
            vmax_final = max(vmax_final, np.max(final_data))
        
        # Draw comparison plots
        for i, (scenario, result) in enumerate(self.results.items()):
            initial_data = result['initial_metrics'][metric]
            final_data = result['final_metrics'][metric]
            
            # Initial state
            im1 = axes[0, i].imshow(initial_data, cmap='viridis', 
                                   vmin=vmin_initial, vmax=vmax_initial)
            axes[0, i].set_title(f'{scenario.capitalize()}\nInitial {metric_names[metric]}')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            
            # Final state
            im2 = axes[1, i].imshow(final_data, cmap='viridis',
                                   vmin=vmin_final, vmax=vmax_final)
            axes[1, i].set_title(f'Final {metric_names[metric]}')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
        
        # Add color bars
        plt.colorbar(im1, ax=axes[0, :], shrink=0.6, label=f'Initial {metric_names[metric]}')
        plt.colorbar(im2, ax=axes[1, :], shrink=0.6, label=f'Final {metric_names[metric]}')
        
        plt.tight_layout()
        plt.show()
    
    def visualize_history_comparison(self):
        """Visualize historical change comparison"""
        if not self.results:
            print("Please run analysis first!")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics = ['mean_shannon', 'mean_carrying_capacity', 'mean_individuals', 
                  'mean_species_count', 'mean_disturbance', 'total_individuals']
        
        metric_titles = [
            'Mean Shannon Diversity Index',
            'Mean Carrying Capacity', 
            'Mean Individuals per Cell',
            'Mean Species Count per Cell',
            'Mean Disturbance Level',
            'Total Individuals'
        ]
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
            row = i // 3
            col = i % 3
            
            for j, (scenario, result) in enumerate(self.results.items()):
                history = result['history']
                steps = len(history[metric])
                axes[row, col].plot(range(steps), history[metric], 
                                  label=scenario.capitalize(), 
                                  color=colors[j % len(colors)], 
                                  linewidth=2, alpha=0.8)
            
            axes[row, col].set_title(title, fontweight='bold')
            axes[row, col].set_xlabel('Steps')
            axes[row, col].set_ylabel(title.split(' ', 1)[1] if ' ' in title else title)
            axes[row, col].legend()
            axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def visualize_corporate_performance(self):
        """Visualize corporate performance comparison"""
        if not self.results:
            print("Please run analysis first!")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        corp_metrics = ['corp_capital', 'corp_biodiv', 'corp_resilience']
        metric_titles = ['Capital', 'Biodiversity Score', 'Resilience']
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (metric, title) in enumerate(zip(corp_metrics, metric_titles)):
            for j, (scenario, result) in enumerate(self.results.items()):
                history = result['history']
                
                # Calculate average values for all corporations
                n_corps = len(history[metric])
                if n_corps > 0:
                    avg_values = []
                    steps = len(history[metric][0])
                    
                    for step in range(steps):
                        step_values = [history[metric][corp][step] for corp in range(n_corps)]
                        avg_values.append(np.mean(step_values))
                    
                    axes[i].plot(range(steps), avg_values,
                               label=scenario.capitalize(),
                               color=colors[j % len(colors)],
                               linewidth=2, alpha=0.8)
            
            axes[i].set_title(f'Average Corporate {title}', fontweight='bold')
            axes[i].set_xlabel('Steps')
            axes[i].set_ylabel(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate summary report"""
        if not self.results:
            print("Please run analysis first!")
            return
        
        print("\n" + "="*80)
        print("ECOSYSTEM IMPACT ANALYSIS SUMMARY REPORT")
        print("="*80)
        
        # Create summary table
        summary_data = []
        
        for scenario, result in self.results.items():
            initial = result['initial_metrics']
            final = result['final_metrics']
            history = result['history']
            
            # Calculate changes
            shannon_change = np.mean(final['shannon']) - np.mean(initial['shannon'])
            capacity_change = np.mean(final['carrying_capacity']) - np.mean(initial['carrying_capacity'])
            individuals_change = np.sum(final['individuals']) - np.sum(initial['individuals'])
            species_change = np.mean(final['species_count']) - np.mean(initial['species_count'])
            disturbance_change = np.mean(final['disturbance']) - np.mean(initial['disturbance'])
            
            # Final corporate average status
            final_capital = np.mean([history['corp_capital'][i][-1] for i in range(len(history['corp_capital']))])
            final_biodiv = np.mean([history['corp_biodiv'][i][-1] for i in range(len(history['corp_biodiv']))])
            
            summary_data.append({
                'Scenario': scenario.capitalize(),
                'Shannon_Change': shannon_change,
                'Capacity_Change': capacity_change,
                'Individuals_Change': individuals_change,
                'Species_Change': species_change,
                'Disturbance_Change': disturbance_change,
                'Final_Corp_Capital': final_capital,
                'Final_Corp_Biodiv': final_biodiv,
                'Steps_Completed': result['steps_completed']
            })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False, float_format='%.3f'))
        
        print(f"\n" + "="*80)
        print("KEY INSIGHTS:")
        print("="*80)
        
        # Find best and worst scenarios
        best_shannon = df.loc[df['Shannon_Change'].idxmax(), 'Scenario']
        worst_shannon = df.loc[df['Shannon_Change'].idxmin(), 'Scenario']
        
        best_individuals = df.loc[df['Individuals_Change'].idxmax(), 'Scenario']
        worst_individuals = df.loc[df['Individuals_Change'].idxmin(), 'Scenario']
        
        highest_disturbance = df.loc[df['Disturbance_Change'].idxmax(), 'Scenario']
        
        print(f"🌟 Best for Shannon Diversity: {best_shannon}")
        print(f"💀 Worst for Shannon Diversity: {worst_shannon}")
        print(f"📈 Best for Population Growth: {best_individuals}")
        print(f"📉 Worst for Population Growth: {worst_individuals}")
        print(f"⚠️  Highest Disturbance Impact: {highest_disturbance}")
        
        return df
    
    def save_results(self, filename: str = 'ecosystem_analysis_results.pkl'):
        """Save analysis results"""
        import pickle
        
        # Remove environment objects (cannot be pickled)
        save_data = {}
        for scenario, result in self.results.items():
            save_data[scenario] = {k: v for k, v in result.items() if k != 'env'}
        
        with open(filename, 'wb') as f:
            pickle.dump({
                'env_params': self.env_params,
                'results': save_data
            }, f)
        
        print(f"Results saved to {filename}")

def main():
    """Main function - run complete analysis"""
    
    # Set parameters
    env_params = {
        'grid_size': 50,
        'n_species': 10,
        'carrying_capacity': 25,
        'disturbance': 0.0,
        'min_age': 8,
        'max_age': 10,
        'max_age_sort': False,
        'lat_steep': 0.1,
        'disp_rate': 0.45,
        'n_corporations': 3,
        'n_investors': 2,
        'max_steps': int(1e6),
        'half': True,
        'birth_first': True,
        'seed': 42
    }
    
    # Create analyzer
    analyzer = EcosystemAnalyzer(**env_params)
    
    # Run analysis (using fewer steps for testing)
    steps = 100  # Can be adjusted to larger values for complete analysis
    
    print("Starting ecosystem impact analysis...")
    print(f"Will run {steps} step simulation, analyzing 5 different corporate behavior scenarios")
    
    # Run all scenarios
    results = analyzer.run_all_scenarios(steps=steps)
    
    # Generate visualizations
    print("\nGenerating visualization results...")
    
    # 1. Grid indicator comparison
    for metric in ['shannon', 'carrying_capacity', 'individuals', 'species_count', 'disturbance']:
        print(f"Plotting {metric} grid comparison...")
        analyzer.visualize_grid_comparison(metric=metric)
    
    # 2. Historical change comparison
    print("Plotting historical change comparison...")
    analyzer.visualize_history_comparison()
    
    # 3. Corporate performance comparison
    print("Plotting corporate performance comparison...")
    analyzer.visualize_corporate_performance()
    
    # 4. Generate summary report
    print("Generating summary report...")
    summary_df = analyzer.generate_summary_report()
    
    # 5. Save results
    analyzer.save_results()
    
    print("\nAnalysis completed! All results displayed and saved.")
    
    return analyzer, results, summary_df

def main_with_mixed_strategies():
    """Main function with mixed strategy analysis included"""
    
    # Set parameters
    env_params = {
        'grid_size': 50,
        'n_species': 10,
        'carrying_capacity': 25,
        'disturbance': 0.0,
        'min_age': 8,
        'max_age': 10,
        'max_age_sort': False,
        'lat_steep': 0.1,
        'disp_rate': 0.45,
        'n_corporations': 3,
        'n_investors': 2,
        'max_steps': int(1e6),
        'half': True,
        'birth_first': True,
        'seed': 42
    }
    
    # Create analyzer
    analyzer = EcosystemAnalyzer(**env_params)
    
    # Run analysis steps
    steps = 100  # Can be adjusted to larger values for complete analysis
    
    print("Starting comprehensive ecosystem impact analysis...")
    print(f"Will run {steps} step simulation, analyzing single and mixed strategies")
    
    # 1. Run all single scenarios
    print("\n" + "="*60)
    print("PHASE 1: Single Strategy Analysis")
    print("="*60)
    results = analyzer.run_all_scenarios(steps=steps)
    
    # 2. Run mixed strategy comparison
    print("\n" + "="*60)
    print("PHASE 2: Mixed Strategy Analysis")
    print("="*60)
    
    # Define mixed strategies to compare
    mixed_strategy_configs = [
        {
            'name': '50% Exploit + 50% Restore',
            'probabilities': {'exploit': 0.5, 'restore': 0.5}
        },
        {
            'name': '50% Exploit + 50% Greenwash',
            'probabilities': {'exploit': 0.5, 'greenwash': 0.5}
        },
        {
            'name': '33% Exploit + 33% Restore + 33% Greenwash',
            'probabilities': {'exploit': 0.33, 'restore': 0.33, 'greenwash': 0.34}
        }
    ]
    
    mixed_results = analyzer.run_mixed_strategy_comparison(
        strategy_configs=mixed_strategy_configs,
        steps=steps,
        seed=42
    )
    
    # Generate visualizations
    print("\n" + "="*60)
    print("PHASE 3: Visualization and Reporting")
    print("="*60)
    
    # Single strategy visualizations
    print("Generating single strategy visualizations...")
    
    # Grid indicator comparisons
    for metric in ['shannon', 'carrying_capacity', 'individuals', 'species_count', 'disturbance']:
        print(f"Plotting {metric} grid comparison...")
        analyzer.visualize_grid_comparison(metric=metric)
    
    # Historical change comparison
    print("Plotting historical change comparison...")
    analyzer.visualize_history_comparison()
    
    # Corporate performance comparison
    print("Plotting corporate performance comparison...")
    analyzer.visualize_corporate_performance()
    
    # Mixed strategy visualizations
    print("Generating mixed strategy visualizations...")
    analyzer.visualize_mixed_strategy_comparison(mixed_results, save_plots=True)
    
    # Generate reports
    print("Generating summary reports...")
    
    # Single strategy report
    print("\n" + "="*60)
    print("Single Strategy Summary Report")
    print("="*60)
    summary_df = analyzer.generate_summary_report()
    
    # Mixed strategy report
    print("\n" + "="*60)
    print("Mixed Strategy Summary Report")
    print("="*60)
    mixed_summary_df = analyzer.generate_mixed_strategy_report(mixed_results)
    
    # Save results
    analyzer.save_results('ecosystem_analysis_complete_with_mixed.pkl')
    
    # Save mixed results separately
    if mixed_summary_df is not None:
        mixed_summary_df.to_csv('mixed_strategy_analysis_summary.csv', index=False)
        print("✅ Mixed strategy results saved to mixed_strategy_analysis_summary.csv")
    
    print("\n" + "="*60)
    print("Comprehensive analysis completed! All results displayed and saved.")
    print("="*60)
    
    return analyzer, results, summary_df, mixed_results, mixed_summary_df

if __name__ == "__main__":
    analyzer, results, summary = main()