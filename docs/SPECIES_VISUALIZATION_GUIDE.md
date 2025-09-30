# Species Visualization Functions User Guide

This module provides a series of functions to visualize various species characteristics, including sensitivity to disturbance, age distribution, abundance patterns, etc.

## Main Functions

### 1. `plot_species_disturbance_sensitivity(species_dynamic)`
Visualize each species' sensitivity to D and B disturbances
- **Input**: SpeciesDynamic object
- **Output**: Three subplots showing D sensitivity, B sensitivity and comparison

### 2. `plot_species_max_age_distribution(list_cells)`
Visualize maximum age distribution of each species
- **Input**: Cell object list
- **Output**: Four subplots showing different statistical perspectives of maximum age

### 3. `plot_species_age_structure(list_cells, species_ids=None)`
Visualize age structure of selected species
- **Input**: Cell object list, optional species ID list
- **Output**: Age structure histograms for each species

### 4. `plot_species_abundance_patterns(list_cells)`
Visualize species abundance patterns
- **Input**: Cell object list
- **Output**: Abundance ranking, distribution, rarity analysis, etc.

### 5. `plot_species_sensitivity_correlation(species_dynamic)`
Analyze correlation between D and B disturbance sensitivity
- **Input**: SpeciesDynamic object
- **Output**: Correlation analysis, sensitivity ratios, etc.

### 6. `create_species_summary_report(species_dynamic, list_cells, save_path=None)`
Create comprehensive species characteristics report
- **Input**: SpeciesDynamic object, Cell object list, optional save path
- **Output**: pandas DataFrame and CSV file

## Quick Usage Example

```python
from species_visualization import *

# Assuming you already have an environment object
env = CorporateBiodiversityEnv()
obs = env.reset()

# 1. View species disturbance sensitivity
plot_species_disturbance_sensitivity(env.sdyn)

# 2. View species maximum age distribution
plot_species_max_age_distribution(env.list_cells)

# 3. View specific species age structure
plot_species_age_structure(env.list_cells, species_ids=[0, 1, 2, 3])

# 4. Analyze species abundance patterns
plot_species_abundance_patterns(env.list_cells)

# 5. Analyze sensitivity correlations
plot_species_sensitivity_correlation(env.sdyn)

# 6. Generate comprehensive report
df_report = create_species_summary_report(env.sdyn, env.list_cells, 
                                        save_path='my_species_report.csv')
```

## One-click Demonstration of All Functions

```python
# Use demonstrate_species_visualization function to display all visualizations at once
df_report = demonstrate_species_visualization(env)
```

## Usage in Jupyter Notebook

In Jupyter Notebook, these functions will directly display charts. Remember to import necessary libraries:

```python
import matplotlib.pyplot as plt
%matplotlib inline

from species_visualization import *
```

## Custom Parameters

Most functions support the `figsize` parameter to adjust chart size:

```python
plot_species_disturbance_sensitivity(env.sdyn, figsize=(18, 6))
plot_species_max_age_distribution(env.list_cells, figsize=(15, 10))
```

## Output Description

- **Charts**: All functions will generate matplotlib charts
- **Statistical Information**: Some functions will print key statistical information to the console
- **CSV Report**: `create_species_summary_report` will generate detailed CSV report

## Practical Usage Recommendations

1. **At experiment start**: Use these functions to understand initial species characteristics
2. **During experiment**: Regularly check species abundance and age structure changes
3. **After experiment completion**: Generate comprehensive report analyzing overall patterns
4. **Parameter tuning**: Compare species characteristics under different parameter settings

## Extended Functions

You can easily create your own analysis based on these functions:

```python
# For example, create time series analysis
def track_species_changes_over_time(env, n_steps=10):
    abundance_history = []
    
    for step in range(n_steps):
        # Record current abundance
        current_abundance = [np.sum([cell.species_hist[i] for cell in env.list_cells]) 
                           for i in range(env.n_species)]
        abundance_history.append(current_abundance)
        
        # Execute one simulation step
        actions = {}  # Your action logic
        obs, rewards, dones, infos = env.step(actions)
    
    # Visualize changes
    abundance_history = np.array(abundance_history)
    plt.figure(figsize=(12, 8))
    for species_id in range(min(5, env.n_species)):
        plt.plot(abundance_history[:, species_id], label=f'Species {species_id}')
    plt.xlabel('Time Steps')
    plt.ylabel('Total Abundance')
    plt.title('Species Abundance Over Time')
    plt.legend()
    plt.show()
```