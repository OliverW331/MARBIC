# Ecosystem Impact Analysis Tool User Guide

## 📋 Overview

This tool contains two main files for analyzing the impact of different corporate behaviors on ecosystem grid maps:

1. **`ecosystem_impact_analysis.py`** - Core analysis script
2. **`ecosystem_impact_showcase.ipynb`** - Visualization showcase notebook

## 🎯 Analysis Content

### Corporate Behavior Types
- **Nothing** - Take no action
- **Exploit** - Only resource extraction
- **Restore** - Only ecological restoration  
- **Greenwash** - Only greenwashing
- **Random** - Random behavior

### Ecological Indicators
- **Shannon Diversity Index** - Measures biodiversity
- **Carrying Capacity** - Environmental carrying capacity
- **Number of Individuals** - Total individuals in each cell
- **Number of Species** - Total species in each cell
- **Disturbance Level** - Degree of environmental disturbance

### Corporate Indicators
- **Capital** - Capital accumulated by corporations
- **Biodiversity Score** - Corporate biodiversity performance
- **Resilience** - Corporate environmental adaptability

## 🚀 Quick Start

### Method 1: Using Python Script

```python
from ecosystem_impact_analysis import EcosystemAnalyzer

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

# Run analysis
results = analyzer.run_all_scenarios(steps=200)

# Generate visualizations
analyzer.visualize_grid_comparison('shannon')
analyzer.visualize_history_comparison()
analyzer.visualize_corporate_performance()

# Generate report
summary = analyzer.generate_summary_report()
```

### Method 2: Run Complete Analysis Directly

```bash
cd /Users/muchenwang/Documents/GitHub/MARBIC
python ecosystem_impact_analysis.py
```

### Method 3: Using Jupyter Notebook

1. Open `ecosystem_impact_showcase.ipynb`
2. Run all cells in order
3. View generated visualization results

## ⚙️ Parameter Description

### Environment Parameters
- `grid_size`: Grid size (recommended: 20-50)
- `n_species`: Number of species (recommended: 5-15)
- `carrying_capacity`: Carrying capacity per cell (recommended: 10-50)
- `disturbance`: Initial disturbance level (Usually set to0.0)
- `min_age`/`max_age`: Species minimum/maximum age
- `disp_rate`: Species dispersal rate (recommended: 0.3-0.6)
- `n_corporations`: Number of corporations (recommended: 2-5)
- `n_investors`: Number of investors (recommended: 1-3)

### Simulation Parameters
- `steps`: Number of simulation steps
  - Quick test: 50-100 steps
  - Detailed analysis: 200-500 steps
  - Complete research: 1000+ steps

## 📊 Output Files

### Automatically Generated Files
1. **`ecosystem_analysis_summary.csv`** - Summary data table
2. **`ecosystem_analysis_complete.pkl`** - Complete analysis results
3. **`ecosystem_analysis_report.txt`** - Detailed text report

### Visualization Charts
1. **grid comparison chart** - Show spatial distribution of initial vs final states
2. **historical trend chart** - Show changes of indicators over time
3. **Corporate Performance Chart** - Show changes in corporate indicators
4. **Comprehensive Ranking Chart** - Show comprehensive performance ranking of scenarios

## 🔧 Custom Analysis

### Adding New Behavior Types

```python
def _generate_actions(self, env, action_type: str):
    actions = {}
    
    # Add new behavior types
    if action_type == 'mixed':
        for aid in env.corp_ids:
            # 50% probability exploit, 50% probability restore
            action = 1 if np.random.random() < 0.5 else 2
            actions[aid] = {
                "action_type": action,
                "cell": np.random.randint(env.n_cells)
            }
    # ... other behavior types
```

### Adding New Evaluation Metrics

```python
def extract_grid_metrics(self, list_cells):
    # Add new indicators
    custom_metric_matrix = np.zeros((grid_size, grid_size))
    
    for i, cell in enumerate(list_cells):
        row = i // grid_size
        col = i % grid_size
        # Calculate custom indicators
        custom_metric_matrix[row, col] = your_custom_calculation(cell)
    
    return {
        # ... existing indicators
        'custom_metric': custom_metric_matrix
    }
```

## 📈 Result Interpretation

### Grid Comparison Chart Interpretation
- **Darker colors** = Higher indicator values
- **Spatial distribution** = Show geographic distribution characteristics of impacts
- **Initial vs Final** = Show spatial patterns of change

### Historical Trend Chart Interpretation
- **Upward trend** = Indicators improving
- **Downward trend** = Indicators deteriorating
- **Volatility** = System stability
- **Convergence** = Whether system reaches equilibrium

### Comprehensive Ranking Interpretation
Ranking based on weighted scores:
- Shannon diversity change: 40%weight
- Individual count change: 30%weight  
- Corporate biodiversity score: 30%weight

## ⚠️ Precautions

### Performance Considerations
- Large grid (>50x50) + Long duration (>500 steps) May require longer runtime
- recommendedfirst use smallparametertest，run complete analysis after confirmation

### Memory Usage
- Each scenario saves complete historical data
- if memory issues occur，canReduceNumber of simulation stepsorGrid size

### Visualization Issues
- If running in headless environment, set `matplotlib.use('Agg')`
- Some environments may require additional graphics dependencies

## 🛠️ Troubleshooting

### Common Issues

1. **Import Error**
   ```bash
   # Ensure in correct directory
   cd /Users/muchenwang/Documents/GitHub/MARBIC
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **Insufficient Memory**
   - Reduce `grid_size` parameter
   - Reduce `steps` parameter
   - Reduce `n_species` parameter

3. **Visualization Issues**
   ```python
   import matplotlib
   matplotlib.use('Agg')  # Use non-interactive backend
   ```

4. **Runtime Too Long**
   - use smaller testparameter
   - Consider parallel processing (advanced usage)

## 📚 Extended Reading

### Related Files
- `gym_marbic.py` - Main environment implementation
- `species_dynamic.py` - Species dynamics simulation
- `species_visualization.py` - Species characteristic visualization
- `cell.py` - Basic cell class definition

### Theoretical Background
- Ecosystem dynamics
- Multi-agent systems
- Reinforcement learning environment
- Sustainable development indicators

## 📞 Support

如有问题orrecommended，please check:
1. parametersettings are reasonable
2. Whether dependencies are completely installed
3. Whether file paths are correct
4. Whether system resources are sufficient

---

**Version**: 1.0  
**Update Date**: 20259  
**Compatibility**: Python 3.8+, NumPy, Matplotlib, Seaborn, Pandas