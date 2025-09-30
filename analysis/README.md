# Analysis Directory

This directory contains analysis tools, visualization modules, and comprehensive research notebooks for the MARBIC ecosystem.

## Files

### Analysis Tools

#### `ecosystem_impact_analysis.py`
Comprehensive ecosystem impact analysis tool:
- `EcosystemAnalyzer` class for scenario comparison
- Policy impact assessment
- Corporate behavior analysis
- Biodiversity impact metrics

#### `species_visualization.py`
Species analysis and visualization toolkit:
- Species disturbance sensitivity plots
- Age distribution analysis
- Abundance pattern visualization
- Comprehensive species reports

### Jupyter Notebooks

#### `ecosystem_impact_showcase.ipynb`
Interactive notebook demonstrating ecosystem analysis:
- Scenario-based analysis
- Corporate performance comparison
- Policy recommendation generation
- Visualization of ecosystem impacts

#### `sensitivity_analysis.ipynb`
Parameter sensitivity studies:
- Grid size sensitivity
- Species count impact analysis
- Disturbance level effects
- Economic parameter tuning

## Usage

### Using Analysis Tools

```python
from analysis.ecosystem_impact_analysis import EcosystemAnalyzer
from analysis.species_visualization import *

# Create analyzer
analyzer = EcosystemAnalyzer()

# Run analysis
results = analyzer.run_scenario_comparison(['baseline', 'conservation'])
```

### Running Notebooks

```bash
cd analysis/
jupyter notebook ecosystem_impact_showcase.ipynb
```

## Output

Analysis tools generate:
- CSV reports with detailed metrics
- PNG visualization files
- Statistical summaries
- Policy recommendations