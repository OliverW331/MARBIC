# MARBIC
**Multi-Agent Reinforcement Learning for Biodiversity, Investor & Corporate Dynamics**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![OpenAI Gym](https://img.shields.io/badge/OpenAI%20Gym-Compatible-green.svg)](https://gym.openai.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

MARBIC is a sophisticated multi-agent reinforcement learning environment that simulates the complex interactions between corporate behavior, investor decisions, and biodiversity conservation. The platform models how economic activities impact ecological systems and provides tools for analyzing ecosystem dynamics.

## 🌟 Features

- **Multi-Agent Environment**: Simulates interactions between corporations and investors
- **Biodiversity Modeling**: Advanced species dynamics with age structure, disturbance sensitivity, and carrying capacity
- **Economic Simulation**: Corporate capital management, investor portfolios, and environmental impact costs
- **Comprehensive Visualization**: Tools for analyzing species patterns, ecosystem impacts, and corporate performance
- **Configurable Parameters**: Adjustable grid size, species count, disturbance levels, and economic parameters
- **Research-Ready**: Built for academic research in environmental economics and multi-agent systems

## 🏗️ Project Structure

```
MARBIC/
├── Agents/                          # Agent implementations
│   ├── corporation.py              # Corporate agent behavior
│   ├── investor.py                 # Investor agent behavior
│   └── reward.py                   # Reward calculation system
├── Env/                            # Environment components
│   ├── cell.py                     # Grid cell implementation
│   ├── species_dynamic.py          # Species population dynamics
│   └── species.py                  # Species characteristics
├── Disturbance/                    # Disturbance modeling
│   ├── disturbance.py             # Disturbance generation
│   └── disturbance_demo.ipynb     # Disturbance examples
├── gym_marbic.py                   # Main Gym environment
├── species_visualization.py        # Species analysis tools
├── ecosystem_impact_analysis.py    # Ecosystem impact analyzer
├── demo_*.py                       # Demonstration scripts
├── *.ipynb                         # Jupyter notebooks
└── *_GUIDE.md                      # User guides
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/OliverW331/MARBIC.git
cd MARBIC

# Install dependencies
pip install numpy pandas matplotlib seaborn gymnasium scipy jupyter
```

### Basic Usage

```python
from gym_marbic import CorporateBiodiversityEnv

# Create environment
env = CorporateBiodiversityEnv(
    grid_size=50,
    n_species=25,
    n_corporations=3,
    n_investors=2,
    max_steps=1000
)

# Reset environment
obs = env.reset()

# Run simulation
for step in range(100):
    # Sample random actions (replace with your policy)
    actions = {}
    for agent_id in env.agent_ids:
        actions[agent_id] = env.action_spaces[agent_id].sample()
    
    # Step environment
    obs, rewards, dones, infos = env.step(actions)
    
    if all(dones.values()):
        break
```

## 🎯 Key Components

### 1. Multi-Agent Environment

The environment supports multiple types of agents:

- **Corporations**: Can exploit resources, restore ecosystems, engage in greenwashing, or invest in resilience
- **Investors**: Allocate capital across corporations based on performance and sustainability metrics

### 2. Biodiversity Simulation

- **Species Dynamics**: Birth, death, aging, and dispersal processes
- **Disturbance Modeling**: Environmental and anthropogenic disturbances
- **Carrying Capacity**: Resource limitations and competition
- **Age Structure**: Age-based mortality and reproduction

### 3. Economic Modeling

- **Corporate Actions**:
  - `EXPLOIT`: Extract resources for profit but damage environment
  - `RESTORE`: Invest in ecosystem restoration
  - `GREEN_WASH`: Marketing without real environmental benefit
  - `RESILIENCE`: Build adaptive capacity

- **Investor Actions**:
  - Portfolio allocation across corporations
  - Investment decisions based on performance metrics

## 📊 Visualization & Analysis

### Species Visualization

```python
from species_visualization import *

# Analyze species characteristics
plot_species_disturbance_sensitivity(env.sdyn)
plot_species_max_age_distribution(env.list_cells)
plot_species_abundance_patterns(env.list_cells)

# Generate comprehensive report
report = create_species_summary_report(env.sdyn, env.list_cells)
```

### Ecosystem Impact Analysis

```python
from ecosystem_impact_analysis import EcosystemAnalyzer

# Create analyzer
analyzer = EcosystemAnalyzer()

# Run scenario analysis
results = analyzer.run_scenario_comparison(
    scenarios=['baseline', 'aggressive_exploit', 'conservation_focus'],
    n_steps=500
)

# Generate insights
insights = analyzer.generate_policy_insights(results)
```

## 🎮 Action Spaces

### Corporate Actions

```python
action_space = spaces.Dict({
    "action_type": spaces.Discrete(5),  # NO_ACTION, EXPLOIT, RESTORE, GREEN_WASH, RESILIENCE
    "cell": spaces.Discrete(n_cells)    # Which cell to act on
})
```

### Investor Actions

```python
action_space = spaces.Dict({
    "action_type": spaces.Discrete(2),        # NO_ACTION, INVEST
    "weights": spaces.Box(                    # Portfolio allocation
        low=0.0, high=1.0, 
        shape=(n_corporations,), 
        dtype=np.float32
    )
})
```

## 📈 Observation Spaces

### Corporate Observations

- Capital level
- Biodiversity score
- Resilience capacity
- Mean environmental disturbance
- Mean biodiversity index

### Investor Observations

- Available cash
- Current portfolio allocation
- Corporate capital levels
- Corporate biodiversity scores

## 🔧 Configuration

Key parameters can be adjusted:

```python
env = CorporateBiodiversityEnv(
    grid_size=100,              # Environment grid size
    n_species=25,               # Number of species
    carrying_capacity=25,       # Cell carrying capacity
    disturbance=0.1,           # Base disturbance level
    n_corporations=3,           # Number of corporate agents
    n_investors=2,              # Number of investor agents
    max_steps=10000,           # Maximum episode length
    seed=42                     # Random seed
)
```

## 📚 Documentation

- [`SPECIES_VISUALIZATION_GUIDE.md`](SPECIES_VISUALIZATION_GUIDE.md) - Species analysis tools
- [`ECOSYSTEM_ANALYSIS_GUIDE.md`](ECOSYSTEM_ANALYSIS_GUIDE.md) - Ecosystem impact analysis
- [`demo_*.py`](.) - Example scripts and demonstrations
- [`*.ipynb`](.) - Interactive Jupyter notebooks

## 🧪 Example Notebooks

- `demo_cell.ipynb` - Basic cell and species interactions
- `ecosystem_impact_showcase.ipynb` - Comprehensive ecosystem analysis
- `sensitivity_analysis.ipynb` - Parameter sensitivity studies
- `disturbance_demo.ipynb` - Disturbance modeling examples

## 🤝 Contributing

We welcome contributions! Please see our contributing guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Research Applications

MARBIC is designed for research in:

- **Environmental Economics**: Modeling corporate environmental responsibility
- **Multi-Agent Systems**: Complex agent interactions and emergent behavior
- **Conservation Biology**: Biodiversity dynamics under human pressure
- **Policy Analysis**: Evaluating environmental regulations and incentives
- **Reinforcement Learning**: Multi-agent learning in complex environments

## 📞 Contact

- **Repository**: [https://github.com/OliverW331/MARBIC](https://github.com/OliverW331/MARBIC)
- **Issues**: [GitHub Issues](https://github.com/OliverW331/MARBIC/issues)

## 🙏 Acknowledgments

This project builds upon research in multi-agent reinforcement learning, ecological modeling, and environmental economics. We thank the open-source community for providing the foundational tools that make this research possible.

---

**Keywords**: Multi-agent reinforcement learning, biodiversity conservation, corporate sustainability, environmental economics, ecosystem modeling, agent-based simulation
