# Species Visualization Functions User Guide

This module provides a series of functions to visualize various species characteristics，including sensitivity todisturbancesensitivity、age distribution、abundance patterns, etc.。

## main functions

### 1. `plot_species_disturbance_sensitivity(species_dynamic)`
can视化各species对D、B disturbancesensitivity
- **输入**: SpeciesDynamicobject
- **Display**: 三个子图展示Dsensitivity、Bsensitivityandcomparison

### 2. `plot_species_max_age_distribution(list_cells)`
can视化各species最大age distribution
- **输入**: Cellobjectlist
- **Display**: 四个子图展示最大age不同statistics视角

### 3. `plot_species_age_structure(list_cells, species_ids=None)`
can视化选定speciesage structure
- **输入**: Cellobjectlist，can选speciesIDlist
- **Display**: 各speciesage structure直方图

### 4. `plot_species_abundance_patterns(list_cells)`
visualize species abundance patterns
- **输入**: Cellobjectlist
- **Display**: abundance ranking, distribution, rarity analysis, etc.

### 5. `plot_species_sensitivity_correlation(species_dynamic)`
Analyze correlation between D and B disturbance sensitivity
- **输入**: SpeciesDynamicobject
- **Display**: correlation analysis, sensitivity ratios, etc.

### 6. `create_species_summary_report(species_dynamic, list_cells, save_path=None)`
createspecies特征comprehensivereport
- **输入**: SpeciesDynamicobject、Cellobjectlist、can选保存路径
- **输出**: pandas DataFrameandCSVfile

## 快速use示例

```python
from species_visualization import *

# 假设你已经有了environmentobject
env = CorporateBiodiversityEnv()
obs = env.reset()

# 1. 查看speciesdisturbancesensitivity
plot_species_disturbance_sensitivity(env.sdyn)

# 2. 查看species最大age distribution
plot_species_max_age_distribution(env.list_cells)

# 3. 查看特定speciesage structure
plot_species_age_structure(env.list_cells, species_ids=[0, 1, 2, 3])

# 4. Analyze species abundance patterns
plot_species_abundance_patterns(env.list_cells)

# 5. Analyze sensitivity correlations
plot_species_sensitivity_correlation(env.sdyn)

# 6. 生成comprehensivereport
df_report = create_species_summary_report(env.sdyn, env.list_cells, 
                                        save_path='my_species_report.csv')
```

## 一键demonstrationall功能

```python
# usedemonstrate_species_visualizationfunction一次性展示allcan视化
df_report = demonstrate_species_visualization(env)
```

## 在Jupyter Notebookinuse

在Jupyter Notebookin，这些function会直接Displaychart。记得导入必要库：

```python
import matplotlib.pyplot as plt
%matplotlib inline

from species_visualization import *
```

## 自定义parameter

大部分function都support`figsize`parameter来adjustmentchart大小：

```python
plot_species_disturbance_sensitivity(env.sdyn, figsize=(18, 6))
plot_species_max_age_distribution(env.list_cells, figsize=(15, 10))
```

## 输出说明

- **chart**: allfunction都会生成matplotlibchart
- **statisticsinformation**: 部分function会打印keystatisticsinformationto控制台
- **CSVreport**: `create_species_summary_report`会生成detailedCSVreport

## 实际userecommendation

1. **实验开始时**: use这些function了解speciesinitial特征
2. **实验过程in**: 定期检查speciesabundanceandage structurechange
3. **After experiment completion**: generate comprehensive report analyzing overall patterns
4. **parameter调优**: compare不同parameter设置下species特征差异

## 扩展功能

You can easily create your own analysis based on these functions：

```python
# For example, create time series analysis
def track_species_changes_over_time(env, n_steps=10):
    abundance_history = []
    
    for step in range(n_steps):
        # 记录currentabundance
        current_abundance = [np.sum([cell.species_hist[i] for cell in env.list_cells]) 
                           for i in range(env.n_species)]
        abundance_history.append(current_abundance)
        
        # execute one simulation step
        actions = {}  # 你action逻辑
        obs, rewards, dones, infos = env.step(actions)
    
    # can视化change
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