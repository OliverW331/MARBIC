#!/usr/bin/env python3
"""
Species Characteristics Visualization Functions

This module provides visualization functions for species characteristics including:
- D, B, K disturbance sensitivity
- Maximum age distribution
- Species abundance and distribution patterns
- Age structure visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
import pandas as pd
from matplotlib.patches import Rectangle

def plot_species_disturbance_sensitivity(species_dynamic, figsize=(15, 5)):
    """
    Visualize species sensitivity toD、B disturbance sensitivity
    
    Parameters:
    -----------
    species_dynamic : SpeciesDynamic
        containing species sensitivity dataSpeciesDynamicobject
    figsize : tuple
        figure size
    """
    n_species = species_dynamic.n_species
    species_ids = np.arange(n_species)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # D disturbance sensitivity (mortality)
    axes[0].bar(species_ids, species_dynamic.D_disturbance_sensitivity, 
                color='red', alpha=0.7, edgecolor='firebrick')
    axes[0].set_title('D Disturbance Sensitivity\n(Mortality Impact)', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('Species ID')
    axes[0].set_ylabel('Sensitivity Value')
    axes[0].grid(True, alpha=0.3)
    
    # B disturbance sensitivity (birth rate)
    axes[1].bar(species_ids, species_dynamic.B_disturbance_sensitivity, 
                color='blue', alpha=0.7, edgecolor='navy')
    axes[1].set_title('B Disturbance Sensitivity\n(Birth Rate Impact)', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('Species ID')
    axes[1].set_ylabel('Sensitivity Value')
    axes[1].grid(True, alpha=0.3)
    
    # Combined comparison
    width = 0.35
    x = np.arange(n_species)
    axes[2].bar(x - width/2, species_dynamic.D_disturbance_sensitivity, width, 
                label='D (Mortality)', color='red', alpha=0.7)
    axes[2].bar(x + width/2, species_dynamic.B_disturbance_sensitivity, width, 
                label='B (Birth Rate)', color='blue', alpha=0.7)
    axes[2].set_title('Disturbance Sensitivity Comparison', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Species ID')
    axes[2].set_ylabel('Sensitivity Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def plot_species_max_age_distribution(list_cells, figsize=(12, 8)):
    """
    Visualize maximum age distribution of all species
    
    Parameters:
    -----------
    list_cells : list
        Cellobjectlist
    figsize : tuple
        figure size
    """
    # from the firstcell获取speciesage structureinformation
    sample_cell = list_cells[0]
    n_species = len(sample_cell.sp_age_dict)
    
    # extract maximum age for each species
    species_max_ages = []
    species_ids = []
    
    for species_id in range(n_species):
        max_age = len(sample_cell.sp_age_dict[species_id])
        species_max_ages.append(max_age)
        species_ids.append(species_id)
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. bar chart showing maximum age of each species
    axes[0,0].bar(species_ids, species_max_ages, color='green', alpha=0.7, edgecolor='forestgreen')
    axes[0,0].set_title('Maximum Age by Species', fontsize=12, fontweight='bold')
    axes[0,0].set_xlabel('Species ID')
    axes[0,0].set_ylabel('Maximum Age (years)')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. maximum age distribution histogram
    axes[0,1].hist(species_max_ages, bins=max(species_max_ages)-min(species_max_ages)+1, 
                   color='orange', alpha=0.7, edgecolor='orangered')
    axes[0,1].set_title('Distribution of Maximum Ages', fontsize=12, fontweight='bold')
    axes[0,1].set_xlabel('Maximum Age (years)')
    axes[0,1].set_ylabel('Number of Species')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. number of species grouped by maximum age
    age_counts = pd.Series(species_max_ages).value_counts().sort_index()
    axes[1,0].bar(age_counts.index, age_counts.values, color='purple', alpha=0.7, edgecolor='indigo')
    axes[1,0].set_title('Number of Species by Maximum Age', fontsize=12, fontweight='bold')
    axes[1,0].set_xlabel('Maximum Age (years)')
    axes[1,0].set_ylabel('Number of Species')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. cumulative distribution
    sorted_ages = np.sort(species_max_ages)
    cumulative = np.arange(1, len(sorted_ages) + 1) / len(sorted_ages)
    axes[1,1].plot(sorted_ages, cumulative, 'o-', color='teal', linewidth=2, markersize=4)
    axes[1,1].set_title('Cumulative Distribution of Maximum Ages', fontsize=12, fontweight='bold')
    axes[1,1].set_xlabel('Maximum Age (years)')
    axes[1,1].set_ylabel('Cumulative Proportion')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 打印statisticsinformation
    print("=== Maximum Age Statistics ===")
    print(f"Mean maximum age: {np.mean(species_max_ages):.2f} years")
    print(f"Median maximum age: {np.median(species_max_ages):.2f} years")
    print(f"Min maximum age: {min(species_max_ages)} years")
    print(f"Max maximum age: {max(species_max_ages)} years")
    print(f"Standard deviation: {np.std(species_max_ages):.2f} years")

def plot_species_age_structure(list_cells, species_ids=None, figsize=(15, 10)):
    """
    visualize age structure of selected species
    
    Parameters:
    -----------
    list_cells : list
        Cellobjectlist
    species_ids : list, optional
        species to visualizeIDlist，ifNone则show top9species
    figsize : tuple
        figure size
    """
    sample_cell = list_cells[0]
    n_species = len(sample_cell.sp_age_dict)
    
    if species_ids is None:
        species_ids = list(range(min(9, n_species)))  # show top9species
    
    # calculate total age structure（allcelltotal sum）
    total_age_structure = {}
    for species_id in species_ids:
        max_age = len(sample_cell.sp_age_dict[species_id])
        total_ages = np.zeros(max_age)
        
        for cell in list_cells:
            total_ages += cell.sp_age_dict[species_id]
        
        total_age_structure[species_id] = total_ages
    
    # create subplots
    n_species_to_plot = len(species_ids)
    rows = int(np.ceil(n_species_to_plot / 3))
    cols = min(3, n_species_to_plot)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_species_to_plot == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.flatten()
    else:
        axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, n_species_to_plot))
    
    for idx, species_id in enumerate(species_ids):
        age_dist = total_age_structure[species_id]
        ages = np.arange(len(age_dist))
        
        axes[idx].bar(ages, age_dist, color=colors[idx], alpha=0.7, 
                     edgecolor='black', linewidth=0.5)
        axes[idx].set_title(f'Species {species_id} Age Structure', fontweight='bold')
        axes[idx].set_xlabel('Age (years)')
        axes[idx].set_ylabel('Number of Individuals')
        axes[idx].grid(True, alpha=0.3)
        
        # 添加statisticsinformation
        total_individuals = np.sum(age_dist)
        mean_age = np.sum(ages * age_dist) / total_individuals if total_individuals > 0 else 0
        axes[idx].text(0.7, 0.9, f'Total: {int(total_individuals)}\nMean age: {mean_age:.1f}', 
                      transform=axes[idx].transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                      facecolor="white", alpha=0.8))
    
    # hide redundant subplots
    for idx in range(n_species_to_plot, len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.show()

def plot_species_abundance_patterns(list_cells, figsize=(15, 10)):
    """
    visualize species abundance patterns
    
    Parameters:
    -----------
    list_cells : list
        Cellobjectlist
    figsize : tuple
        figure size
    """
    n_species = len(list_cells[0].species_hist)
    
    # calculate每species的totalindividual数
    total_abundance = np.zeros(n_species)
    for cell in list_cells:
        total_abundance += cell.species_hist
    
    # calculate occurrence for each speciescellquantity
    species_occurrence = np.zeros(n_species)
    for species_id in range(n_species):
        for cell in list_cells:
            if cell.species_hist[species_id] > 0:
                species_occurrence[species_id] += 1
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. sort species by total abundance
    sorted_indices = np.argsort(total_abundance)[::-1]
    axes[0,0].bar(range(n_species), total_abundance[sorted_indices], 
                  color='steelblue', alpha=0.7)
    axes[0,0].set_title('Species Total Abundance (Ranked)', fontweight='bold')
    axes[0,0].set_xlabel('Species Rank')
    axes[0,0].set_ylabel('Total Individuals')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. abundance分布（logarithmic scale）
    axes[0,1].hist(total_abundance[total_abundance > 0], bins=20, 
                   color='coral', alpha=0.7, edgecolor='firebrick')
    axes[0,1].set_title('Abundance Distribution', fontweight='bold')
    axes[0,1].set_xlabel('Total Individuals')
    axes[0,1].set_ylabel('Number of Species')
    axes[0,1].set_yscale('log')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. species occurrence frequency vs abundance
    axes[1,0].scatter(species_occurrence, total_abundance, alpha=0.6, 
                      c=range(n_species), cmap='viridis', s=50)
    axes[1,0].set_title('Species Occurrence vs Abundance', fontweight='bold')
    axes[1,0].set_xlabel('Number of Cells with Species')
    axes[1,0].set_ylabel('Total Individuals')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. rare species identification
    rare_threshold = np.percentile(total_abundance[total_abundance > 0], 25)
    common_threshold = np.percentile(total_abundance, 75)
    
    rare_species = np.sum(total_abundance <= rare_threshold)
    common_species = np.sum(total_abundance >= common_threshold)
    intermediate_species = n_species - rare_species - common_species
    
    categories = ['Rare', 'Intermediate', 'Common']
    counts = [rare_species, intermediate_species, common_species]
    colors_pie = ['lightcoral', 'lightskyblue', 'lightgreen']
    
    axes[1,1].pie(counts, labels=categories, colors=colors_pie, autopct='%1.1f%%', 
                  startangle=90)
    axes[1,1].set_title('Species Abundance Categories', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print("=== Species Abundance Statistics ===")
    print(f"Total number of species: {n_species}")
    print(f"Species with individuals: {np.sum(total_abundance > 0)}")
    print(f"Rare species (≤{rare_threshold:.0f} individuals): {rare_species}")
    print(f"Common species (≥{common_threshold:.0f} individuals): {common_species}")
    print(f"Most abundant species: {sorted_indices[0]} ({total_abundance[sorted_indices[0]]:.0f} individuals)")
    print(f"Least abundant species (with individuals): {sorted_indices[-1]} ({total_abundance[sorted_indices[-1]]:.0f} individuals)")

def plot_species_sensitivity_correlation(species_dynamic, figsize=(10, 8)):
    """
    Analyze correlation between D and B disturbance sensitivity
    
    Parameters:
    -----------
    species_dynamic : SpeciesDynamic
        SpeciesDynamic object containing species sensitivity data
    figsize : tuple
        Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    
    # 1. D vs B sensitivityscatter plot
    axes[0,0].scatter(species_dynamic.D_disturbance_sensitivity, 
                      species_dynamic.B_disturbance_sensitivity, 
                      alpha=0.7, s=50, c=range(species_dynamic.n_species), cmap='viridis')
    axes[0,0].set_xlabel('D Disturbance Sensitivity')
    axes[0,0].set_ylabel('B Disturbance Sensitivity')
    axes[0,0].set_title('D vs B Sensitivity Correlation', fontweight='bold')
    axes[0,0].grid(True, alpha=0.3)
    
    # Calculate correlation coefficient
    correlation = np.corrcoef(species_dynamic.D_disturbance_sensitivity, 
                             species_dynamic.B_disturbance_sensitivity)[0,1]
    axes[0,0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                   transform=axes[0,0].transAxes, bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor="white", alpha=0.8))
    
    # 2. Total sensitivity distribution
    total_sensitivity = species_dynamic.D_disturbance_sensitivity + species_dynamic.B_disturbance_sensitivity
    axes[0,1].hist(total_sensitivity, bins=15, color='purple', alpha=0.7, edgecolor='indigo')
    axes[0,1].set_xlabel('Total Sensitivity (D + B)')
    axes[0,1].set_ylabel('Number of Species')
    axes[0,1].set_title('Total Sensitivity Distribution', fontweight='bold')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Sensitivity heatmap by species ID
    sensitivity_matrix = np.array([species_dynamic.D_disturbance_sensitivity, 
                                  species_dynamic.B_disturbance_sensitivity])
    
    im = axes[1,0].imshow(sensitivity_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    axes[1,0].set_yticks([0, 1])
    axes[1,0].set_yticklabels(['D Sensitivity', 'B Sensitivity'])
    axes[1,0].set_xlabel('Species ID')
    axes[1,0].set_title('Sensitivity Heatmap', fontweight='bold')
    plt.colorbar(im, ax=axes[1,0])
    
    # 4. Sensitivity ratio analysis
    sensitivity_ratio = species_dynamic.D_disturbance_sensitivity / (species_dynamic.B_disturbance_sensitivity + 1e-6)
    species_ids = np.arange(species_dynamic.n_species)
    
    colors = ['red' if ratio > 1 else 'blue' for ratio in sensitivity_ratio]
    axes[1,1].bar(species_ids, sensitivity_ratio, color=colors, alpha=0.7)
    axes[1,1].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[1,1].set_xlabel('Species ID')
    axes[1,1].set_ylabel('D/B Sensitivity Ratio')
    axes[1,1].set_title('D/B Sensitivity Ratio\n(Red: D>B, Blue: B>D)', fontweight='bold')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def create_species_summary_report(species_dynamic, list_cells, save_path=None):
    """
    create comprehensive species characteristics report
    
    Parameters:
    -----------
    species_dynamic : SpeciesDynamic
        containing species sensitivity dataSpeciesDynamicobject
    list_cells : list
        Cellobjectlist
    save_path : str, optional
        path to save report
    """
    n_species = species_dynamic.n_species
    
    # collectdata
    data = []
    for species_id in range(n_species):
        # baseThisinformation
        max_age = len(list_cells[0].sp_age_dict[species_id])
        
        # abundanceinformation
        total_individuals = sum(cell.species_hist[species_id] for cell in list_cells)
        occurrence = sum(1 for cell in list_cells if cell.species_hist[species_id] > 0)
        
        # age structure
        total_age_dist = np.zeros(max_age)
        for cell in list_cells:
            total_age_dist += cell.sp_age_dict[species_id]
        
        mean_age = np.sum(np.arange(max_age) * total_age_dist) / total_individuals if total_individuals > 0 else 0
        
        data.append({
            'Species_ID': species_id,
            'Max_Age': max_age,
            'D_Sensitivity': species_dynamic.D_disturbance_sensitivity[species_id],
            'B_Sensitivity': species_dynamic.B_disturbance_sensitivity[species_id],
            'Total_Individuals': total_individuals,
            'Cell_Occurrence': occurrence,
            'Mean_Age': mean_age,
            'Abundance_Rank': 0  # fill later
        })
    
    # createDataFrame
    df = pd.DataFrame(data)
    
    # calculate abundance ranking
    df['Abundance_Rank'] = df['Total_Individuals'].rank(ascending=False, method='min').astype(int)
    
    # reorder columns
    df = df[['Species_ID', 'Abundance_Rank', 'Total_Individuals', 'Cell_Occurrence', 
             'Max_Age', 'Mean_Age', 'D_Sensitivity', 'B_Sensitivity']]
    
    # display report
    print("=" * 80)
    print("SPECIES CHARACTERISTICS SUMMARY REPORT")
    print("=" * 80)
    print(df.to_string(index=False, float_format='%.3f'))
    
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total species: {n_species}")
    print(f"Species with individuals: {(df['Total_Individuals'] > 0).sum()}")
    print(f"Average max age: {df['Max_Age'].mean():.2f} ± {df['Max_Age'].std():.2f}")
    print(f"Average D sensitivity: {df['D_Sensitivity'].mean():.3f} ± {df['D_Sensitivity'].std():.3f}")
    print(f"Average B sensitivity: {df['B_Sensitivity'].mean():.3f} ± {df['B_Sensitivity'].std():.3f}")
    print(f"D-B sensitivity correlation: {df['D_Sensitivity'].corr(df['B_Sensitivity']):.3f}")
    
    # save to file
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nReport saved to: {save_path}")
    
    return df

# example function usage
def demonstrate_species_visualization(env_or_components):
    """
    demonstrate all visualization features
    
    Parameters:
    -----------
    env_or_components : CorporateBiodiversityEnv or tuple
        environmentobjector(species_dynamic, list_cells)tuple
    """
    if hasattr(env_or_components, 'sdyn'):
        # if environmentobject
        species_dynamic = env_or_components.sdyn
        list_cells = env_or_components.list_cells
    else:
        # 如果是tuple
        species_dynamic, list_cells = env_or_components
    
    print("Visualizing species disturbance sensitivity...")
    plot_species_disturbance_sensitivity(species_dynamic)
    
    print("Visualizing species maximum age distribution...")
    plot_species_max_age_distribution(list_cells)
    
    print("Visualizing species age structure...")
    plot_species_age_structure(list_cells)
    
    print("Visualizing species abundance patterns...")
    plot_species_abundance_patterns(list_cells)
    
    print("Visualizing species sensitivity correlations...")
    plot_species_sensitivity_correlation(species_dynamic)
    
    print("Creating species summary report...")
    df = create_species_summary_report(species_dynamic, list_cells)
    
    return df

if __name__ == "__main__":
    print("Species visualization functions loaded successfully!")
    print("Use these functions to analyze species characteristics in your biodiversity model.")