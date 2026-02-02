#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Plotting module for task 3 results."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any, List

# Set plot style
sns.set_style("whitegrid")
sns.set_palette("husl")
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300
})

class Plotter:
    """Plotter class for task 3 results."""
    
    def __init__(self, output_dir: str):
        """Initialize the plotter.
        
        Args:
            output_dir: Output directory for plots.
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def plot_cdd_hdd_scatter(self, features: pd.DataFrame, locations: List[str]) -> str:
        """Plot CDD vs HDD scatter plot.
        
        Args:
            features: Features dataframe.
            locations: Location names.
            
        Returns:
            Plot file path.
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        plot_df = features.copy()
        if 'CDD24' not in plot_df.columns or 'HDD18' not in plot_df.columns:
            plot_df['CDD24'] = plot_df.get('CDD24', 0.0)
            plot_df['HDD18'] = plot_df.get('HDD18', 0.0)

        if 'location' not in plot_df.columns:
            if len(locations) == len(plot_df):
                plot_df['location'] = locations
            else:
                plot_df['location'] = [f'Loc{i+1}' for i in range(len(plot_df))]

        plot_df = plot_df[['CDD24', 'HDD18', 'location']].dropna()

        if plot_df.empty:
            ax.text(0.5, 0.5, 'No valid CDD/HDD data', ha='center', va='center')
        else:
            for _, row in plot_df.iterrows():
                ax.scatter(
                    row['CDD24'],
                    row['HDD18'],
                    label=row['location'],
                    s=100,
                    alpha=0.7
                )
        
        ax.set_title('Cooling Degree Days vs Heating Degree Days')
        ax.set_xlabel('CDD24 (°C·day)')
        ax.set_ylabel('HDD18 (°C·day)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, 'Fig_T3_1_CDD_HDD_scatter.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_feature_bar(self, features: pd.DataFrame, location: str) -> str:
        """Plot feature bar chart.
        
        Args:
            features: Features dataframe.
            location: Location name.
            
        Returns:
            Plot file path.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Select relevant features
        feature_cols = ['HDD18', 'CDD24', 'S_summer', 'S_winter', 'Cloud_proxy', 'Humid_proxy', 'T_range']
        safe_features = features.copy()
        for col in feature_cols:
            if col not in safe_features.columns:
                safe_features[col] = 0.0
        feature_data = safe_features[feature_cols].iloc[0]

        # Normalize each feature by a meaningful reference range so small proxies are visible
        ref_ranges = {
            'HDD18': 120000.0,
            'CDD24': 30000.0,
            'S_summer': 600000.0,
            'S_winter': 600000.0,
            'Cloud_proxy': 1.0,
            'Humid_proxy': 100.0,
            'T_range': 20.0
        }
        normalized_vals = []
        for col in feature_cols:
            denom = ref_ranges.get(col, 1.0)
            denom = denom if denom > 0 else 1.0
            normalized_vals.append(float(feature_data[col]) / denom)
        normalized_data = pd.Series(normalized_vals, index=feature_cols)

        normalized_data.plot(kind='bar', ax=ax)
        
        ax.set_title(f'Climate Features for {location}')
        ax.set_xlabel('Feature')
        ax.set_ylabel('Normalized Value')
        ax.set_xticklabels([col.replace('_', ' ').title() for col in feature_cols], rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        output_path = os.path.join(self.output_dir, 'Fig_T3_2_feature_bar_or_radar.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_rules(self, rules: Dict[str, List[str]]) -> str:
        """Plot rules visualization.
        
        Args:
            rules: Rules dictionary.
            
        Returns:
            Plot file path.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create a text box with rules
        rule_text = "Rules:\n\n"
        for target, rule_list in rules.items():
            rule_text += f"{target}:\n"
            for rule in rule_list[:5]:  # Show first 5 rules
                rule_text += f"  - {rule}\n"
            rule_text += "\n"
        
        ax.text(0.05, 0.95, rule_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10, 
                bbox=dict(boxstyle='round', alpha=0.1))
        
        ax.set_title('Decision Rules')
        ax.axis('off')
        
        output_path = os.path.join(self.output_dir, 'Fig_T3_3_rules.png')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_perf_compare(self, comparison: Any) -> str:
        """Plot performance comparison.
        
        Args:
            comparison: Comparison results.
            
        Returns:
            Plot file path.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        if isinstance(comparison, pd.DataFrame):
            perf_df = comparison.copy()
        else:
            location = comparison.get('location', 'Location')
            baseline = comparison['baseline']
            template = comparison['template']
            optimized = comparison['optimized']
            perf_df = pd.DataFrame([
                {'location': location, 'solution': 'baseline', 'E_cool': baseline['E_cool'], 'E_heat': baseline['E_heat'], 'ASE': baseline['ASE'], 'R': baseline['R']},
                {'location': location, 'solution': 'template', 'E_cool': template['E_cool'], 'E_heat': template['E_heat'], 'ASE': template['ASE'], 'R': template['R']},
                {'location': location, 'solution': 'optimized', 'E_cool': optimized['E_cool'], 'E_heat': optimized['E_heat'], 'ASE': optimized['ASE'], 'R': optimized['R']},
            ])

        locations = perf_df['location'].unique().tolist()
        solutions = ['baseline', 'template', 'optimized']
        colors = {'baseline': '#4C72B0', 'template': '#55A868', 'optimized': '#C44E52'}

        def plot_metric(ax, metric: str, title: str, ylabel: str):
            pivot = perf_df.pivot(index='location', columns='solution', values=metric).reindex(locations)
            x = np.arange(len(locations))
            width = 0.25
            for i, sol in enumerate(solutions):
                vals = pivot[sol].values if sol in pivot.columns else np.zeros(len(locations))
                ax.bar(x + (i - 1) * width, vals, width=width, label=sol.title(), color=colors.get(sol))
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xticks(x)
            ax.set_xticklabels(locations, rotation=0)
            ax.grid(True, alpha=0.3, axis='y')
            ax.legend()

        plot_metric(axes[0, 0], 'E_cool', 'Cooling Energy (kWh)', 'Energy (kWh)')
        plot_metric(axes[0, 1], 'E_heat', 'Heating Energy (kWh)', 'Energy (kWh)')
        plot_metric(axes[1, 0], 'ASE', 'ASE (hours)', 'Hours')
        plot_metric(axes[1, 1], 'R', 'R Score', 'Score')
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, 'Fig_T3_4_perf_compare.png')
        plt.savefig(output_path)
        plt.close()
        
        return output_path
    
    def plot_riyadh_external(self, riyadh_results: Dict[str, Any]) -> str:
        """Plot Riyadh external validation.
        
        Args:
            riyadh_results: Riyadh validation results.
            
        Returns:
            Plot file path.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        # Adjust margins to accommodate all decorations
        fig.subplots_adjust(bottom=0.2, top=0.9)
        
        # Extract data
        baseline = riyadh_results['baseline']
        template = riyadh_results['template']
        optimized = riyadh_results['optimized']
        
        # Calculate savings
        if baseline['E_cool'] > 0:
            cooling_savings = (baseline['E_cool'] - template['E_cool']) / baseline['E_cool'] * 100
        else:
            cooling_savings = 0.0
        
        if baseline['E_heat'] > 0:
            heating_savings = (baseline['E_heat'] - template['E_heat']) / baseline['E_heat'] * 100
        else:
            heating_savings = 0.0
        
        if baseline['ASE'] > 0:
            ase_reduction = (baseline['ASE'] - template['ASE']) / baseline['ASE'] * 100
        else:
            ase_reduction = 0.0
        
        savings = [cooling_savings, heating_savings, ase_reduction]
        labels = ['Cooling Savings', 'Heating Savings', 'ASE Reduction']
        
        ax.bar(labels, savings, color=['green', 'blue', 'purple'])
        ax.axhline(0, color='black', linewidth=0.5)
        
        ax.set_title('Riyadh External Validation: Savings vs Baseline')
        ax.set_ylabel('Savings (%)')
        # First set ticks, then set labels
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha='right')
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(savings):
            ax.text(i, v + 0.5, f'{v:.1f}%', ha='center')
        
        output_path = os.path.join(self.output_dir, 'Fig_T3_5_Riyadh_external.png')
        # Already adjusted margins with subplots_adjust, no need for tight_layout
        plt.savefig(output_path)
        plt.close()
        
        return output_path

def plot_cdd_hdd_scatter(features: pd.DataFrame, locations: List[str], output_dir: str) -> str:
    """Plot CDD vs HDD scatter plot.
    
    Args:
        features: Features dataframe.
        locations: Location names.
        output_dir: Output directory.
        
    Returns:
        Plot file path.
    """
    plotter = Plotter(output_dir)
    return plotter.plot_cdd_hdd_scatter(features, locations)

def plot_feature_bar(features: pd.DataFrame, location: str, output_dir: str) -> str:
    """Plot feature bar chart.
    
    Args:
        features: Features dataframe.
        location: Location name.
        output_dir: Output directory.
        
    Returns:
        Plot file path.
    """
    plotter = Plotter(output_dir)
    return plotter.plot_feature_bar(features, location)

def plot_rules(rules: Dict[str, List[str]], output_dir: str) -> str:
    """Plot rules visualization.
    
    Args:
        rules: Rules dictionary.
        output_dir: Output directory.
        
    Returns:
        Plot file path.
    """
    plotter = Plotter(output_dir)
    return plotter.plot_rules(rules)

def plot_perf_compare(comparison: Any, output_dir: str) -> str:
    """Plot performance comparison.
    
    Args:
        comparison: Comparison results.
        output_dir: Output directory.
        
    Returns:
        Plot file path.
    """
    plotter = Plotter(output_dir)
    return plotter.plot_perf_compare(comparison)

def plot_riyadh_external(riyadh_results: Dict[str, Any], output_dir: str) -> str:
    """Plot Riyadh external validation.
    
    Args:
        riyadh_results: Riyadh validation results.
        output_dir: Output directory.
        
    Returns:
        Plot file path.
    """
    plotter = Plotter(output_dir)
    return plotter.plot_riyadh_external(riyadh_results)
