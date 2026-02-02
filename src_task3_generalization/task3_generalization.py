#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CLI entry point for Task 3 generalization."""

import os
import sys
import argparse
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Any

from .adapters import read_epw
from .features import compute_climate_features, infer_params_from_results, compute_seasonal_features, classify_climate
from .mapping import fit_mapping, recommend_template
from .evaluation import evaluate_template, evaluate_baseline, compare_solutions, cross_validate
from .plots import plot_cdd_hdd_scatter, plot_feature_bar, plot_rules, plot_perf_compare, plot_riyadh_external

def setup_logging(log_dir: str) -> logging.Logger:
    """Set up logging.
    
    Args:
        log_dir: Log directory.
        
    Returns:
        Logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    logger = logging.getLogger('task3_generalization')
    logger.setLevel(logging.INFO)
    
    # File handler
    log_file = os.path.join(log_dir, 'task3_run.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def resolve_epw_path(filename: str) -> str:
    """Resolve EPW file path.
    
    Args:
        filename: EPW filename or path.
        
    Returns:
        Resolved EPW path.
    """
    # If it's already a full path, return it
    if os.path.exists(filename):
        return filename
    
    # Search in root directory
    root_path = os.path.join(os.getcwd(), filename)
    if os.path.exists(root_path):
        return root_path
    
    # Search in src directory
    src_path = os.path.join(os.getcwd(), 'src', filename)
    if os.path.exists(src_path):
        return src_path
    
    # Raise error if not found
    raise FileNotFoundError(f"EPW file not found: {filename}")

def main() -> int:
    """Main function.
    
    Returns:
        Exit code.
    """
    # Parse arguments
    parser = argparse.ArgumentParser(description='Task 3 generalization')
    parser.add_argument('--epw_train', nargs='+', required=True, help='Training EPW files')
    parser.add_argument('--epw_ext', required=True, help='External validation EPW file (Riyadh)')
    parser.add_argument('--results_dir', default='results', help='Results directory')
    parser.add_argument('--outdir', default='results/task3', help='Output directory')
    parser.add_argument('--mapping_method', choices=['rules', 'decision_tree'], default='rules', help='Mapping method')
    parser.add_argument('--seasons_mode', type=int, default=1, help='Seasons mode (1=enable)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    np.random.seed(args.seed)
    
    # Create output directories
    tables_dir = os.path.join(args.outdir, 'tables')
    figs_dir = os.path.join(args.outdir, 'figs')
    logs_dir = os.path.join(args.outdir, 'logs')
    
    os.makedirs(tables_dir, exist_ok=True)
    os.makedirs(figs_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    
    # Setup logging
    logger = setup_logging(logs_dir)
    
    try:
        logger.info('Starting Task 3 generalization')
        logger.info(f'Using mapping method: {args.mapping_method}')
        logger.info(f'Seasons mode: {args.seasons_mode}')
        
        # Resolve EPW paths
        train_epws = []
        for epw in args.epw_train:
            try:
                resolved_path = resolve_epw_path(epw)
                train_epws.append(resolved_path)
                logger.info(f'Resolved training EPW: {resolved_path}')
            except FileNotFoundError as e:
                logger.error(f"EPW not found: {epw}")
                return 1
        
        try:
            ext_epw = resolve_epw_path(args.epw_ext)
            logger.info(f'Resolved external EPW: {ext_epw}')
        except FileNotFoundError as e:
            logger.error(f"External EPW not found: {args.epw_ext}")
            return 1
        
        # Read EPW data
        logger.info('Reading EPW data')
        train_data = []
        train_meta = []
        train_locations = []
        
        for epw_path in train_epws:
            try:
                epw_df, meta = read_epw(epw_path)
                if epw_df is None:
                    logger.error(f"Failed to read EPW: {epw_path}")
                    continue
                
                train_data.append(epw_df)
                train_meta.append(meta)
                location = os.path.basename(epw_path).split('_')[0].upper()
                train_locations.append(location)
                logger.info(f'Read EPW for {location}')
            except Exception as e:
                logger.error(f"Error reading EPW {epw_path}: {e}")
                continue
        
        if not train_data:
            logger.error('No valid training EPW data')
            return 1
        
        # Read external EPW
        try:
            ext_df, ext_meta = read_epw(ext_epw)
            if ext_df is None:
                logger.error(f"Failed to read external EPW: {ext_epw}")
                return 1
            ext_location = 'Riyadh'
            logger.info(f'Read external EPW for {ext_location}')
        except Exception as e:
            logger.error(f"Error reading external EPW {ext_epw}: {e}")
            return 1
        
        # Infer parameters from results
        logger.info('Inferring parameters from results')
        params_result = infer_params_from_results(args.results_dir)
        params = params_result['params']
        
        # Log parameter inference
        for log in params_result['logs']:
            logger.info(log)
        
        logger.info(f'Inferred parameters: {params}')
        
        # Compute climate features
        logger.info('Computing climate features')
        all_features = []
        all_labels = []
        
        # Training locations
        for i, (epw_df, meta, location) in enumerate(zip(train_data, train_meta, train_locations)):
            if args.seasons_mode:
                # Compute seasonal features
                seasonal_features = compute_seasonal_features(epw_df, meta)
                for season, features in seasonal_features.items():
                    features['location'] = location
                    features['season'] = season
                    all_features.append(features)
                    all_labels.append(classify_climate(features))
            else:
                # Compute annual features
                features = compute_climate_features(epw_df, meta)
                features['location'] = location
                all_features.append(features)
                all_labels.append(classify_climate(features))
        
        # External validation (Riyadh)
        ext_features = compute_climate_features(ext_df, ext_meta)
        ext_features['location'] = ext_location
        ext_labels = classify_climate(ext_features)
        
        # Create dataframes
        features_df = pd.DataFrame(all_features)
        labels_df = pd.DataFrame(all_labels)
        
        # Save climate features and labels
        features_df.to_csv(os.path.join(tables_dir, 'climate_features.csv'), index=False)
        labels_df.to_csv(os.path.join(tables_dir, 'climate_labels.csv'), index=False)
        logger.info('Saved climate features and labels')
        
        # Prepare data for mapping
        logger.info('Preparing data for mapping')
        
        # Select features for mapping
        feature_cols = ['phi', 'HDD18', 'CDD24', 'S_summer', 'S_winter', 'Cloud_proxy', 'Humid_proxy', 'T_range']
        X = features_df[feature_cols]
        
        # Select target parameters
        # Try to find shade and mass parameters
        target_cols = []
        for col in params:
            if 'shade' in col.lower() or 'mass' in col.lower():
                target_cols.append(col)
        
        # If no target cols found, use first few params
        if not target_cols and params:
            target_cols = list(params.keys())[:2]
        
        if not target_cols:
            logger.error('No target parameters found for mapping')
            return 1
        
        # Create target dataframe
        y = pd.DataFrame()
        for col in target_cols:
            y[col] = [params.get(col, 0.0) for _ in range(len(X))]
        
        logger.info(f'Selected target parameters: {target_cols}')
        
        # Fit mapping
        logger.info(f'Fitting mapping with method: {args.mapping_method}')
        mapping = fit_mapping(X, y, method=args.mapping_method)
        
        # Get rules
        rules = mapping.get_rules()
        for target, rule_list in rules.items():
            logger.info(f'Rules for {target}:')
            for rule in rule_list:
                logger.info(f'  {rule}')
        
        # Recommend template for each training location
        logger.info('Recommending templates')
        templates = []
        
        for i, location in enumerate(train_locations):
            if args.seasons_mode:
                # Use annual features
                annual_features = compute_climate_features(train_data[i], train_meta[i])
            else:
                annual_features = all_features[i]
            
            template_result = recommend_template(mapping, annual_features)
            template = template_result['template']
            template['location'] = location
            templates.append(template)
            
            logger.info(f'Template for {location}: {template}')
            logger.info(f'Rule trace: {"\n".join(template_result["rule_trace"])}')
        
        # Recommend template for external validation (Riyadh)
        ext_template_result = recommend_template(mapping, ext_features)
        ext_template = ext_template_result['template']
        ext_template['location'] = ext_location
        templates.append(ext_template)
        
        logger.info(f'Template for {ext_location}: {ext_template}')
        logger.info(f'Rule trace: {"\n".join(ext_template_result["rule_trace"])}')
        
        # Save templates
        templates_df = pd.DataFrame(templates)
        templates_df.to_csv(os.path.join(tables_dir, 'template_params.csv'), index=False)
        logger.info('Saved template parameters')
        
        # Evaluate solutions
        logger.info('Evaluating solutions')
        
        all_comparisons = []
        
        # Evaluate training locations
        for i, (epw_path, location) in enumerate(zip(train_epws, train_locations)):
            # Get template for this location
            location_template = templates[i]
            
            # Compare solutions
            comparison = compare_solutions(epw_path, location_template, args.results_dir)
            comparison['location'] = location
            all_comparisons.append(comparison)
            
            logger.info(f'Evaluation for {location}: {comparison}')
        
        # Evaluate external validation (Riyadh)
        ext_comparison = compare_solutions(ext_epw, ext_template, args.results_dir)
        ext_comparison['location'] = ext_location
        all_comparisons.append(ext_comparison)
        
        logger.info(f'Evaluation for {ext_location}: {ext_comparison}')
        
        # Save performance comparison
        perf_data = []
        for comp in all_comparisons:
            location = comp['location']
            for sol_type in ['baseline', 'template', 'optimized']:
                sol_data = comp[sol_type]
                perf_data.append({
                    'location': location,
                    'solution': sol_type,
                    'E_cool': sol_data.get('E_cool', 0.0),
                    'E_heat': sol_data.get('E_heat', 0.0),
                    'ASE': sol_data.get('ASE', 0.0),
                    'R': sol_data.get('R', 0.0),
                    'cost': sol_data.get('cost', 0.0)
                })
        
        perf_df = pd.DataFrame(perf_data)
        perf_df.to_csv(os.path.join(tables_dir, 'performance_compare.csv'), index=False)
        logger.info('Saved performance comparison')
        
        # Cross validation
        logger.info('Performing cross validation')
        cv_results = cross_validate(train_epws, [ext_epw], mapping, args.results_dir)
        
        # Save cross validation errors
        cv_df = pd.DataFrame([cv_results])
        cv_df.to_csv(os.path.join(tables_dir, 'cv_errors.csv'), index=False)
        logger.info('Saved cross validation errors')
        logger.info(f'Cross validation results: {cv_results}')
        
        # Generate plots
        logger.info('Generating plots')
        
        # Plot CDD vs HDD
        if len(train_epws) > 0:
            # Create annual features for plotting
            annual_features = []
            for i, (epw_df, meta) in enumerate(zip(train_data, train_meta)):
                features = compute_climate_features(epw_df, meta)
                annual_features.append(features)
            annual_features_df = pd.DataFrame(annual_features)
            
            plot_cdd_hdd_scatter(annual_features_df, train_locations, figs_dir)
            logger.info('Generated CDD vs HDD scatter plot')
        
        # Plot feature bar for first location
        if len(all_features) > 0:
            plot_feature_bar(pd.DataFrame([all_features[0]]), train_locations[0], figs_dir)
            logger.info('Generated feature bar plot')
        
        # Plot rules
        plot_rules(rules, figs_dir)
        logger.info('Generated rules plot')
        
        # Plot performance comparison
        if all_comparisons:
            plot_perf_compare(all_comparisons[0], figs_dir)
            logger.info('Generated performance comparison plot')
        
        # Plot Riyadh external validation
        if ext_comparison:
            plot_riyadh_external(ext_comparison, figs_dir)
            logger.info('Generated Riyadh external validation plot')
        
        logger.info('Task 3 generalization completed successfully')
        
        return 0
        
    except Exception as e:
        logger.error(f'Error: {e}', exc_info=True)
        return 1

if __name__ == '__main__':
    sys.exit(main())
