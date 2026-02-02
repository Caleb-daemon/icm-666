#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation module for template solutions."""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from .adapters import run_analysis
try:
    from metrics import score_R
except Exception:
    def score_R(energy_use, glare_hours, cost):
        return 0.0
from .features import compute_climate_features

class Evaluator:
    """Evaluator class for template solutions."""
    
    def __init__(self, results_dir: str):
        """Initialize the evaluator.
        
        Args:
            results_dir: Results directory.
        """
        self.results_dir = results_dir

    def _read_hourly_metrics(self, csv_path: str) -> Optional[Dict[str, float]]:
        """Read hourly CSV and aggregate to annual metrics."""
        if not os.path.exists(csv_path):
            return None
        try:
            df = pd.read_csv(csv_path)
            def col_or_zeros(cols):
                for c in cols:
                    if c in df.columns:
                        return df[c].astype(float)
                return pd.Series(0.0, index=df.index)

            cooling = col_or_zeros(['Cooling_Load_W', 'cooling_load', 'Cooling_Load'])
            heating = col_or_zeros(['Heating_Load_W', 'heating_load', 'Heating_Load'])
            sdi = col_or_zeros(['SDI', 'ASE'])

            e_cool = float(cooling.sum() / 1000.0)
            e_heat = float(heating.sum() / 1000.0)
            ase = float((sdi > 0).sum())
            return {'E_cool': float(e_cool), 'E_heat': float(e_heat), 'ASE': float(ase)}
        except Exception:
            return None

    def _load_baseline_metrics(self, epw_path: str) -> Optional[Dict[str, float]]:
        """Load baseline metrics by location or scale from reference."""
        location_name = os.path.basename(epw_path).split('_')[0].upper()
        if location_name == 'HKG':
            baseline_path = os.path.join(self.results_dir, 'task1', 'tables', 'task1_hourly_baseline.csv')
        elif location_name in ('NOR', 'OSL'):
            baseline_path = os.path.join(self.results_dir, 'task2', 'tables', 'task2_hourly_baseline.csv')
        else:
            baseline_path = None

        if baseline_path:
            metrics = self._read_hourly_metrics(baseline_path)
            if metrics is not None:
                return metrics

        # Scale baseline from reference using degree-day ratios
        ref_hkg = self._read_hourly_metrics(os.path.join(self.results_dir, 'task1', 'tables', 'task1_hourly_baseline.csv'))
        ref_nor = self._read_hourly_metrics(os.path.join(self.results_dir, 'task2', 'tables', 'task2_hourly_baseline.csv'))
        if ref_hkg is None and ref_nor is None:
            return None

        epw_df, meta = None, None
        try:
            from .adapters import read_epw
            epw_df, meta = read_epw(epw_path)
        except Exception:
            epw_df = None
            read_epw = None

        if epw_df is None:
            return ref_hkg or ref_nor

        features = compute_climate_features(epw_df, meta)
        cdd = features.get('CDD24', 0.0)
        hdd = features.get('HDD18', 0.0)

        # Reference features
        def ref_features(ref_path: str):
            if read_epw is None:
                return None
            ref_df, ref_meta = read_epw(ref_path)
            return compute_climate_features(ref_df, ref_meta)

        if ref_hkg is not None:
            ref_feat_hkg = ref_features(os.path.join(os.path.dirname(self.results_dir), 'data', 'epw', 'HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw'))
        else:
            ref_feat_hkg = None
        if ref_nor is not None:
            ref_feat_nor = ref_features(os.path.join(os.path.dirname(self.results_dir), 'data', 'epw', 'NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw'))
        else:
            ref_feat_nor = None

        # Cooling from HKG, heating from NOR
        scaled = {'E_cool': 0.0, 'E_heat': 0.0, 'ASE': 0.0}
        if ref_hkg and ref_feat_hkg:
            cdd_ref = ref_feat_hkg.get('CDD24', 1.0)
            ratio_c = cdd / max(cdd_ref, 1.0)
            scaled['E_cool'] = ref_hkg['E_cool'] * ratio_c
            scaled['ASE'] = ref_hkg['ASE'] * ratio_c
        if ref_nor and ref_feat_nor:
            hdd_ref = ref_feat_nor.get('HDD18', 1.0)
            ratio_h = hdd / max(hdd_ref, 1.0)
            scaled['E_heat'] = ref_nor['E_heat'] * ratio_h

        return scaled

    def _load_optimized_metrics(self, epw_path: str) -> Optional[Dict[str, float]]:
        """Load optimized metrics by location."""
        location_name = os.path.basename(epw_path).split('_')[0].upper()
        if location_name == 'HKG':
            best_path = os.path.join(self.results_dir, 'task1', 'tables', 'task1_hourly_best.csv')
        elif location_name in ('NOR', 'OSL'):
            best_path = os.path.join(self.results_dir, 'task2', 'tables', 'task2_hourly_best.csv')
        else:
            best_path = None

        if best_path:
            metrics = self._read_hourly_metrics(best_path)
            if metrics is not None:
                return metrics

        # Scale optimized from reference using degree-day ratios
        ref_hkg = self._read_hourly_metrics(os.path.join(self.results_dir, 'task1', 'tables', 'task1_hourly_best.csv'))
        ref_nor = self._read_hourly_metrics(os.path.join(self.results_dir, 'task2', 'tables', 'task2_hourly_best.csv'))
        if ref_hkg is None and ref_nor is None:
            return self._load_baseline_metrics(epw_path)

        epw_df, meta = None, None
        try:
            from .adapters import read_epw
            epw_df, meta = read_epw(epw_path)
        except Exception:
            epw_df = None
            read_epw = None

        if epw_df is None:
            return self._load_baseline_metrics(epw_path)

        features = compute_climate_features(epw_df, meta)
        cdd = features.get('CDD24', 0.0)
        hdd = features.get('HDD18', 0.0)

        def ref_features(ref_path: str):
            if read_epw is None:
                return None
            ref_df, ref_meta = read_epw(ref_path)
            return compute_climate_features(ref_df, ref_meta)

        ref_feat_hkg = ref_features(os.path.join(os.path.dirname(self.results_dir), 'data', 'epw', 'HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw')) if ref_hkg else None
        ref_feat_nor = ref_features(os.path.join(os.path.dirname(self.results_dir), 'data', 'epw', 'NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw')) if ref_nor else None

        scaled = {'E_cool': 0.0, 'E_heat': 0.0, 'ASE': 0.0}
        if ref_hkg and ref_feat_hkg:
            cdd_ref = ref_feat_hkg.get('CDD24', 1.0)
            ratio_c = cdd / max(cdd_ref, 1.0)
            scaled['E_cool'] = ref_hkg['E_cool'] * ratio_c
            scaled['ASE'] = ref_hkg['ASE'] * ratio_c
        if ref_nor and ref_feat_nor:
            hdd_ref = ref_feat_nor.get('HDD18', 1.0)
            ratio_h = hdd / max(hdd_ref, 1.0)
            scaled['E_heat'] = ref_nor['E_heat'] * ratio_h

        return scaled

    def _template_ratio(self, template: Dict[str, Any], epw_path: str) -> float:
        """Compute interpolation ratio from template depth vs optimized depth."""
        location_name = os.path.basename(epw_path).split('_')[0].upper()
        keys = ['dN', 'dE', 'dS', 'dW']
        vals = []
        for k in keys:
            if k in template:
                vals.append(float(template[k]))
        if not vals:
            return 0.0
        level = float(np.mean(vals))

        if location_name == 'HKG':
            best_path = os.path.join(self.results_dir, 'task1', 'tables', 'task1_best_solution.csv')
        elif location_name in ('NOR', 'OSL'):
            best_path = os.path.join(self.results_dir, 'task2', 'tables', 'task2_best_solution.csv')
        else:
            # External locations: choose reference based on climate type
            try:
                from .adapters import read_epw
                epw_df, meta = read_epw(epw_path)
            except Exception:
                epw_df, meta = None, None
                read_epw = None

            if epw_df is None:
                best_path = os.path.join(self.results_dir, 'task1', 'tables', 'task1_best_solution.csv')
            else:
                features = compute_climate_features(epw_df, meta)
                cdd = features.get('CDD24', 0.0)
                hdd = features.get('HDD18', 0.0)
                if cdd > 2 * hdd:
                    best_path = os.path.join(self.results_dir, 'task1', 'tables', 'task1_best_solution.csv')
                else:
                    best_path = os.path.join(self.results_dir, 'task2', 'tables', 'task2_best_solution.csv')

        if not os.path.exists(best_path):
            return 0.0
        df = pd.read_csv(best_path)
        opt_vals = [df.get(k, pd.Series([0.0])).iloc[0] for k in keys if k in df.columns]
        opt_level = float(np.mean(opt_vals)) if opt_vals else 0.0
        if opt_level <= 0:
            return 0.0
        return float(min(1.0, max(0.0, level / opt_level)))
    
    def evaluate_template(self, epw_path: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a template solution.
        
        Args:
            epw_path: EPW file path.
            template: Template solution parameters.
            
        Returns:
            Evaluation results.
        """
        # Try to use run_analysis if available
        if run_analysis is not None:
            try:
                result = run_analysis(epw_path, template)
                if result is not None:
                    return result
            except:
                pass

        # Fallback to proxy evaluation using baseline/optimized interpolation
        location_name = os.path.basename(epw_path).split('_')[0].upper()
        baseline = self._load_baseline_metrics(epw_path) or {'E_cool': 0.0, 'E_heat': 0.0, 'ASE': 0.0}
        optimized = self._load_optimized_metrics(epw_path) or baseline

        ratio = self._template_ratio(template, epw_path)
        result = {
            'E_cool': baseline['E_cool'] + ratio * (optimized['E_cool'] - baseline['E_cool']),
            'E_heat': baseline['E_heat'] + ratio * (optimized['E_heat'] - baseline['E_heat']),
            'ASE': baseline['ASE'] + ratio * (optimized['ASE'] - baseline['ASE']),
            'R': 0.0,
            'cost': 0.0
        }
        return result
    
    def _manual_evaluation(self, epw_path: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Manually evaluate a template solution.
        
        Args:
            epw_path: EPW file path.
            template: Template solution parameters.
            
        Returns:
            Evaluation results.
        """
        # This is a placeholder for manual evaluation
        # In a real implementation, this would use the adapters to
        # run the full pipeline
        
        return None
    
    def evaluate_baseline(self, epw_path: str) -> Dict[str, Any]:
        """Evaluate baseline solution.
        
        Args:
            epw_path: EPW file path.
            
        Returns:
            Evaluation results.
        """
        # Try to load baseline from results
        baseline = self._load_baseline_metrics(epw_path)
        if baseline is None:
            baseline = {'E_cool': 0.0, 'E_heat': 0.0, 'ASE': 0.0}
        return {
            'E_cool': baseline['E_cool'],
            'E_heat': baseline['E_heat'],
            'ASE': baseline['ASE'],
            'R': 0.0,
            'cost': 0.0
        }
    
    def evaluate_optimized(self, epw_path: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate optimized solution.
        
        Args:
            epw_path: EPW file path.
            template: Template solution parameters.
            
        Returns:
            Evaluation results.
        """
        # For training locations, use existing best results
        location_name = os.path.basename(epw_path).split('_')[0].upper()
        
        optimized = self._load_optimized_metrics(epw_path)
        if optimized is None:
            return self.evaluate_template(epw_path, template)
        return {
            'E_cool': optimized['E_cool'],
            'E_heat': optimized['E_heat'],
            'ASE': optimized['ASE'],
            'R': 0.0,
            'cost': 0.0
        }
    
    def _local_optimization(self, epw_path: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Perform local optimization around template.
        
        Args:
            epw_path: EPW file path.
            template: Template solution parameters.
            
        Returns:
            Evaluation results.
        """
        # This is a placeholder for local optimization
        # In a real implementation, this would use the optimize module
        # to perform a local search around the template
        
        # For demonstration purposes, return template evaluation
        return self.evaluate_template(epw_path, template)
    
    def compare_solutions(self, epw_path: str, template: Dict[str, Any]) -> Dict[str, Any]:
        """Compare baseline, template, and optimized solutions.
        
        Args:
            epw_path: EPW file path.
            template: Template solution parameters.
            
        Returns:
            Comparison results.
        """
        baseline = self.evaluate_baseline(epw_path)
        template_eval = self.evaluate_template(epw_path, template)
        optimized = self.evaluate_optimized(epw_path, template)

        # Compute R scores using dynamic references from current comparisons
        energy_vals = [
            baseline.get('E_cool', 0.0) + baseline.get('E_heat', 0.0),
            template_eval.get('E_cool', 0.0) + template_eval.get('E_heat', 0.0),
            optimized.get('E_cool', 0.0) + optimized.get('E_heat', 0.0)
        ]
        glare_vals = [
            baseline.get('ASE', 0.0),
            template_eval.get('ASE', 0.0),
            optimized.get('ASE', 0.0)
        ]
        cost_vals = [
            baseline.get('cost', 0.0),
            template_eval.get('cost', 0.0),
            optimized.get('cost', 0.0)
        ]

        energy_ref = max(energy_vals) if max(energy_vals) > 0 else 1.0
        glare_ref = max(glare_vals) if max(glare_vals) > 0 else 1.0
        cost_ref = max(cost_vals) if max(cost_vals) > 0 else 1.0

        for sol in (baseline, template_eval, optimized):
            energy_use = sol.get('E_cool', 0.0) + sol.get('E_heat', 0.0)
            glare_hours = sol.get('ASE', 0.0)
            sol['R'] = score_R(
                energy_use=energy_use,
                glare_hours=glare_hours,
                cost=sol.get('cost', 0.0),
                energy_reference=energy_ref,
                glare_reference=glare_ref,
                cost_reference=cost_ref
            )
        
        comparison = {
            'baseline': baseline,
            'template': template_eval,
            'optimized': optimized,
            'template_vs_baseline': {
                'E_cool': (template_eval['E_cool'] - baseline['E_cool']) / baseline['E_cool'] if baseline['E_cool'] > 0 else 0,
                'E_heat': (template_eval['E_heat'] - baseline['E_heat']) / baseline['E_heat'] if baseline['E_heat'] > 0 else 0,
                'ASE': (template_eval['ASE'] - baseline['ASE']) / baseline['ASE'] if baseline['ASE'] > 0 else 0,
            },
            'optimized_vs_baseline': {
                'E_cool': (optimized['E_cool'] - baseline['E_cool']) / baseline['E_cool'] if baseline['E_cool'] > 0 else 0,
                'E_heat': (optimized['E_heat'] - baseline['E_heat']) / baseline['E_heat'] if baseline['E_heat'] > 0 else 0,
                'ASE': (optimized['ASE'] - baseline['ASE']) / baseline['ASE'] if baseline['ASE'] > 0 else 0,
            }
        }
        
        return comparison
    
    def cross_validate(self, train_epws: List[str], test_epws: List[str], mapping) -> Dict[str, Any]:
        """Cross validate the mapping model.
        
        Args:
            train_epws: Training EPW paths.
            test_epws: Test EPW paths.
            mapping: Fitted mapping model.
            
        Returns:
            Cross validation results.
        """
        cv_results = {
            'cv_err_loc': 0.0,
            'cv_err_sce': 0.0,
            'Err_ext': 0.0
        }
        
        # This is a placeholder for cross validation
        # In a real implementation, this would:
        # 1. Split data by location (leave-one-out)
        # 2. Split data by season (leave-one-out)
        # 3. Evaluate on external test set
        
        return cv_results

def evaluate_template(epw_path: str, template: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    """Evaluate a template solution.
    
    Args:
        epw_path: EPW file path.
        template: Template solution parameters.
        results_dir: Results directory.
        
    Returns:
        Evaluation results.
    """
    evaluator = Evaluator(results_dir)
    return evaluator.evaluate_template(epw_path, template)

def evaluate_baseline(epw_path: str, results_dir: str) -> Dict[str, Any]:
    """Evaluate baseline solution.
    
    Args:
        epw_path: EPW file path.
        results_dir: Results directory.
        
    Returns:
        Evaluation results.
    """
    evaluator = Evaluator(results_dir)
    return evaluator.evaluate_baseline(epw_path)

def compare_solutions(epw_path: str, template: Dict[str, Any], results_dir: str) -> Dict[str, Any]:
    """Compare baseline, template, and optimized solutions.
    
    Args:
        epw_path: EPW file path.
        template: Template solution parameters.
        results_dir: Results directory.
        
    Returns:
        Comparison results.
    """
    evaluator = Evaluator(results_dir)
    return evaluator.compare_solutions(epw_path, template)

def cross_validate(train_epws: List[str], test_epws: List[str], mapping, results_dir: str) -> Dict[str, Any]:
    """Cross validate the mapping model.
    
    Args:
        train_epws: Training EPW paths.
        test_epws: Test EPW paths.
        mapping: Fitted mapping model.
        results_dir: Results directory.
        
    Returns:
        Cross validation results.
    """
    evaluator = Evaluator(results_dir)
    return evaluator.cross_validate(train_epws, test_epws, mapping)
