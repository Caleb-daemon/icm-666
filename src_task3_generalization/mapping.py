#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mapping module for climate features to template solutions."""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.tree import DecisionTreeRegressor, export_text

class Mapping:
    """Mapping class for climate features to template solutions."""
    
    def __init__(self, method: str = "rules"):
        """Initialize the mapping.
        
        Args:
            method: Mapping method, either "rules" or "decision_tree".
        """
        self.method = method
        self.models = {}
        self.rules = {}
        self.rule_map = {}
        self.rule_default = {}
    
    def fit(self, X: pd.DataFrame, y: pd.DataFrame) -> "Mapping":
        """Fit the mapping model.
        
        Args:
            X: Features dataframe.
            y: Targets dataframe.
            
        Returns:
            Self.
        """
        if self.method == "rules":
            self._fit_rules(X, y)
        elif self.method == "decision_tree":
            self._fit_decision_tree(X, y)
        return self
    
    def _fit_rules(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit rule-based mapping.
        
        Args:
            X: Features dataframe.
            y: Targets dataframe.
        """
        # Create rules based on climate classification
        for target_col in y.columns:
            rules, rule_map, default_value = self._generate_rules(X, y[target_col])
            self.rules[target_col] = rules
            self.rule_map[target_col] = rule_map
            self.rule_default[target_col] = default_value
    
    def _generate_rules(self, X: pd.DataFrame, y: pd.Series):
        """Generate rules for a target variable.
        
        Args:
            X: Features dataframe.
            y: Target series.
            
        Returns:
            List of rules.
        """
        rules = []

        def classify_row(row):
            classification = {}

            # Cooling/Heating/Mixed
            hdd = row.get('HDD18', 0)
            cdd = row.get('CDD24', 0)
            if hdd > 2 * cdd:
                classification['load_type'] = 'heating'
            elif cdd > 2 * hdd:
                classification['load_type'] = 'cooling'
            else:
                classification['load_type'] = 'mixed'

            # Solar potential
            s_summer = row.get('S_summer', 0)
            classification['solar'] = 'high' if s_summer > 1e9 else 'low'

            # Humidity
            humid = row.get('Humid_proxy', 0)
            classification['humidity'] = 'humid' if humid > 60 else 'dry'

            # Temperature swing
            t_range = row.get('T_range', 0)
            classification['swing'] = 'large' if t_range > 10 else 'small'

            return classification
        
        # Group by classification and generate rules
        grouped = X.apply(classify_row, axis=1).apply(pd.Series).join(y)

        rule_map = {}
        for (load_type, solar, humidity, swing), group in grouped.groupby(['load_type', 'solar', 'humidity', 'swing']):
            if len(group) > 0:
                avg_value = group.iloc[:, -1].mean()
                rule = f"IF load_type={load_type} AND solar={solar} AND humidity={humidity} AND swing={swing} THEN {y.name}={avg_value:.4f}"
                rules.append(rule)
                rule_map[(load_type, solar, humidity, swing)] = float(avg_value)

        # Add default rule
        default_value = float(y.mean())
        rules.append(f"ELSE {y.name}={default_value:.4f}")

        return rules, rule_map, default_value
    
    def _fit_decision_tree(self, X: pd.DataFrame, y: pd.DataFrame) -> None:
        """Fit decision tree mapping.
        
        Args:
            X: Features dataframe.
            y: Targets dataframe.
        """
        for target_col in y.columns:
            # Train decision tree with max depth 3
            model = DecisionTreeRegressor(max_depth=3, random_state=42)
            model.fit(X, y[target_col])
            self.models[target_col] = model
            
            # Export rules
            feature_names = list(X.columns)
            rule_text = export_text(model, feature_names=feature_names)
            self.rules[target_col] = rule_text.split('\n')
    
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict template solutions.
        
        Args:
            X: Features dataframe.
            
        Returns:
            Predictions dataframe.
        """
        predictions = {}
        
        if self.method == "rules":
            for target_col, rules in self.rules.items():
                predictions[target_col] = X.apply(lambda row: self._apply_rules(row, target_col), axis=1)
        elif self.method == "decision_tree":
            # Get feature names used during training
            if hasattr(self.models[next(iter(self.models))], 'feature_names_in_'):
                feature_names = self.models[next(iter(self.models))].feature_names_in_
                # Ensure only training features are present
                X = X[feature_names]
            
            for target_col, model in self.models.items():
                predictions[target_col] = model.predict(X)
        
        return pd.DataFrame(predictions)
    
    def _apply_rules(self, row: pd.Series, target_col: str) -> float:
        """Apply rules to a row.
        
        Args:
            row: Features row.
            rules: List of rules.
            
        Returns:
            Predicted value.
        """
        if not self.rule_map:
            return 0.0

        # Recreate the classification key
        hdd = row.get('HDD18', 0)
        cdd = row.get('CDD24', 0)
        if hdd > 2 * cdd:
            load_type = 'heating'
        elif cdd > 2 * hdd:
            load_type = 'cooling'
        else:
            load_type = 'mixed'

        s_summer = row.get('S_summer', 0)
        solar = 'high' if s_summer > 1e9 else 'low'

        humid = row.get('Humid_proxy', 0)
        humidity = 'humid' if humid > 60 else 'dry'

        t_range = row.get('T_range', 0)
        swing = 'large' if t_range > 10 else 'small'

        key = (load_type, solar, humidity, swing)

        # Find default for the current target
        rule_map = self.rule_map.get(target_col, {})
        return rule_map.get(key, self.rule_default.get(target_col, 0.0))
    
    def get_rules(self) -> Dict[str, List[str]]:
        """Get the rules for each target variable.
        
        Returns:
            Dictionary of rules.
        """
        return self.rules
    
    def recommend_template(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend template solution for a location.
        
        Args:
            features: Climate features.
            
        Returns:
            Template solution and rule trace.
        """
        # Convert features to dataframe
        X = pd.DataFrame([features])
        
        # Predict template
        template = self.predict(X).iloc[0].to_dict()
        
        # Generate rule trace
        rule_trace = []
        for target_col, rules in self.rules.items():
            rule_trace.append(f"For {target_col}:")
            if self.method == "rules":
                for rule in rules:
                    rule_trace.append(f"  {rule}")
            elif self.method == "decision_tree":
                for rule_line in rules:
                    rule_trace.append(f"  {rule_line}")
        
        return {
            'template': template,
            'rule_trace': rule_trace
        }

def fit_mapping(X: pd.DataFrame, y: pd.DataFrame, method: str = "rules") -> Mapping:
    """Fit a mapping model.
    
    Args:
        X: Features dataframe.
        y: Targets dataframe.
        method: Mapping method, either "rules" or "decision_tree".
        
    Returns:
        Fitted mapping model.
    """
    mapping = Mapping(method=method)
    return mapping.fit(X, y)

def recommend_template(mapping: Mapping, features: Dict[str, Any]) -> Dict[str, Any]:
    """Recommend template solution for a location.
    
    Args:
        mapping: Fitted mapping model.
        features: Climate features.
        
    Returns:
        Template solution and rule trace.
    """
    return mapping.recommend_template(features)
