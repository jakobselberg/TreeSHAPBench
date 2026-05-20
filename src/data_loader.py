"""
Data loading utilities for TreeSHAP benchmarking.
Handles census+income and superconductivity datasets.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


class DataLoader:
    """Load and preprocess datasets for benchmarking."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
    
    def load_census_income(self, test_size: float = 0.2, random_state: int = 42):
        """
        Load census income dataset (Adult dataset).
        
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Load the data
        names_file = self.data_dir / "census+income" / "adult.names"
        data_file = self.data_dir / "census+income" / "adult.data"
        
        # Parse feature names from .names file
        feature_names = []
        with open(names_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('|') and ':' in line:
                    name = line.split(':')[0].strip()
                    feature_names.append(name)
        
        # Load data
        df = pd.read_csv(data_file, names=feature_names, skipinitialspace=True)
        
        # Separate features and target (last column is target)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        # Encode categorical variables
        X_encoded = X.copy()
        categorical_cols = X.select_dtypes(include=['object']).columns
        
        label_encoders = {}
        for col in categorical_cols:
            le = LabelEncoder()
            X_encoded[col] = le.fit_transform(X_encoded[col].astype(str))
            label_encoders[col] = le
        
        # Encode target
        y_encoded = LabelEncoder().fit_transform(y.astype(str))
        
        # Convert to numeric
        X_encoded = X_encoded.astype(np.float32)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y_encoded, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, X.columns.tolist()
    
    def load_superconductivity(self, test_size: float = 0.2, random_state: int = 42):
        """
        Load superconductivity dataset.
        
        Returns:
            X_train, X_test, y_train, y_test, feature_names
        """
        # Load the data
        data_file = self.data_dir / "superconductivity" / "train.csv"
        
        df = pd.read_csv(data_file)
        
        # Last column is the target (critical temperature)
        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
        
        feature_names = X.columns.tolist()
        
        # Convert to numpy arrays
        X = X.astype(np.float32).values
        y = y.astype(np.float32).values
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        return X_train, X_test, y_train, y_test, feature_names


def load_dataset(name: str, data_dir: str = "data", **kwargs):
    """
    Convenience function to load a dataset by name.
    
    Args:
        name: 'census_income' or 'superconductivity'
        data_dir: Path to data directory
        **kwargs: Additional arguments passed to loader
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
    """
    loader = DataLoader(data_dir)
    
    if name.lower() == 'census_income':
        return loader.load_census_income(**kwargs)
    elif name.lower() == 'superconductivity':
        return loader.load_superconductivity(**kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}")
