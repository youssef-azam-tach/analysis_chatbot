"""
Exploratory Data Analysis Module
Automatic detection and analysis of data patterns
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple


class EDAAnalyzer:
    """Perform automatic EDA on dataframe"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    def get_missing_values(self) -> Dict:
        """Analyze missing values per column"""
        missing = {
            "column": [],
            "missing_count": [],
            "missing_percentage": [],
        }
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            missing["column"].append(col)
            missing["missing_count"].append(missing_count)
            missing["missing_percentage"].append(missing_pct)
        
        return pd.DataFrame(missing).sort_values("missing_count", ascending=False)
    
    def get_unique_values(self) -> Dict:
        """Get unique value counts per column"""
        unique_stats = {}
        for col in self.df.columns:
            unique_stats[col] = {
                "unique_count": self.df[col].nunique(),
                "unique_percentage": (self.df[col].nunique() / len(self.df)) * 100,
            }
        return unique_stats
    
    def detect_outliers(self, method: str = "iqr") -> Dict:
        """Detect outliers in numeric columns"""
        outliers = {}
        
        for col in self.numeric_cols:
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (self.df[col] < lower_bound) | (self.df[col] > upper_bound)
                outliers[col] = {
                    "count": outlier_mask.sum(),
                    "percentage": (outlier_mask.sum() / len(self.df)) * 100,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }
        
        return outliers
    
    def get_numerical_stats(self) -> pd.DataFrame:
        """Get statistical summary for numeric columns"""
        return self.df[self.numeric_cols].describe().T
    
    def get_categorical_stats(self) -> Dict:
        """Get statistics for categorical columns"""
        stats = {}
        for col in self.categorical_cols:
            stats[col] = {
                "unique_count": self.df[col].nunique(),
                "top_value": self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else None,
                "top_value_count": self.df[col].value_counts().iloc[0] if len(self.df[col].value_counts()) > 0 else 0,
            }
        return stats
    
    def get_correlation_matrix(self) -> pd.DataFrame:
        """Get correlation matrix for numeric columns"""
        if len(self.numeric_cols) > 0:
            return self.df[self.numeric_cols].corr()
        return pd.DataFrame()
    
    def get_high_correlations(self, threshold: float = 0.7) -> List[Tuple]:
        """Get pairs of highly correlated columns"""
        corr_matrix = self.get_correlation_matrix()
        
        if corr_matrix.empty:
            return []
        
        high_corr = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                if abs(corr_matrix.iloc[i, j]) >= threshold:
                    high_corr.append({
                        "col1": corr_matrix.columns[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": corr_matrix.iloc[i, j],
                    })
        
        return sorted(high_corr, key=lambda x: abs(x["correlation"]), reverse=True)
    
    def get_column_summary(self) -> Dict:
        """Get summary statistics for each column"""
        summary = {}
        for col in self.df.columns:
            summary[col] = {
                "dtype": str(self.df[col].dtype),
                "non_null": self.df[col].notna().sum(),
                "null_count": self.df[col].isna().sum(),
                "unique": self.df[col].nunique(),
            }
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                summary[col].update({
                    "mean": float(self.df[col].mean()),
                    "median": float(self.df[col].median()),
                    "std": float(self.df[col].std()),
                    "min": float(self.df[col].min()),
                    "max": float(self.df[col].max()),
                })
        
        return summary
    
    def get_summary_report(self) -> Dict:
        """Generate comprehensive summary report"""
        return {
            "shape": self.df.shape,
            "columns": len(self.df.columns),
            "rows": len(self.df),
            "numeric_columns": len(self.numeric_cols),
            "categorical_columns": len(self.categorical_cols),
            "datetime_columns": len(self.datetime_cols),
            "memory_usage_mb": self.df.memory_usage(deep=True).sum() / 1024**2,
            "duplicates": self.df.duplicated().sum(),
            "duplicate_percentage": (self.df.duplicated().sum() / len(self.df)) * 100,
        }
