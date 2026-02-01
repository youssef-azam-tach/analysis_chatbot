"""
Data Loader Module - Handles loading data from various sources
Currently supports: Excel
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import streamlit as st


class ExcelLoader:
    """Load and manage Excel files"""
    
    def __init__(self, file_path: str):
        self.file_path = Path(file_path)
        self.excel_file = None
        self.sheets = []
        self.df = None
        
    def get_sheets(self) -> List[str]:
        """Get all sheet names from Excel file"""
        try:
            # For .xls files, we need the xlrd engine
            if self.file_path.suffix.lower() == '.xls':
                self.excel_file = pd.ExcelFile(self.file_path, engine='xlrd')
            else:
                self.excel_file = pd.ExcelFile(self.file_path)
                
            self.sheets = self.excel_file.sheet_names
            return self.sheets
        except Exception as e:
            st.error(f"Error reading Excel file ({self.file_path.suffix}): {str(e)}")
            if "xlrd" in str(e).lower():
                st.info("💡 Tip: Try installing xlrd: `pip install xlrd`")
            return []
    
    def load_sheet(self, sheet_name: str) -> pd.DataFrame:
        """Load specific sheet from Excel file"""
        try:
            # Use appropriate engine for .xls
            if self.file_path.suffix.lower() == '.xls':
                self.df = pd.read_excel(self.file_path, sheet_name=sheet_name, engine='xlrd')
            else:
                self.df = pd.read_excel(self.file_path, sheet_name=sheet_name)
            return self.df
        except Exception as e:
            st.error(f"Error loading sheet '{sheet_name}': {str(e)}")
            return None
    
    def get_preview(self, n_rows: int = 5) -> pd.DataFrame:
        """Get preview of data"""
        if self.df is not None:
            return self.df.head(n_rows)
        return None
    
    def get_shape(self) -> Tuple[int, int]:
        """Get dataframe shape"""
        if self.df is not None:
            return self.df.shape
        return (0, 0)
    
    def get_columns(self) -> List[str]:
        """Get column names"""
        if self.df is not None:
            return self.df.columns.tolist()
        return []
    
    def get_dtypes(self) -> dict:
        """Get data types of columns"""
        if self.df is not None:
            return self.df.dtypes.to_dict()
        return {}
    
    def auto_detect_types(self) -> dict:
        """Auto-detect column types (numeric, categorical, datetime)"""
        if self.df is None:
            return {}
        
        type_mapping = {}
        for col in self.df.columns:
            if pd.api.types.is_numeric_dtype(self.df[col]):
                type_mapping[col] = "numeric"
            elif pd.api.types.is_datetime64_any_dtype(self.df[col]):
                type_mapping[col] = "datetime"
            elif pd.api.types.is_categorical_dtype(self.df[col]):
                type_mapping[col] = "categorical"
            else:
                # Check if it looks like categorical (few unique values)
                unique_ratio = self.df[col].nunique() / len(self.df)
                if unique_ratio < 0.05:
                    type_mapping[col] = "categorical"
                else:
                    type_mapping[col] = "text"
        
        return type_mapping
    
    def get_column_summary(self) -> dict:
        """Get summary statistics for each column"""
        if self.df is None:
            return {}
        
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


def load_excel(file_path: str) -> Optional[pd.DataFrame]:
    """Simple function to load Excel file"""
    try:
        df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading Excel file: {str(e)}")
        return None
