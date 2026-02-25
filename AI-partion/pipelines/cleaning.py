"""
Data Cleaning Pipeline
Handles missing values, outliers, encoding, and scaling
Also includes Power Query-like features: Merge, Append, Add Custom Columns

CRITICAL RULES:
- Duplicates are removed per ENTIRE ROW, not per column
- Key columns (IDs, Keys) are NOT treated as numeric measures
- Key columns are NOT checked for outliers
- Always detect column role before applying transformations
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from typing import Dict, Tuple, Optional, List, Any, Callable
import re


class IntelligentColumnDetector:
    """
    Detects column roles to apply appropriate cleaning logic.
    Key columns should not be treated as measures.
    """
    
    KEY_PATTERNS = [
        r'.*key$', r'.*_key$', r'.*id$', r'.*_id$', r'^id$',
        r'.*code$', r'.*_code$', r'.*number$', r'.*_no$', r'.*_num$',
        r'.*index$', r'.*_idx$', r'.*pk$', r'.*fk$',
        r'sku', r'upc', r'barcode', r'serial',
    ]
    
    MEASURE_PATTERNS = [
        r'.*amount.*', r'.*price.*', r'.*cost.*',
        r'.*revenue.*', r'.*sales.*', r'.*profit.*',
        r'.*quantity.*', r'.*qty.*',
        r'.*total.*', r'.*sum.*', r'.*avg.*',
        r'.*rate.*', r'.*percent.*', r'.*pct.*',
        r'.*weight.*', r'.*height.*', r'.*width.*',
        r'.*score.*', r'.*rating.*',
    ]
    
    @staticmethod
    def is_key_column(df: pd.DataFrame, column: str) -> bool:
        """Check if a column is likely a key/identifier (should not analyze as measure)."""
        col_lower = column.lower().strip()
        
        # Check name patterns
        for pattern in IntelligentColumnDetector.KEY_PATTERNS:
            if re.match(pattern, col_lower, re.IGNORECASE):
                return True
        
        # Check data characteristics for numeric columns
        if np.issubdtype(df[column].dtype, np.number):
            uniqueness = df[column].nunique() / len(df) if len(df) > 0 else 0
            # High uniqueness + sequential pattern = likely key
            if uniqueness > 0.9:
                return True
        
        return False
    
    @staticmethod
    def is_measure_column(df: pd.DataFrame, column: str) -> bool:
        """Check if a column is a measure (appropriate for numeric analysis)."""
        if not np.issubdtype(df[column].dtype, np.number):
            return False
        
        col_lower = column.lower().strip()
        
        # Must NOT be a key column
        if IntelligentColumnDetector.is_key_column(df, column):
            return False
        
        # Check measure patterns
        for pattern in IntelligentColumnDetector.MEASURE_PATTERNS:
            if re.match(pattern, col_lower, re.IGNORECASE):
                return True
        
        # If numeric and not a key, treat as measure by default
        return True
    
    @staticmethod
    def get_key_columns(df: pd.DataFrame) -> List[str]:
        """Get list of all key/identifier columns."""
        return [col for col in df.columns if IntelligentColumnDetector.is_key_column(df, col)]
    
    @staticmethod
    def get_measure_columns(df: pd.DataFrame) -> List[str]:
        """Get list of all measure columns (appropriate for numeric analysis)."""
        return [col for col in df.columns if IntelligentColumnDetector.is_measure_column(df, col)]


class DataCleaner:
    """
    Clean and preprocess data with INTELLIGENT column understanding.
    
    CRITICAL RULES:
    - Remove duplicates by ENTIRE ROW, not individual columns
    - Key columns are excluded from outlier detection
    - Data-driven decisions, not blind rules
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.original_df = df.copy()
        self.cleaning_log = []
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Intelligent column detection
        self.key_columns = IntelligentColumnDetector.get_key_columns(df)
        self.measure_columns = IntelligentColumnDetector.get_measure_columns(df)
    
    def handle_missing_values(self, strategy: str = "mean", columns: Optional[List] = None) -> pd.DataFrame:
        """
        Handle missing values
        Strategies: mean, median, mode, drop, forward_fill, backward_fill
        """
        if columns is None:
            columns = self.df.columns
        
        for col in columns:
            if self.df[col].isna().sum() == 0:
                continue
            
            if col in self.numeric_cols:
                if strategy == "mean":
                    self.df[col].fillna(self.df[col].mean(), inplace=True)
                elif strategy == "median":
                    self.df[col].fillna(self.df[col].median(), inplace=True)
                elif strategy == "drop":
                    self.df.dropna(subset=[col], inplace=True)
                elif strategy == "forward_fill":
                    self.df[col].fillna(method='ffill', inplace=True)
                elif strategy == "backward_fill":
                    self.df[col].fillna(method='bfill', inplace=True)
            else:
                if strategy == "mode":
                    mode_val = self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else "Unknown"
                    self.df[col].fillna(mode_val, inplace=True)
                elif strategy == "drop":
                    self.df.dropna(subset=[col], inplace=True)
        
        self.cleaning_log.append(f"Handled missing values using {strategy} strategy")
        return self.df
    
    def remove_outliers(self, method: str = "iqr", columns: Optional[List] = None) -> pd.DataFrame:
        """
        Remove outliers - ONLY from MEASURE columns (not key columns).
        
        CRITICAL: Key columns (IDs, Keys) are automatically excluded even if passed.
        Methods: iqr, zscore
        """
        if columns is None:
            # Only use measure columns, never key columns
            columns = self.measure_columns
        else:
            # Filter out key columns from the provided list
            columns = [col for col in columns if col not in self.key_columns]
        
        initial_rows = len(self.df)
        
        for col in columns:
            # Double-check: skip if not a valid measure column
            if col not in self.numeric_cols:
                continue
            if col in self.key_columns:
                self.cleaning_log.append(f"⚠️ Skipped outlier removal for '{col}' (key/identifier column)")
                continue
            
            if method == "iqr":
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                if IQR > 0:  # Only apply if there's variance
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    self.df = self.df[(self.df[col] >= lower_bound) & (self.df[col] <= upper_bound)]
            
            elif method == "zscore":
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < 3]
        
        removed_rows = initial_rows - len(self.df)
        self.cleaning_log.append(f"Removed {removed_rows} outlier rows using {method} method (key columns excluded)")
        return self.df
    
    def encode_categorical(self, method: str = "label", columns: Optional[List] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Encode categorical variables
        Methods: label, onehot
        """
        if columns is None:
            columns = self.categorical_cols
        
        encoders = {}
        
        for col in columns:
            if col not in self.categorical_cols:
                continue
            
            if method == "label":
                le = LabelEncoder()
                self.df[col] = le.fit_transform(self.df[col].astype(str))
                encoders[col] = le
            
            elif method == "onehot":
                dummies = pd.get_dummies(self.df[col], prefix=col, drop_first=True)
                self.df = pd.concat([self.df, dummies], axis=1)
                self.df.drop(col, axis=1, inplace=True)
                encoders[col] = "onehot"
        
        self.cleaning_log.append(f"Encoded categorical variables using {method} method")
        return self.df, encoders
    
    def scale_numeric(self, method: str = "standard", columns: Optional[List] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Scale numeric variables - ONLY MEASURE columns (not key columns).
        Methods: standard, minmax
        """
        if columns is None:
            # Only scale measure columns, never key columns
            columns = self.measure_columns
        else:
            # Filter out key columns
            columns = [col for col in columns if col not in self.key_columns]
        
        scalers = {}
        
        for col in columns:
            if col not in self.numeric_cols:
                continue
            if col in self.key_columns:
                continue
            
            if method == "standard":
                scaler = StandardScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                scalers[col] = scaler
            
            elif method == "minmax":
                scaler = MinMaxScaler()
                self.df[col] = scaler.fit_transform(self.df[[col]])
                scalers[col] = scaler
        
        self.cleaning_log.append(f"Scaled numeric variables using {method} method (key columns excluded)")
        return self.df, scalers
    
    def remove_duplicates(self, subset: Optional[List] = None) -> pd.DataFrame:
        """
        Remove duplicate rows - ENTIRE ROW duplicates only.
        
        IMPORTANT: This removes rows where ALL columns (or specified subset) are identical.
        Duplicate values in individual columns (like Product Name, Brand) are NORMAL
        and should NOT be treated as issues.
        
        Args:
            subset: Optional list of columns to check for duplicates.
                   If None, checks ALL columns (full row duplicates).
        """
        initial_rows = len(self.df)
        self.df.drop_duplicates(subset=subset, inplace=True)
        removed_rows = initial_rows - len(self.df)
        
        if subset:
            self.cleaning_log.append(f"Removed {removed_rows} duplicate rows based on columns: {subset}")
        else:
            self.cleaning_log.append(f"Removed {removed_rows} fully duplicate rows (all columns identical)")
        
        return self.df
    
    def get_outliers_in_column(self, column: str, method: str = "iqr") -> dict:
        """
        Get information about outliers in a column without removing them.
        
        Returns a dict with:
        - 'outlier_indices': List of row indices that are outliers
        - 'outlier_values': List of actual values that are outliers
        - 'total_count': Total number of outliers
        - 'bounds': {'lower': lower_bound, 'upper': upper_bound} for IQR method
        """
        if column not in self.numeric_cols or column in self.key_columns:
            return {'outlier_indices': [], 'outlier_values': [], 'total_count': 0}
        
        outlier_data = {'outlier_indices': [], 'outlier_values': [], 'total_count': 0}
        
        if method == "iqr":
            Q1 = self.df[column].quantile(0.25)
            Q3 = self.df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_mask = (self.df[column] < lower_bound) | (self.df[column] > upper_bound)
                
                outlier_data['bounds'] = {'lower': lower_bound, 'upper': upper_bound}
                outlier_data['outlier_indices'] = self.df[outlier_mask].index.tolist()
                outlier_data['outlier_values'] = self.df[outlier_mask][column].tolist()
                outlier_data['total_count'] = len(outlier_data['outlier_indices'])
        
        elif method == "zscore":
            z_scores = np.abs((self.df[column] - self.df[column].mean()) / self.df[column].std())
            outlier_mask = z_scores >= 3
            
            outlier_data['outlier_indices'] = self.df[outlier_mask].index.tolist()
            outlier_data['outlier_values'] = self.df[outlier_mask][column].tolist()
            outlier_data['total_count'] = len(outlier_data['outlier_indices'])
        
        return outlier_data
    
    def remove_outliers_selective(self, column: str, row_indices: List[int], method: str = "iqr") -> pd.DataFrame:
        """
        Remove outliers from ONLY specified row indices (selective removal).
        
        This allows the user to fix only SOME of the detected outliers, not all.
        
        Args:
            column: The column to remove outliers from
            row_indices: List of row indices to remove
            method: Method used (for logging purposes)
        
        Returns:
            Updated dataframe with only specified rows removed
        """
        if column in self.key_columns:
            self.cleaning_log.append(f"⚠️ Skipped selective outlier removal for '{column}' (key column)")
            return self.df
        
        if not row_indices:
            return self.df
        
        initial_rows = len(self.df)
        # Remove only the specified rows
        self.df = self.df.drop(row_indices)
        self.df = self.df.reset_index(drop=True)
        
        removed_rows = initial_rows - len(self.df)
        self.cleaning_log.append(f"Selectively removed {removed_rows} outliers from '{column}' ({removed_rows}/{len(row_indices)} selected)")
        return self.df
    
    def get_missing_values_info(self, column: str) -> dict:
        """
        Get information about missing values in a column.
        
        Returns a dict with:
        - 'missing_indices': List of row indices with missing values
        - 'total_count': Total number of missing values
        - 'percentage': Percentage of missing values
        """
        missing_mask = self.df[column].isna()
        missing_data = {
            'missing_indices': self.df[missing_mask].index.tolist(),
            'total_count': missing_mask.sum(),
            'percentage': (missing_mask.sum() / len(self.df)) * 100 if len(self.df) > 0 else 0
        }
        return missing_data
    
    def handle_missing_values_selective(
        self, 
        column: str, 
        row_indices: List[int],
        strategy: str = "mean"
    ) -> pd.DataFrame:
        """
        Handle missing values in ONLY specified rows (selective handling).
        
        This allows the user to fix only SOME of the detected missing values, not all.
        
        Args:
            column: The column with missing values
            row_indices: List of row indices to fix
            strategy: How to fill the values ('mean', 'median', 'mode', 'drop', 'forward_fill', 'backward_fill')
        
        Returns:
            Updated dataframe with specified missing values handled
        """
        if not row_indices:
            return self.df
        
        initial_missing = self.df[column].isna().sum()
        
        if strategy == "mean":
            if column in self.numeric_cols:
                fill_value = self.df[column].mean()
                self.df.loc[row_indices, column] = fill_value
                self.cleaning_log.append(f"Selectively filled {len(row_indices)} missing values in '{column}' with mean: {fill_value:.2f}")
            else:
                self.cleaning_log.append(f"⚠️ Cannot use mean for non-numeric column '{column}'")
        
        elif strategy == "median":
            if column in self.numeric_cols:
                fill_value = self.df[column].median()
                self.df.loc[row_indices, column] = fill_value
                self.cleaning_log.append(f"Selectively filled {len(row_indices)} missing values in '{column}' with median: {fill_value:.2f}")
            else:
                self.cleaning_log.append(f"⚠️ Cannot use median for non-numeric column '{column}'")
        
        elif strategy == "mode":
            fill_value = self.df[column].mode()[0] if len(self.df[column].mode()) > 0 else None
            if fill_value is not None:
                self.df.loc[row_indices, column] = fill_value
                self.cleaning_log.append(f"Selectively filled {len(row_indices)} missing values in '{column}' with mode: {fill_value}")
            else:
                self.cleaning_log.append(f"⚠️ Could not determine mode for column '{column}'")
        
        elif strategy == "drop":
            self.df = self.df.drop(row_indices)
            self.df = self.df.reset_index(drop=True)
            self.cleaning_log.append(f"Selectively dropped {len(row_indices)} rows with missing values in '{column}'")
        
        elif strategy == "forward_fill":
            self.df.loc[row_indices, column] = self.df.loc[row_indices, column].fillna(method='ffill')
            self.cleaning_log.append(f"Selectively filled {len(row_indices)} missing values in '{column}' using forward fill")
        
        elif strategy == "backward_fill":
            self.df.loc[row_indices, column] = self.df.loc[row_indices, column].fillna(method='bfill')
            self.cleaning_log.append(f"Selectively filled {len(row_indices)} missing values in '{column}' using backward fill")
        
        return self.df
    
    def get_cleaning_log(self) -> List[str]:
        """Get log of all cleaning operations"""
        return self.cleaning_log
    
    def get_cleaned_df(self) -> pd.DataFrame:
        """Get cleaned dataframe"""
        return self.df
    
    def reset(self) -> pd.DataFrame:
        """Reset to original dataframe"""
        self.df = self.original_df.copy()
        self.cleaning_log = []
        return self.df


# ====================================================================================
# POWER QUERY EDITOR FEATURES
# ====================================================================================

class PowerQueryOperations:
    """
    Power Query-like operations for data transformation.
    Similar to Excel Power Query Editor functionality.
    """
    
    @staticmethod
    def merge_queries(
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        left_on: str,
        right_on: str,
        how: str = 'left',
        suffixes: Tuple[str, str] = ('', '_right')
    ) -> pd.DataFrame:
        """
        Merge two DataFrames - similar to VLOOKUP/XLOOKUP in Excel.
        
        This is equivalent to Power Query's Merge Queries operation.
        
        Args:
            left_df: The main/primary DataFrame
            right_df: The lookup DataFrame
            left_on: Column name in left_df to match on
            right_on: Column name in right_df to match on
            how: Type of merge - 'left', 'right', 'inner', 'outer'
                 - 'left': Keep all rows from left (like VLOOKUP)
                 - 'inner': Only matching rows
                 - 'outer': All rows from both
            suffixes: Suffixes for overlapping column names
            
        Returns:
            Merged DataFrame
        """
        result = pd.merge(
            left_df,
            right_df,
            left_on=left_on,
            right_on=right_on,
            how=how,
            suffixes=suffixes
        )
        return result
    
    @staticmethod
    def append_queries(
        dfs: List[pd.DataFrame],
        ignore_index: bool = True
    ) -> pd.DataFrame:
        """
        Append (stack) multiple DataFrames vertically.
        
        This is equivalent to Power Query's Append Queries operation.
        Similar to UNION in SQL.
        
        Args:
            dfs: List of DataFrames to append
            ignore_index: Whether to reset the index
            
        Returns:
            Combined DataFrame
        """
        result = pd.concat(dfs, ignore_index=ignore_index)
        return result
    
    @staticmethod
    def add_custom_column(
        df: pd.DataFrame,
        new_column_name: str,
        expression: str,
        column_mappings: Optional[Dict[str, str]] = None
    ) -> pd.DataFrame:
        """
        Add a custom calculated column using an expression.
        
        This is equivalent to Power Query's Add Column > Custom Column.
        
        Args:
            df: The DataFrame to modify
            new_column_name: Name for the new column
            expression: Python expression (can reference columns as variables)
                       Example: "Price * Quantity" or "df['Price'] * df['Quantity']"
            column_mappings: Optional dict mapping expression variables to actual column names
            
        Returns:
            DataFrame with new column added
        """
        result = df.copy()
        
        try:
            def _to_series(v: Any) -> pd.Series:
                if isinstance(v, pd.Series):
                    return v
                if isinstance(v, (list, tuple, np.ndarray)):
                    return pd.Series(v, index=result.index)
                return pd.Series([v] * len(result), index=result.index)

            def IF(condition: Any, true_val: Any, false_val: Any) -> pd.Series:
                cond_series = _to_series(condition).astype(bool)
                t_series = _to_series(true_val)
                f_series = _to_series(false_val)
                return pd.Series(np.where(cond_series, t_series, f_series), index=result.index)

            def COALESCE(value: Any, fallback: Any) -> pd.Series:
                val_series = _to_series(value)
                fb_series = _to_series(fallback)
                return val_series.where(val_series.notna(), fb_series)

            def YEAR(value: Any) -> pd.Series:
                return pd.to_datetime(_to_series(value), errors='coerce').dt.year

            def MONTH(value: Any) -> pd.Series:
                return pd.to_datetime(_to_series(value), errors='coerce').dt.month

            def DAY(value: Any) -> pd.Series:
                return pd.to_datetime(_to_series(value), errors='coerce').dt.day

            def WEEKDAY(value: Any) -> pd.Series:
                return pd.to_datetime(_to_series(value), errors='coerce').dt.weekday

            def SUM(value: Any) -> float:
                return float(pd.to_numeric(_to_series(value), errors='coerce').sum())

            def AVG(value: Any) -> float:
                return float(pd.to_numeric(_to_series(value), errors='coerce').mean())

            def MEDIAN(value: Any) -> float:
                return float(pd.to_numeric(_to_series(value), errors='coerce').median())

            def MIN(value: Any) -> Any:
                series = _to_series(value)
                if pd.api.types.is_numeric_dtype(series):
                    return float(pd.to_numeric(series, errors='coerce').min())
                return series.min()

            def MAX(value: Any) -> Any:
                series = _to_series(value)
                if pd.api.types.is_numeric_dtype(series):
                    return float(pd.to_numeric(series, errors='coerce').max())
                return series.max()

            def COUNT(value: Any) -> int:
                return int(_to_series(value).notna().sum())

            def NUNIQUE(value: Any) -> int:
                return int(_to_series(value).nunique(dropna=True))

            # Create a safe evaluation context with column access
            # Include built-in functions needed for type conversions
            context = {
                'df': result, 
                'np': np, 
                'pd': pd,
                'int': int,
                'float': float,
                'str': str,
                'bool': bool,
                'len': len,
                'abs': abs,
                'round': round,
                'min': min,
                'max': max,
                'sum': sum,
                'IF': IF,
                'if_': IF,
                'COALESCE': COALESCE,
                'coalesce': COALESCE,
                'YEAR': YEAR,
                'year': YEAR,
                'MONTH': MONTH,
                'month': MONTH,
                'DAY': DAY,
                'day': DAY,
                'WEEKDAY': WEEKDAY,
                'weekday': WEEKDAY,
                'SUM': SUM,
                'sum_': SUM,
                'AVG': AVG,
                'avg': AVG,
                'MEDIAN': MEDIAN,
                'median': MEDIAN,
                'MIN': MIN,
                'min_': MIN,
                'MAX': MAX,
                'max_': MAX,
                'COUNT': COUNT,
                'count_': COUNT,
                'NUNIQUE': NUNIQUE,
                'nunique': NUNIQUE,
            }
            
            # Add column values as variables
            for col in result.columns:
                # Clean column name to be a valid Python variable
                var_name = re.sub(r'[^a-zA-Z0-9_]', '_', col)
                context[var_name] = result[col]
            
            # Apply column mappings if provided
            if column_mappings:
                for var, col in column_mappings.items():
                    context[var] = result[col]
            
            # Evaluate the expression with proper builtins
            # Create globals dict with proper __builtins__ access
            eval_globals = {**context, '__builtins__': __builtins__}
            result[new_column_name] = eval(expression, eval_globals)
            
        except Exception as e:
            raise ValueError(f"Error evaluating expression '{expression}': {str(e)}")
        
        return result
    
    @staticmethod
    def add_column_from_lookup(
        main_df: pd.DataFrame,
        lookup_df: pd.DataFrame,
        main_key: str,
        lookup_key: str,
        value_column: str,
        new_column_name: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Add a column from another table based on key matching.
        
        This is exactly like VLOOKUP/XLOOKUP in Excel.
        
        Args:
            main_df: The main DataFrame to add column to
            lookup_df: The lookup table
            main_key: Column in main_df to match
            lookup_key: Column in lookup_df to match
            value_column: Column from lookup_df to bring over
            new_column_name: Optional name for the new column (defaults to value_column)
            
        Returns:
            DataFrame with new column added
        """
        if new_column_name is None:
            new_column_name = value_column
        
        # Create a lookup mapping
        lookup_map = lookup_df.set_index(lookup_key)[value_column].to_dict()
        
        # Apply the lookup
        result = main_df.copy()
        result[new_column_name] = result[main_key].map(lookup_map)
        
        return result
    
    @staticmethod
    def group_and_aggregate(
        df: pd.DataFrame,
        group_by: List[str],
        aggregations: Dict[str, str]
    ) -> pd.DataFrame:
        """
        Group by columns and aggregate.
        
        Similar to Power Query's Group By operation.
        
        Args:
            df: The DataFrame
            group_by: List of columns to group by
            aggregations: Dict of {column: aggregation_function}
                         Functions: 'sum', 'mean', 'count', 'min', 'max', 'first', 'last'
        
        Returns:
            Grouped and aggregated DataFrame
        """
        result = df.groupby(group_by, as_index=False).agg(aggregations)
        return result
    
    @staticmethod
    def pivot_table(
        df: pd.DataFrame,
        index: List[str],
        columns: str,
        values: str,
        aggfunc: str = 'sum'
    ) -> pd.DataFrame:
        """
        Create a pivot table.
        
        Similar to Power Query's Pivot Column operation.
        
        Args:
            df: The DataFrame
            index: Columns to use as row labels
            columns: Column to pivot (values become columns)
            values: Column to aggregate
            aggfunc: Aggregation function ('sum', 'mean', 'count', etc.)
        
        Returns:
            Pivoted DataFrame
        """
        result = pd.pivot_table(
            df,
            index=index,
            columns=columns,
            values=values,
            aggfunc=aggfunc
        ).reset_index()
        
        # Flatten column names if multi-level
        if isinstance(result.columns, pd.MultiIndex):
            result.columns = [' '.join(map(str, col)).strip() for col in result.columns.values]
        
        return result
    
    @staticmethod
    def unpivot(
        df: pd.DataFrame,
        id_vars: List[str],
        value_vars: Optional[List[str]] = None,
        var_name: str = 'Attribute',
        value_name: str = 'Value'
    ) -> pd.DataFrame:
        """
        Unpivot (melt) a DataFrame.
        
        Similar to Power Query's Unpivot Columns operation.
        
        Args:
            df: The DataFrame
            id_vars: Columns to keep as identifiers
            value_vars: Columns to unpivot (None = all except id_vars)
            var_name: Name for the attribute column
            value_name: Name for the value column
        
        Returns:
            Unpivoted DataFrame
        """
        result = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name
        )
        return result
    
    @staticmethod
    def split_column(
        df: pd.DataFrame,
        column: str,
        delimiter: str,
        new_column_names: Optional[List[str]] = None,
        keep_original: bool = False
    ) -> pd.DataFrame:
        """
        Split a column by delimiter.
        
        Similar to Power Query's Split Column by Delimiter.
        
        Args:
            df: The DataFrame
            column: Column to split
            delimiter: Delimiter to split on
            new_column_names: Names for new columns (optional)
            keep_original: Whether to keep the original column
        
        Returns:
            DataFrame with split columns
        """
        result = df.copy()
        
        # Split the column
        split_data = result[column].astype(str).str.split(delimiter, expand=True)
        
        # Name the new columns
        if new_column_names:
            for i, name in enumerate(new_column_names):
                if i < split_data.shape[1]:
                    split_data = split_data.rename(columns={i: name})
        else:
            split_data.columns = [f"{column}_{i+1}" for i in range(split_data.shape[1])]
        
        # Add to result
        result = pd.concat([result, split_data], axis=1)
        
        # Remove original if requested
        if not keep_original:
            result = result.drop(columns=[column])
        
        return result
    
    @staticmethod
    def change_column_type(
        df: pd.DataFrame,
        column: str,
        new_type: str
    ) -> pd.DataFrame:
        """
        Change column data type.
        
        Similar to Power Query's Change Type.
        
        Args:
            df: The DataFrame
            column: Column to change
            new_type: New type ('int', 'float', 'str', 'datetime', 'category')
        
        Returns:
            DataFrame with changed column type
        """
        result = df.copy()
        
        type_mapping = {
            'int': 'int64',
            'float': 'float64',
            'str': 'object',
            'string': 'object',
            'text': 'object',
            'datetime': 'datetime64[ns]',
            'date': 'datetime64[ns]',
            'category': 'category',
            'bool': 'bool',
            'boolean': 'bool'
        }
        
        target_type = type_mapping.get(new_type.lower(), new_type)
        
        try:
            if target_type == 'datetime64[ns]':
                result[column] = pd.to_datetime(result[column], errors='coerce')
            elif target_type == 'category':
                result[column] = result[column].astype('category')
            else:
                result[column] = result[column].astype(target_type)
        except Exception as e:
            raise ValueError(f"Could not convert column '{column}' to {new_type}: {str(e)}")
        
        return result
    
    @staticmethod
    def fill_down(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fill null values with the previous non-null value.
        
        Similar to Power Query's Fill Down.
        """
        result = df.copy()
        for col in columns:
            result[col] = result[col].ffill()
        return result
    
    @staticmethod
    def fill_up(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """
        Fill null values with the next non-null value.
        
        Similar to Power Query's Fill Up.
        """
        result = df.copy()
        for col in columns:
            result[col] = result[col].bfill()
        return result
    
    @staticmethod
    def remove_duplicates_full_row(df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove FULL ROW duplicates only.
        
        IMPORTANT: This removes only rows where ALL columns are identical.
        Duplicate values in individual columns are NORMAL and NOT removed.
        """
        return df.drop_duplicates()
    
    @staticmethod
    def remove_duplicates_by_keys(
        df: pd.DataFrame,
        key_columns: List[str],
        keep: str = 'first'
    ) -> pd.DataFrame:
        """
        Remove duplicates based on specific key columns.
        
        Use this when you have a defined business key (composite primary key).
        
        Args:
            df: The DataFrame
            key_columns: Columns that form the unique key
            keep: Which duplicate to keep ('first', 'last', False for none)
        
        Returns:
            DataFrame with duplicates removed based on keys
        """
        return df.drop_duplicates(subset=key_columns, keep=keep)
