"""
KPI Intelligence Module
Intelligent KPI generation with proper aggregation function selection

CRITICAL RULES:
- Primary/Foreign Keys must NEVER be summed or averaged
- Keys are ONLY used for COUNT, DISTINCT COUNT, grouping, filtering
- Every KPI must document: column used, function applied, business definition
- Aggregation must match business meaning, not default behavior
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import re


class ColumnRole(Enum):
    """Column role classification for KPI generation"""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    IDENTIFIER = "identifier"
    CATEGORY = "category"
    DIMENSION = "dimension"
    MEASURE = "measure"
    DATE = "date"
    TEXT = "text"
    UNKNOWN = "unknown"


class AggregationFunction(Enum):
    """Valid aggregation functions for KPIs"""
    COUNT = "COUNT"
    DISTINCT_COUNT = "DISTINCT COUNT"
    SUM = "SUM"
    AVERAGE = "AVG"
    MIN = "MIN"
    MAX = "MAX"
    MEDIAN = "MEDIAN"
    FIRST = "FIRST"
    LAST = "LAST"


@dataclass
class KPIDefinition:
    """Complete KPI definition with documentation"""
    name: str
    column: str
    function: AggregationFunction
    value: Any
    formatted_value: str
    business_definition: str
    column_role: ColumnRole
    is_valid: bool
    warning: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'name': self.name,
            'column': self.column,
            'function': self.function.value,
            'value': self.value,
            'formatted_value': self.formatted_value,
            'business_definition': self.business_definition,
            'column_role': self.column_role.value,
            'is_valid': self.is_valid,
            'warning': self.warning
        }
    
    def get_display_label(self) -> str:
        """Get display label with function indicator"""
        return f"{self.name} ({self.function.value})"


class KPIColumnAnalyzer:
    """
    Analyze columns to determine their role and appropriate aggregation functions.
    
    CRITICAL: Keys must NEVER be summed/averaged.
    """
    
    # Patterns for key columns - MUST NOT BE SUMMED
    KEY_PATTERNS = [
        r'.*key$', r'.*_key$', r'.*key_.*',
        r'.*id$', r'.*_id$', r'.*id_.*', r'^id$',
        r'.*code$', r'.*_code$',
        r'.*number$', r'.*_no$', r'.*_num$', r'^no$',
        r'.*index$', r'.*_idx$',
        r'.*pk$', r'.*fk$',
        r'sku', r'upc', r'barcode', r'serial',
        r'.*invoice.*', r'.*order.*id', r'.*customer.*id',
        r'.*product.*id', r'.*employee.*id', r'.*supplier.*id',
    ]
    
    # Patterns for measure columns - CAN be summed/averaged
    MEASURE_PATTERNS = [
        r'.*amount.*', r'.*price.*', r'.*cost.*',
        r'.*revenue.*', r'.*sales.*', r'.*profit.*',
        r'.*quantity.*', r'.*qty.*', 
        r'.*total.*', r'.*subtotal.*',
        r'.*discount.*', r'.*tax.*', r'.*fee.*',
        r'.*weight.*', r'.*height.*', r'.*width.*', r'.*length.*',
        r'.*score.*', r'.*rating.*', r'.*points.*',
        r'.*balance.*', r'.*value$', r'.*_value$',
        r'.*margin.*', r'.*rate$', r'.*_rate$',
        r'.*percent.*', r'.*pct.*', r'.*ratio.*',
    ]
    
    # Patterns for count-only columns - use COUNT, not SUM
    COUNT_PATTERNS = [
        r'^count.*', r'.*_count$', r'.*count_.*',
        r'^num_.*', r'.*_num$',
        r'^n_.*', r'.*_n$',
    ]
    
    @classmethod
    def detect_column_role(cls, df: pd.DataFrame, column: str) -> ColumnRole:
        """
        Detect the role of a column to determine valid aggregation functions.
        
        Args:
            df: The DataFrame
            column: Column name to analyze
            
        Returns:
            ColumnRole enum value
        """
        col_lower = column.lower().strip()
        col_data = df[column]
        dtype = str(col_data.dtype)
        
        # Check for date columns first
        if 'datetime' in dtype or 'date' in col_lower or 'time' in col_lower:
            return ColumnRole.DATE
        
        # Check if it matches key patterns - CRITICAL: These cannot be summed
        for pattern in cls.KEY_PATTERNS:
            if re.match(pattern, col_lower, re.IGNORECASE):
                # Verify it behaves like a key (high uniqueness)
                uniqueness_ratio = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
                if uniqueness_ratio > 0.5:  # More than 50% unique values
                    if 'pk' in col_lower or col_lower.endswith('key') or col_lower == 'id':
                        return ColumnRole.PRIMARY_KEY
                    elif 'fk' in col_lower or '_id' in col_lower:
                        return ColumnRole.FOREIGN_KEY
                    else:
                        return ColumnRole.IDENTIFIER
                # Even if not very unique, key-named columns are identifiers
                return ColumnRole.IDENTIFIER
        
        # Check for measure patterns (only if numeric)
        if np.issubdtype(col_data.dtype, np.number):
            for pattern in cls.MEASURE_PATTERNS:
                if re.match(pattern, col_lower, re.IGNORECASE):
                    return ColumnRole.MEASURE
        
        # Data-driven detection for text columns
        if col_data.dtype == 'object' or str(col_data.dtype) == 'category':
            uniqueness = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
            if uniqueness < 0.05:  # Less than 5% unique = likely category
                return ColumnRole.CATEGORY
            elif uniqueness > 0.9:  # Very high uniqueness = likely text/ID
                return ColumnRole.IDENTIFIER
            else:
                return ColumnRole.DIMENSION
        
        # Numeric column analysis
        if np.issubdtype(col_data.dtype, np.number):
            uniqueness = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
            
            # Check if it's actually an identifier disguised as numeric
            if col_data.dtype in ['int64', 'int32', 'int']:
                min_val = col_data.min()
                max_val = col_data.max()
                expected_range = max_val - min_val + 1 if max_val > min_val else 1
                actual_unique = col_data.nunique()
                
                # If range roughly matches unique count and high uniqueness, likely an ID
                if expected_range > 0 and (actual_unique / expected_range) > 0.8 and uniqueness > 0.9:
                    return ColumnRole.IDENTIFIER
            
            # Otherwise, treat as measure
            return ColumnRole.MEASURE
        
        return ColumnRole.UNKNOWN
    
    @classmethod
    def get_valid_aggregations(cls, role: ColumnRole) -> List[AggregationFunction]:
        """
        Get list of valid aggregation functions for a column role.
        
        CRITICAL: Keys can ONLY use COUNT/DISTINCT_COUNT
        """
        if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]:
            # Keys can ONLY be counted, NEVER summed or averaged
            return [AggregationFunction.COUNT, AggregationFunction.DISTINCT_COUNT]
        
        elif role == ColumnRole.MEASURE:
            # Measures can use all numeric aggregations
            return [
                AggregationFunction.SUM,
                AggregationFunction.AVERAGE,
                AggregationFunction.COUNT,
                AggregationFunction.MIN,
                AggregationFunction.MAX,
                AggregationFunction.MEDIAN
            ]
        
        elif role in [ColumnRole.CATEGORY, ColumnRole.DIMENSION]:
            # Categories can only be counted
            return [AggregationFunction.COUNT, AggregationFunction.DISTINCT_COUNT]
        
        elif role == ColumnRole.DATE:
            # Dates can be counted or get min/max
            return [
                AggregationFunction.COUNT,
                AggregationFunction.MIN,
                AggregationFunction.MAX,
                AggregationFunction.DISTINCT_COUNT
            ]
        
        else:
            # Unknown - be conservative
            return [AggregationFunction.COUNT]
    
    @classmethod
    def get_recommended_aggregation(cls, df: pd.DataFrame, column: str) -> Tuple[AggregationFunction, str]:
        """
        Get the recommended aggregation function for a column based on its role.
        
        Returns:
            Tuple of (AggregationFunction, explanation)
        """
        role = cls.detect_column_role(df, column)
        col_lower = column.lower()
        
        if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]:
            # For keys, ALWAYS use COUNT or DISTINCT COUNT
            return (
                AggregationFunction.DISTINCT_COUNT,
                f"'{column}' is a {role.value} - using DISTINCT COUNT (keys cannot be summed)"
            )
        
        elif role == ColumnRole.MEASURE:
            # Determine best aggregation based on column name
            if any(k in col_lower for k in ['price', 'rate', 'avg', 'average', 'mean', 'percent', 'pct', 'ratio']):
                return (
                    AggregationFunction.AVERAGE,
                    f"'{column}' appears to be a rate/average metric - using AVERAGE"
                )
            elif any(k in col_lower for k in ['count', 'num', 'number']):
                return (
                    AggregationFunction.SUM,
                    f"'{column}' appears to be a count metric - using SUM to aggregate counts"
                )
            else:
                return (
                    AggregationFunction.SUM,
                    f"'{column}' is a measure - using SUM for total"
                )
        
        elif role in [ColumnRole.CATEGORY, ColumnRole.DIMENSION]:
            return (
                AggregationFunction.DISTINCT_COUNT,
                f"'{column}' is a {role.value} - using DISTINCT COUNT to count unique values"
            )
        
        elif role == ColumnRole.DATE:
            return (
                AggregationFunction.COUNT,
                f"'{column}' is a date column - using COUNT for number of records"
            )
        
        else:
            return (
                AggregationFunction.COUNT,
                f"'{column}' has unknown role - defaulting to COUNT"
            )
    
    @classmethod
    def is_aggregation_valid(cls, df: pd.DataFrame, column: str, aggregation: AggregationFunction) -> Tuple[bool, Optional[str]]:
        """
        Check if an aggregation function is valid for a column.
        
        Returns:
            Tuple of (is_valid, warning_message)
        """
        role = cls.detect_column_role(df, column)
        valid_aggregations = cls.get_valid_aggregations(role)
        
        if aggregation in valid_aggregations:
            return (True, None)
        
        # Invalid aggregation
        if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]:
            return (
                False,
                f"⚠️ INVALID: '{column}' is a {role.value}. Cannot use {aggregation.value}. "
                f"Keys can ONLY use COUNT or DISTINCT COUNT."
            )
        
        return (
            False,
            f"⚠️ {aggregation.value} is not recommended for {role.value} column '{column}'. "
            f"Valid options: {', '.join([a.value for a in valid_aggregations])}"
        )


class IntelligentKPIGenerator:
    """
    Generate KPIs with intelligent aggregation function selection.
    
    Every KPI includes:
    - Column used
    - Aggregation function applied
    - Business definition
    - Validation status
    """
    
    def __init__(self, df: pd.DataFrame, dataset_name: str = "Dataset"):
        self.df = df
        self.dataset_name = dataset_name
        self.column_roles = {}
        self._analyze_all_columns()
    
    def _analyze_all_columns(self):
        """Analyze all columns and cache their roles."""
        for col in self.df.columns:
            self.column_roles[col] = KPIColumnAnalyzer.detect_column_role(self.df, col)
    
    def get_column_role(self, column: str) -> ColumnRole:
        """Get the role of a column."""
        return self.column_roles.get(column, ColumnRole.UNKNOWN)
    
    def get_key_columns(self) -> List[str]:
        """Get all columns that are keys/identifiers (should NOT be summed)."""
        return [col for col, role in self.column_roles.items() 
                if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]]
    
    def get_measure_columns(self) -> List[str]:
        """Get all columns that are measures (CAN be summed)."""
        return [col for col, role in self.column_roles.items() 
                if role == ColumnRole.MEASURE]
    
    def get_category_columns(self) -> List[str]:
        """Get all category/dimension columns."""
        return [col for col, role in self.column_roles.items() 
                if role in [ColumnRole.CATEGORY, ColumnRole.DIMENSION]]
    
    def generate_kpi(
        self,
        column: str,
        aggregation: Optional[AggregationFunction] = None,
        custom_name: Optional[str] = None
    ) -> KPIDefinition:
        """
        Generate a single KPI with proper validation and documentation.
        
        If no aggregation is specified, the system will choose the correct one.
        """
        role = self.get_column_role(column)
        
        # Auto-select aggregation if not provided
        if aggregation is None:
            aggregation, explanation = KPIColumnAnalyzer.get_recommended_aggregation(self.df, column)
        
        # Validate the aggregation
        is_valid, warning = KPIColumnAnalyzer.is_aggregation_valid(self.df, column, aggregation)
        
        # If invalid aggregation on a key column, force correct aggregation
        if not is_valid and role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]:
            aggregation = AggregationFunction.DISTINCT_COUNT
            warning = f"⚠️ Corrected: Changed to DISTINCT COUNT (keys cannot be {aggregation.value})"
            is_valid = True
        
        # Calculate the value
        value = self._calculate_aggregation(column, aggregation)
        
        # Format the value
        formatted_value = self._format_value(column, value, aggregation)
        
        # Generate business definition
        business_definition = self._generate_business_definition(column, aggregation, role)
        
        # Generate name
        if custom_name:
            name = custom_name
        else:
            name = self._generate_kpi_name(column, aggregation)
        
        return KPIDefinition(
            name=name,
            column=column,
            function=aggregation,
            value=value,
            formatted_value=formatted_value,
            business_definition=business_definition,
            column_role=role,
            is_valid=is_valid,
            warning=warning
        )
    
    def _calculate_aggregation(self, column: str, aggregation: AggregationFunction) -> Any:
        """Calculate the aggregation value."""
        col_data = self.df[column]
        
        if aggregation == AggregationFunction.COUNT:
            return col_data.count()
        elif aggregation == AggregationFunction.DISTINCT_COUNT:
            return col_data.nunique()
        elif aggregation == AggregationFunction.SUM:
            return col_data.sum() if np.issubdtype(col_data.dtype, np.number) else 0
        elif aggregation == AggregationFunction.AVERAGE:
            return col_data.mean() if np.issubdtype(col_data.dtype, np.number) else 0
        elif aggregation == AggregationFunction.MIN:
            return col_data.min()
        elif aggregation == AggregationFunction.MAX:
            return col_data.max()
        elif aggregation == AggregationFunction.MEDIAN:
            return col_data.median() if np.issubdtype(col_data.dtype, np.number) else 0
        else:
            return col_data.count()
    
    def _format_value(self, column: str, value: Any, aggregation: AggregationFunction) -> str:
        """Format the KPI value for display."""
        col_lower = column.lower()
        
        # Currency formatting
        if any(k in col_lower for k in ['revenue', 'amount', 'total', 'sales', 'cost', 'price', 'profit', 'value']):
            if isinstance(value, (int, float)):
                if abs(value) >= 1_000_000:
                    return f"${value/1_000_000:,.1f}M"
                elif abs(value) >= 1_000:
                    return f"${value/1_000:,.1f}K"
                else:
                    return f"${value:,.2f}"
        
        # Percentage formatting
        if any(k in col_lower for k in ['percent', 'pct', 'rate', 'ratio']):
            if isinstance(value, (int, float)):
                return f"{value:.1f}%"
        
        # Count formatting
        if aggregation in [AggregationFunction.COUNT, AggregationFunction.DISTINCT_COUNT]:
            if isinstance(value, (int, float)):
                return f"{int(value):,}"
        
        # Default numeric formatting
        if isinstance(value, float):
            if abs(value) >= 1_000_000:
                return f"{value/1_000_000:,.1f}M"
            elif abs(value) >= 1_000:
                return f"{value/1_000:,.1f}K"
            elif abs(value) >= 1:
                return f"{value:,.2f}"
            else:
                return f"{value:.4f}"
        elif isinstance(value, int):
            return f"{value:,}"
        else:
            return str(value)
    
    def _generate_kpi_name(self, column: str, aggregation: AggregationFunction) -> str:
        """Generate a descriptive KPI name."""
        col_clean = column.replace('_', ' ').title()
        
        if aggregation == AggregationFunction.SUM:
            return f"Total {col_clean}"
        elif aggregation == AggregationFunction.AVERAGE:
            return f"Average {col_clean}"
        elif aggregation == AggregationFunction.COUNT:
            return f"{col_clean} Count"
        elif aggregation == AggregationFunction.DISTINCT_COUNT:
            return f"Unique {col_clean}"
        elif aggregation == AggregationFunction.MIN:
            return f"Min {col_clean}"
        elif aggregation == AggregationFunction.MAX:
            return f"Max {col_clean}"
        elif aggregation == AggregationFunction.MEDIAN:
            return f"Median {col_clean}"
        else:
            return col_clean
    
    def _generate_business_definition(self, column: str, aggregation: AggregationFunction, role: ColumnRole) -> str:
        """Generate a business definition for the KPI."""
        col_clean = column.replace('_', ' ').lower()
        
        if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]:
            return f"Number of unique {col_clean} values (identifier column - count only)"
        
        if aggregation == AggregationFunction.SUM:
            return f"Sum of all {col_clean} values across all records"
        elif aggregation == AggregationFunction.AVERAGE:
            return f"Average {col_clean} value per record"
        elif aggregation == AggregationFunction.COUNT:
            return f"Total number of records with {col_clean}"
        elif aggregation == AggregationFunction.DISTINCT_COUNT:
            return f"Number of unique {col_clean} values"
        elif aggregation == AggregationFunction.MIN:
            return f"Minimum {col_clean} value"
        elif aggregation == AggregationFunction.MAX:
            return f"Maximum {col_clean} value"
        elif aggregation == AggregationFunction.MEDIAN:
            return f"Median {col_clean} value"
        else:
            return f"{col_clean} metric"
    
    def generate_all_kpis(self, max_kpis: int = 20) -> List[KPIDefinition]:
        """
        Generate all relevant KPIs for the dataset with proper aggregation functions.
        
        Prioritizes:
        1. Measure columns (SUM/AVG)
        2. Key columns (DISTINCT COUNT)
        3. Category columns (DISTINCT COUNT)
        """
        kpis = []
        
        # Generate KPIs for measure columns (most important)
        for col in self.get_measure_columns():
            if len(kpis) >= max_kpis:
                break
            kpis.append(self.generate_kpi(col))
        
        # Generate KPIs for key columns (count unique values)
        for col in self.get_key_columns():
            if len(kpis) >= max_kpis:
                break
            kpis.append(self.generate_kpi(col, AggregationFunction.DISTINCT_COUNT))
        
        # Generate KPIs for category columns
        for col in self.get_category_columns()[:5]:  # Limit categories
            if len(kpis) >= max_kpis:
                break
            kpis.append(self.generate_kpi(col, AggregationFunction.DISTINCT_COUNT))
        
        return kpis
    
    def get_column_summary(self) -> Dict[str, List[str]]:
        """Get a summary of columns by role."""
        return {
            'keys': self.get_key_columns(),
            'measures': self.get_measure_columns(),
            'categories': self.get_category_columns(),
            'dates': [col for col, role in self.column_roles.items() if role == ColumnRole.DATE]
        }


def validate_kpi_request(df: pd.DataFrame, column: str, aggregation: str) -> Tuple[bool, str, Optional[str]]:
    """
    Validate a KPI request and return the corrected aggregation if needed.
    
    Args:
        df: DataFrame
        column: Column name
        aggregation: Requested aggregation (sum, avg, count, etc.)
        
    Returns:
        Tuple of (is_valid, corrected_aggregation, warning_message)
    """
    agg_mapping = {
        'sum': AggregationFunction.SUM,
        'avg': AggregationFunction.AVERAGE,
        'average': AggregationFunction.AVERAGE,
        'mean': AggregationFunction.AVERAGE,
        'count': AggregationFunction.COUNT,
        'distinctcount': AggregationFunction.DISTINCT_COUNT,
        'distinct_count': AggregationFunction.DISTINCT_COUNT,
        'min': AggregationFunction.MIN,
        'max': AggregationFunction.MAX,
        'median': AggregationFunction.MEDIAN,
    }
    
    requested_agg = agg_mapping.get(aggregation.lower(), AggregationFunction.COUNT)
    
    is_valid, warning = KPIColumnAnalyzer.is_aggregation_valid(df, column, requested_agg)
    
    if is_valid:
        return (True, aggregation, None)
    else:
        # Get the recommended aggregation
        correct_agg, _ = KPIColumnAnalyzer.get_recommended_aggregation(df, column)
        
        # Map back to string
        corrected = correct_agg.value.lower().replace(' ', '')
        
        return (False, corrected, warning)
