"""
Data Quality Assessment Module
Comprehensive data quality analysis and recommendations
Part of MVP Phase 3: Data Quality Assessment

CRITICAL RULES:
- Duplicate values in individual columns are NORMAL for categorical data
- Only flag FULL ROW duplicates as issues
- Primary/Foreign keys must NOT be treated as numeric measures
- Keys must NOT be checked for outliers
- Always analyze column role before applying rules
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from enum import Enum
import re


class IssueSeverity(Enum):
    """Issue severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ColumnRole(Enum):
    """Column role classification"""
    PRIMARY_KEY = "primary_key"
    FOREIGN_KEY = "foreign_key"
    IDENTIFIER = "identifier"  # Generic ID columns
    CATEGORY = "category"
    DIMENSION = "dimension"
    MEASURE = "measure"
    DATE = "date"
    TEXT = "text"
    UNKNOWN = "unknown"


class IntelligentColumnAnalyzer:
    """
    Intelligent column role detection based on:
    - Column name patterns
    - Data type
    - Data distribution
    - Value patterns
    """
    
    # Patterns for key columns (should NOT be analyzed as numeric measures)
    KEY_PATTERNS = [
        r'.*key$', r'.*_key$', r'.*key_.*',
        r'.*id$', r'.*_id$', r'.*id_.*', r'^id$',
        r'.*code$', r'.*_code$',
        r'.*number$', r'.*_no$', r'.*_num$',
        r'.*index$', r'.*_idx$',
        r'.*pk$', r'.*fk$',
        r'sku', r'upc', r'barcode', r'serial',
    ]
    
    # Patterns for category columns (duplicates are EXPECTED)
    CATEGORY_PATTERNS = [
        r'.*category.*', r'.*type$', r'.*_type$',
        r'.*status.*', r'.*state$',
        r'.*name$', r'.*_name$',
        r'brand', r'color', r'size', r'region',
        r'country', r'city', r'department',
        r'gender', r'segment', r'channel',
        r'group', r'class', r'tier', r'level',
    ]
    
    # Patterns for measure columns (numeric analysis is appropriate)
    MEASURE_PATTERNS = [
        r'.*amount.*', r'.*price.*', r'.*cost.*',
        r'.*revenue.*', r'.*sales.*', r'.*profit.*',
        r'.*quantity.*', r'.*qty.*', r'.*count$',
        r'.*total.*', r'.*sum.*', r'.*avg.*',
        r'.*rate.*', r'.*percent.*', r'.*pct.*',
        r'.*weight.*', r'.*height.*', r'.*width.*',
        r'.*score.*', r'.*rating.*',
    ]
    
    @staticmethod
    def detect_column_role(df: pd.DataFrame, column: str) -> ColumnRole:
        """
        Detect the role of a column based on name, type, and data distribution.
        
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
        if 'datetime' in dtype or 'date' in col_lower:
            return ColumnRole.DATE
        
        # Check if it matches key patterns
        for pattern in IntelligentColumnAnalyzer.KEY_PATTERNS:
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
                # Even if not very unique, key-named columns should be treated as identifiers
                return ColumnRole.IDENTIFIER
        
        # Check for category patterns
        for pattern in IntelligentColumnAnalyzer.CATEGORY_PATTERNS:
            if re.match(pattern, col_lower, re.IGNORECASE):
                return ColumnRole.CATEGORY
        
        # Check for measure patterns (only if numeric)
        if np.issubdtype(col_data.dtype, np.number):
            for pattern in IntelligentColumnAnalyzer.MEASURE_PATTERNS:
                if re.match(pattern, col_lower, re.IGNORECASE):
                    return ColumnRole.MEASURE
        
        # Data-driven detection
        if col_data.dtype == 'object' or str(col_data.dtype) == 'category':
            # Text or categorical data
            uniqueness = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
            if uniqueness < 0.05:  # Less than 5% unique = likely category
                return ColumnRole.CATEGORY
            elif uniqueness > 0.9:  # Very high uniqueness = likely text/description
                return ColumnRole.TEXT
            else:
                return ColumnRole.DIMENSION
        
        # Numeric column analysis
        if np.issubdtype(col_data.dtype, np.number):
            # Check if it's actually an identifier disguised as numeric
            uniqueness = col_data.nunique() / len(col_data) if len(col_data) > 0 else 0
            
            # Check for sequential integer pattern (typical of IDs)
            if col_data.dtype in ['int64', 'int32', 'int']:
                # Check if values are dense/sequential
                min_val = col_data.min()
                max_val = col_data.max()
                expected_range = max_val - min_val + 1
                actual_unique = col_data.nunique()
                
                # If range roughly matches unique count, likely an ID
                if expected_range > 0 and (actual_unique / expected_range) > 0.8:
                    if uniqueness > 0.9:
                        return ColumnRole.IDENTIFIER
            
            # Otherwise, treat as measure
            return ColumnRole.MEASURE
        
        return ColumnRole.UNKNOWN
    
    @staticmethod
    def get_all_column_roles(df: pd.DataFrame) -> Dict[str, ColumnRole]:
        """Get roles for all columns in a DataFrame."""
        return {col: IntelligentColumnAnalyzer.detect_column_role(df, col) for col in df.columns}
    
    @staticmethod
    def get_key_columns(df: pd.DataFrame) -> Set[str]:
        """Get all columns that are identifiers/keys (should NOT be analyzed as numeric)."""
        roles = IntelligentColumnAnalyzer.get_all_column_roles(df)
        return {col for col, role in roles.items() 
                if role in [ColumnRole.PRIMARY_KEY, ColumnRole.FOREIGN_KEY, ColumnRole.IDENTIFIER]}
    
    @staticmethod
    def get_measure_columns(df: pd.DataFrame) -> Set[str]:
        """Get all columns that are measures (appropriate for numeric analysis)."""
        roles = IntelligentColumnAnalyzer.get_all_column_roles(df)
        # Only include numeric columns that are actually measures
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        return {col for col in numeric_cols 
                if roles.get(col) == ColumnRole.MEASURE}
    
    @staticmethod
    def get_category_columns(df: pd.DataFrame) -> Set[str]:
        """Get all columns where duplicates are expected/normal."""
        roles = IntelligentColumnAnalyzer.get_all_column_roles(df)
        return {col for col, role in roles.items() 
                if role in [ColumnRole.CATEGORY, ColumnRole.DIMENSION, ColumnRole.FOREIGN_KEY]}


class DataQualityAssessor:
    """
    Comprehensive data quality assessment with INTELLIGENT column understanding.
    
    CRITICAL RULES:
    - Analyzes column role BEFORE applying any rules
    - Does NOT flag duplicate values in category columns
    - Only flags FULL ROW duplicates
    - Does NOT check key columns for outliers
    - Data-driven, context-aware decisions
    
    Responsibilities:
    - Detect data quality issues
    - Generate recommendations
    - Calculate quality scores
    - Provide actionable insights
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.column_roles = IntelligentColumnAnalyzer.get_all_column_roles(df)
        self.key_columns = IntelligentColumnAnalyzer.get_key_columns(df)
        self.measure_columns = IntelligentColumnAnalyzer.get_measure_columns(df)
        self.category_columns = IntelligentColumnAnalyzer.get_category_columns(df)
        
        # Traditional type detection (for backward compatibility)
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
        self.issues = []
    
    # ==================== PHASE 3.1: Missing Values Detection ====================
    
    def assess_missing_values(self) -> List[Dict]:
        """
        Detect and assess missing values
        
        Returns:
            List of issues with recommendations
        """
        issues = []
        
        for col in self.df.columns:
            missing_count = self.df[col].isna().sum()
            missing_pct = (missing_count / len(self.df)) * 100
            
            if missing_count > 0:
                severity = self._calculate_missing_severity(missing_pct)
                
                issue = {
                    'column': col,
                    'type': 'missing_values',
                    'severity': severity,
                    'count': missing_count,
                    'percentage': missing_pct,
                    'recommendation': self._recommend_missing_strategy(col, missing_pct),
                    'explanation': f"Column '{col}' has {missing_count} missing values ({missing_pct:.1f}%)"
                }
                issues.append(issue)
        
        return issues
    
    def _calculate_missing_severity(self, missing_pct: float) -> IssueSeverity:
        """Calculate severity based on missing percentage"""
        if missing_pct > 50:
            return IssueSeverity.CRITICAL
        elif missing_pct > 25:
            return IssueSeverity.HIGH
        elif missing_pct > 10:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    def _recommend_missing_strategy(self, col: str, missing_pct: float) -> str:
        """Recommend strategy for handling missing values"""
        if missing_pct > 50:
            return "Consider removing this column (too many missing values)"
        elif missing_pct > 25:
            return "Consider removing rows with missing values or use advanced imputation"
        elif col in self.numeric_cols:
            return "Use mean or median imputation"
        elif col in self.categorical_cols:
            return "Use mode imputation or create 'Unknown' category"
        else:
            return "Use forward fill or backward fill for time-series data"
    
    # ==================== PHASE 3.2: Duplicate Detection ====================
    
    def assess_duplicates(self) -> List[Dict]:
        """
        Detect duplicate rows - ONLY FULL ROW DUPLICATES.
        
        IMPORTANT: 
        - Duplicate values in individual columns (like Product Name, Brand, Color)
          are NORMAL for categorical data and should NOT be flagged as issues.
        - Only full row duplicates (all columns identical) are actual data quality issues.
        
        Returns:
            List of issues with recommendations
        """
        issues = []
        
        # ONLY check for FULL ROW duplicates - this is the ONLY valid duplicate check
        full_dup_count = self.df.duplicated().sum()
        full_dup_pct = (full_dup_count / len(self.df)) * 100 if len(self.df) > 0 else 0
        
        if full_dup_count > 0:
            severity = self._calculate_duplicate_severity(full_dup_pct)
            
            issue = {
                'column': 'all_columns',
                'type': 'duplicate_rows',
                'severity': severity,
                'count': full_dup_count,
                'percentage': full_dup_pct,
                'recommendation': 'Remove duplicate rows (entire row is identical)',
                'explanation': f"Found {full_dup_count} fully duplicate rows ({full_dup_pct:.1f}%) - where ALL columns have identical values"
            }
            issues.append(issue)
        
        # NOTE: We intentionally do NOT check for partial duplicates in individual columns
        # because duplicate values in columns like Product Name, Brand, Color, Category, etc.
        # are NORMAL and EXPECTED in categorical/dimension columns.
        # 
        # Previous (incorrect) logic that flagged things like:
        #   "Column 'Product Name' has 77743 duplicate values"
        #   "Column 'Brand' has 80227 duplicate values"
        # These are NOT data quality issues - they are the nature of categorical data.
        
        return issues
    
    def _calculate_duplicate_severity(self, dup_pct: float) -> IssueSeverity:
        """Calculate severity based on duplicate percentage"""
        if dup_pct > 20:
            return IssueSeverity.CRITICAL
        elif dup_pct > 10:
            return IssueSeverity.HIGH
        elif dup_pct > 5:
            return IssueSeverity.MEDIUM
        else:
            return IssueSeverity.LOW
    
    # ==================== PHASE 3.3: Outlier Detection ====================
    
    def assess_outliers(self) -> List[Dict]:
        """
        Detect statistical outliers - ONLY in MEASURE columns.
        
        CRITICAL RULES:
        - Key columns (CustomerKey, ProductKey, OrderID, etc.) must NOT be checked for outliers
        - Keys are identifiers, not numeric values - their distribution is irrelevant
        - Only actual MEASURE columns (price, quantity, amount, etc.) should be analyzed
        
        Returns:
            List of issues with recommendations
        """
        issues = []
        
        # Only analyze columns that are actual MEASURES (not keys/identifiers)
        columns_to_check = self.measure_columns
        
        for col in columns_to_check:
            if col not in self.df.columns:
                continue
                
            # Double-check: skip if column looks like a key
            if col in self.key_columns:
                continue
            
            # Skip columns with too few values
            valid_values = self.df[col].dropna()
            if len(valid_values) < 10:
                continue
            
            Q1 = valid_values.quantile(0.25)
            Q3 = valid_values.quantile(0.75)
            IQR = Q3 - Q1
            
            # Skip if IQR is 0 (constant values)
            if IQR == 0:
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outlier_mask = (valid_values < lower_bound) | (valid_values > upper_bound)
            outlier_count = outlier_mask.sum()
            outlier_pct = (outlier_count / len(self.df)) * 100
            
            # Only flag if there are significant outliers (>1%)
            if outlier_count > 0 and outlier_pct > 1:
                severity = self._calculate_outlier_severity(outlier_pct)
                
                issue = {
                    'column': col,
                    'type': 'outliers',
                    'severity': severity,
                    'count': outlier_count,
                    'percentage': outlier_pct,
                    'bounds': {'lower': lower_bound, 'upper': upper_bound},
                    'recommendation': self._recommend_outlier_strategy(outlier_pct),
                    'explanation': f"Measure column '{col}' has {outlier_count} outliers ({outlier_pct:.1f}%)"
                }
                issues.append(issue)
        
        return issues
    
    def get_column_role_summary(self) -> Dict:
        """
        Get a summary of detected column roles for transparency.
        Useful for debugging and user understanding.
        """
        summary = {
            'primary_keys': [],
            'foreign_keys': [],
            'identifiers': [],
            'categories': [],
            'measures': [],
            'dates': [],
            'text': [],
            'unknown': []
        }
        
        role_mapping = {
            ColumnRole.PRIMARY_KEY: 'primary_keys',
            ColumnRole.FOREIGN_KEY: 'foreign_keys',
            ColumnRole.IDENTIFIER: 'identifiers',
            ColumnRole.CATEGORY: 'categories',
            ColumnRole.DIMENSION: 'categories',  # Group dimensions with categories
            ColumnRole.MEASURE: 'measures',
            ColumnRole.DATE: 'dates',
            ColumnRole.TEXT: 'text',
            ColumnRole.UNKNOWN: 'unknown'
        }
        
        for col, role in self.column_roles.items():
            key = role_mapping.get(role, 'unknown')
            summary[key].append(col)
        
        return summary
        
        return issues
    
    def _calculate_outlier_severity(self, outlier_pct: float) -> IssueSeverity:
        """Calculate severity based on outlier percentage"""
        if outlier_pct > 10:
            return IssueSeverity.MEDIUM
        elif outlier_pct > 5:
            return IssueSeverity.LOW
        else:
            return IssueSeverity.LOW
    
    def _recommend_outlier_strategy(self, outlier_pct: float) -> str:
        """Recommend strategy for handling outliers"""
        if outlier_pct > 10:
            return "Review outliers - may indicate data entry errors or legitimate extreme values"
        else:
            return "Consider removing or capping outliers depending on business context"
    
    # ==================== PHASE 3.4: Data Type Validation ====================
    
    def assess_data_types(self) -> List[Dict]:
        """
        Validate data types and detect mismatches
        
        Returns:
            List of issues with recommendations
        """
        issues = []
        
        for col in self.df.columns:
            # Check for numeric columns with non-numeric values
            if col in self.numeric_cols:
                non_numeric = pd.to_numeric(self.df[col], errors='coerce').isna().sum()
                if non_numeric > 0:
                    issue = {
                        'column': col,
                        'type': 'type_mismatch',
                        'severity': IssueSeverity.MEDIUM,
                        'count': non_numeric,
                        'percentage': (non_numeric / len(self.df)) * 100,
                        'recommendation': 'Convert to numeric or handle non-numeric values',
                        'explanation': f"Column '{col}' has {non_numeric} non-numeric values"
                    }
                    issues.append(issue)
            
            # Check for categorical columns with too many unique values
            if col in self.categorical_cols:
                unique_pct = (self.df[col].nunique() / len(self.df)) * 100
                if unique_pct > 90:
                    issue = {
                        'column': col,
                        'type': 'high_cardinality',
                        'severity': IssueSeverity.LOW,
                        'unique_count': self.df[col].nunique(),
                        'percentage': unique_pct,
                        'recommendation': 'Consider if this should be numeric or if grouping is needed',
                        'explanation': f"Column '{col}' has {self.df[col].nunique()} unique values"
                    }
                    issues.append(issue)
        
        return issues
    
    # ==================== PHASE 3.5: Categorical Consistency ====================
    
    def assess_categorical_consistency(self) -> List[Dict]:
        """
        Check for inconsistent categories
        
        Returns:
            List of issues with recommendations
        """
        issues = []
        
        for col in self.categorical_cols:
            # Check for whitespace issues
            has_whitespace = self.df[col].astype(str).str.strip() != self.df[col].astype(str)
            if has_whitespace.any():
                issue = {
                    'column': col,
                    'type': 'whitespace_inconsistency',
                    'severity': IssueSeverity.MEDIUM,
                    'count': has_whitespace.sum(),
                    'recommendation': 'Strip whitespace from values',
                    'explanation': f"Column '{col}' has values with leading/trailing whitespace"
                }
                issues.append(issue)
            
            # Check for case inconsistency
            unique_values = self.df[col].unique()
            if len(unique_values) > 1:
                lower_values = [str(v).lower() for v in unique_values]
                if len(set(lower_values)) < len(unique_values):
                    issue = {
                        'column': col,
                        'type': 'case_inconsistency',
                        'severity': IssueSeverity.LOW,
                        'recommendation': 'Standardize case (e.g., convert to lowercase)',
                        'explanation': f"Column '{col}' has inconsistent case variations"
                    }
                    issues.append(issue)
        
        return issues
    
    # ==================== PHASE 3.6: Correlation & Relationship Analysis ====================
    
    def assess_relationships(self) -> List[Dict]:
        """
        Analyze data relationships and correlations
        
        Returns:
            List of insights about relationships
        """
        insights = []
        
        if len(self.numeric_cols) < 2:
            return insights
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find high correlations
        for i in range(len(corr_matrix.columns)):
            for j in range(i + 1, len(corr_matrix.columns)):
                corr_value = corr_matrix.iloc[i, j]
                
                if abs(corr_value) > 0.9:
                    insight = {
                        'type': 'high_correlation',
                        'col1': corr_matrix.columns[i],
                        'col2': corr_matrix.columns[j],
                        'correlation': corr_value,
                        'severity': IssueSeverity.MEDIUM,
                        'recommendation': 'Consider removing one of these highly correlated columns',
                        'explanation': f"Columns '{corr_matrix.columns[i]}' and '{corr_matrix.columns[j]}' are highly correlated ({corr_value:.2f})"
                    }
                    insights.append(insight)
        
        return insights
    
    # ==================== Comprehensive Assessment ====================
    
    def assess_all(self) -> Dict[str, any]:
        """
        Perform comprehensive data quality assessment
        
        Returns:
            Complete quality assessment report
        """
        all_issues = []
        
        all_issues.extend(self.assess_missing_values())
        all_issues.extend(self.assess_duplicates())
        all_issues.extend(self.assess_outliers())
        all_issues.extend(self.assess_data_types())
        all_issues.extend(self.assess_categorical_consistency())
        all_issues.extend(self.assess_relationships())
        
        # Calculate quality score
        quality_score = self._calculate_quality_score(all_issues)
        
        # Categorize by severity
        by_severity = {
            'critical': [i for i in all_issues if i['severity'] == IssueSeverity.CRITICAL],
            'high': [i for i in all_issues if i['severity'] == IssueSeverity.HIGH],
            'medium': [i for i in all_issues if i['severity'] == IssueSeverity.MEDIUM],
            'low': [i for i in all_issues if i['severity'] == IssueSeverity.LOW],
        }
        
        return {
            'quality_score': quality_score,
            'total_issues': len(all_issues),
            'by_severity': by_severity,
            'all_issues': all_issues,
            'summary': self._generate_quality_summary(quality_score, by_severity),
            'priority_actions': self._generate_priority_actions(by_severity),
        }
    
    def _calculate_quality_score(self, issues: List[Dict]) -> float:
        """Calculate overall data quality score (0-100)"""
        if not issues:
            return 100.0
        
        severity_weights = {
            IssueSeverity.CRITICAL: 10,
            IssueSeverity.HIGH: 5,
            IssueSeverity.MEDIUM: 2,
            IssueSeverity.LOW: 1,
        }
        
        total_weight = sum(severity_weights.get(i['severity'], 0) for i in issues)
        max_weight = len(self.df.columns) * 10  # Maximum possible weight
        
        score = max(0, 100 - (total_weight / max_weight) * 100)
        return round(score, 1)
    
    def _generate_quality_summary(self, score: float, by_severity: Dict) -> str:
        """Generate human-readable quality summary"""
        if score >= 90:
            status = "Excellent"
        elif score >= 75:
            status = "Good"
        elif score >= 60:
            status = "Fair"
        else:
            status = "Poor"
        
        return f"{status} data quality ({score:.1f}/100) - {by_severity['critical'].__len__()} critical, {by_severity['high'].__len__()} high priority issues"
    
    def _generate_priority_actions(self, by_severity: Dict) -> List[str]:
        """Generate priority actions for data cleaning"""
        actions = []
        
        if by_severity['critical']:
            actions.append(f"🔴 CRITICAL: Address {len(by_severity['critical'])} critical issues first")
            for issue in by_severity['critical'][:3]:
                actions.append(f"  - {issue['explanation']}")
        
        if by_severity['high']:
            actions.append(f"🟠 HIGH: Address {len(by_severity['high'])} high-priority issues")
            for issue in by_severity['high'][:2]:
                actions.append(f"  - {issue['explanation']}")
        
        if by_severity['medium']:
            actions.append(f"🟡 MEDIUM: Consider {len(by_severity['medium'])} medium-priority improvements")
        
        return actions
