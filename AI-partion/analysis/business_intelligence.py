"""
Business Intelligence Module
Automatic business context understanding and KPI identification
Part of MVP Phase 2: Data Intelligence & Finetuning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime


class BusinessIntelligence:
    """
    Analyze dataset to understand business context
    
    Responsibilities:
    - Infer business type (Sales, HR, Finance, Inventory, etc.)
    - Identify key entities (customers, orders, products, etc.)
    - Detect KPIs and metrics
    - Identify time dimensions
    - Generate business context summary
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.columns = df.columns.tolist()
        self.column_lower = [col.lower() for col in self.columns]
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.datetime_cols = df.select_dtypes(include=['datetime64']).columns.tolist()
    
    # ==================== PHASE 2.1: Business Type Inference ====================
    
    def infer_business_type(self) -> Dict[str, any]:
        """
        Infer the business domain from data patterns
        
        Returns:
            Dict with business_type, confidence, indicators
        """
        indicators = {
            'sales': self._check_sales_indicators(),
            'hr': self._check_hr_indicators(),
            'finance': self._check_finance_indicators(),
            'inventory': self._check_inventory_indicators(),
            'customer': self._check_customer_indicators(),
            'operations': self._check_operations_indicators(),
        }
        
        # Find highest scoring business type
        scores = {k: v['score'] for k, v in indicators.items()}
        top_type = max(scores, key=scores.get)
        confidence = scores[top_type] / 100.0
        
        return {
            'business_type': top_type,
            'confidence': confidence,
            'indicators': indicators[top_type]['indicators'],
            'all_scores': scores,
            'description': self._get_business_description(top_type)
        }
    
    def _check_sales_indicators(self) -> Dict:
        """Check for sales data indicators"""
        indicators = []
        score = 0
        
        sales_keywords = ['sales', 'revenue', 'order', 'customer', 'product', 'quantity', 'price', 'total']
        matching = sum(1 for col in self.column_lower if any(kw in col for kw in sales_keywords))
        
        if matching >= 3:
            indicators.append(f"Found {matching} sales-related columns")
            score += 30
        
        if any('date' in col.lower() for col in self.column_lower):
            indicators.append("Time dimension detected")
            score += 20
        
        if any('customer' in col.lower() for col in self.column_lower):
            indicators.append("Customer dimension detected")
            score += 20
        
        if any('product' in col.lower() for col in self.column_lower):
            indicators.append("Product dimension detected")
            score += 20
        
        if len(self.numeric_cols) >= 3:
            indicators.append("Multiple numeric metrics")
            score += 10
        
        return {'score': score, 'indicators': indicators}
    
    def _check_hr_indicators(self) -> Dict:
        """Check for HR data indicators"""
        indicators = []
        score = 0
        
        hr_keywords = ['employee', 'salary', 'department', 'hire', 'performance', 'experience', 'status']
        matching = sum(1 for col in self.column_lower if any(kw in col for kw in hr_keywords))
        
        if matching >= 2:
            indicators.append(f"Found {matching} HR-related columns")
            score += 40
        
        if any('salary' in col.lower() or 'compensation' in col.lower() for col in self.column_lower):
            indicators.append("Compensation data detected")
            score += 20
        
        if any('department' in col.lower() for col in self.column_lower):
            indicators.append("Department structure detected")
            score += 15
        
        if any('performance' in col.lower() for col in self.column_lower):
            indicators.append("Performance metrics detected")
            score += 15
        
        return {'score': score, 'indicators': indicators}
    
    def _check_finance_indicators(self) -> Dict:
        """Check for finance data indicators"""
        indicators = []
        score = 0
        
        finance_keywords = ['transaction', 'amount', 'budget', 'expense', 'income', 'category', 'department']
        matching = sum(1 for col in self.column_lower if any(kw in col for kw in finance_keywords))
        
        if matching >= 2:
            indicators.append(f"Found {matching} finance-related columns")
            score += 35
        
        if any('amount' in col.lower() or 'value' in col.lower() for col in self.column_lower):
            indicators.append("Monetary values detected")
            score += 20
        
        if any('category' in col.lower() for col in self.column_lower):
            indicators.append("Categorization detected")
            score += 15
        
        if any('date' in col.lower() for col in self.column_lower):
            indicators.append("Time-series data detected")
            score += 15
        
        return {'score': score, 'indicators': indicators}
    
    def _check_inventory_indicators(self) -> Dict:
        """Check for inventory data indicators"""
        indicators = []
        score = 0
        
        inventory_keywords = ['product', 'stock', 'quantity', 'supplier', 'warehouse', 'reorder', 'cost']
        matching = sum(1 for col in self.column_lower if any(kw in col for kw in inventory_keywords))
        
        if matching >= 2:
            indicators.append(f"Found {matching} inventory-related columns")
            score += 40
        
        if any('stock' in col.lower() or 'quantity' in col.lower() for col in self.column_lower):
            indicators.append("Stock levels detected")
            score += 20
        
        if any('supplier' in col.lower() for col in self.column_lower):
            indicators.append("Supplier information detected")
            score += 15
        
        if any('reorder' in col.lower() for col in self.column_lower):
            indicators.append("Reorder logic detected")
            score += 15
        
        return {'score': score, 'indicators': indicators}
    
    def _check_customer_indicators(self) -> Dict:
        """Check for customer data indicators"""
        indicators = []
        score = 0
        
        customer_keywords = ['customer', 'client', 'email', 'phone', 'city', 'country', 'purchase', 'segment']
        matching = sum(1 for col in self.column_lower if any(kw in col for kw in customer_keywords))
        
        if matching >= 3:
            indicators.append(f"Found {matching} customer-related columns")
            score += 35
        
        if any('email' in col.lower() or 'phone' in col.lower() for col in self.column_lower):
            indicators.append("Contact information detected")
            score += 15
        
        if any('segment' in col.lower() or 'category' in col.lower() for col in self.column_lower):
            indicators.append("Customer segmentation detected")
            score += 15
        
        if any('purchase' in col.lower() or 'spent' in col.lower() for col in self.column_lower):
            indicators.append("Purchase behavior detected")
            score += 15
        
        return {'score': score, 'indicators': indicators}
    
    def _check_operations_indicators(self) -> Dict:
        """Check for operations data indicators"""
        indicators = []
        score = 0
        
        ops_keywords = ['process', 'status', 'date', 'time', 'duration', 'efficiency', 'metric']
        matching = sum(1 for col in self.column_lower if any(kw in col for kw in ops_keywords))
        
        if matching >= 2:
            indicators.append(f"Found {matching} operations-related columns")
            score += 25
        
        if any('date' in col.lower() or 'time' in col.lower() for col in self.column_lower):
            indicators.append("Time tracking detected")
            score += 15
        
        if any('status' in col.lower() for col in self.column_lower):
            indicators.append("Status tracking detected")
            score += 15
        
        return {'score': score, 'indicators': indicators}
    
    def _get_business_description(self, business_type: str) -> str:
        """Get human-readable description of business type"""
        descriptions = {
            'sales': 'Sales & Revenue Analysis - Track sales transactions, revenue trends, and customer purchases',
            'hr': 'Human Resources - Employee data, compensation, performance, and organizational structure',
            'finance': 'Financial Management - Budget tracking, expenses, income, and financial transactions',
            'inventory': 'Inventory Management - Stock levels, products, suppliers, and warehouse operations',
            'customer': 'Customer Analytics - Customer information, behavior, segmentation, and lifetime value',
            'operations': 'Operations Management - Process tracking, efficiency metrics, and operational performance',
        }
        return descriptions.get(business_type, 'General Business Data')
    
    # ==================== PHASE 2.2: Entity Detection ====================
    
    def detect_entities(self) -> Dict[str, List[str]]:
        """
        Identify key business entities in the dataset
        
        Returns:
            Dict mapping entity types to detected columns
        """
        entities = {
            'customers': self._detect_customer_columns(),
            'products': self._detect_product_columns(),
            'orders': self._detect_order_columns(),
            'transactions': self._detect_transaction_columns(),
            'time': self._detect_time_columns(),
            'locations': self._detect_location_columns(),
            'metrics': self._detect_metric_columns(),
        }
        
        return {k: v for k, v in entities.items() if v}
    
    def _detect_customer_columns(self) -> List[str]:
        """Detect customer-related columns"""
        keywords = ['customer', 'client', 'user', 'account', 'name', 'email', 'phone']
        return [col for col in self.columns if any(kw in col.lower() for kw in keywords)]
    
    def _detect_product_columns(self) -> List[str]:
        """Detect product-related columns"""
        keywords = ['product', 'item', 'sku', 'category', 'brand']
        return [col for col in self.columns if any(kw in col.lower() for kw in keywords)]
    
    def _detect_order_columns(self) -> List[str]:
        """Detect order-related columns"""
        keywords = ['order', 'transaction', 'sale', 'purchase', 'invoice']
        return [col for col in self.columns if any(kw in col.lower() for kw in keywords)]
    
    def _detect_transaction_columns(self) -> List[str]:
        """Detect transaction-related columns"""
        keywords = ['amount', 'value', 'price', 'cost', 'revenue', 'expense']
        return [col for col in self.columns if any(kw in col.lower() for kw in keywords)]
    
    def _detect_time_columns(self) -> List[str]:
        """Detect time-related columns"""
        time_cols = self.datetime_cols.copy()
        keywords = ['date', 'time', 'year', 'month', 'day', 'quarter']
        time_cols.extend([col for col in self.columns if any(kw in col.lower() for kw in keywords)])
        return list(set(time_cols))
    
    def _detect_location_columns(self) -> List[str]:
        """Detect location-related columns"""
        keywords = ['city', 'country', 'state', 'region', 'location', 'address']
        return [col for col in self.columns if any(kw in col.lower() for kw in keywords)]
    
    def _detect_metric_columns(self) -> List[str]:
        """Detect metric/KPI columns"""
        return self.numeric_cols.copy()
    
    # ==================== PHASE 2.3: KPI Identification ====================
    
    def identify_kpis(self) -> Dict[str, List[Dict]]:
        """
        Identify key performance indicators based on business type
        
        Returns:
            Dict with KPI suggestions by category
        """
        business_type = self.infer_business_type()['business_type']
        
        kpi_map = {
            'sales': self._get_sales_kpis(),
            'hr': self._get_hr_kpis(),
            'finance': self._get_finance_kpis(),
            'inventory': self._get_inventory_kpis(),
            'customer': self._get_customer_kpis(),
            'operations': self._get_operations_kpis(),
        }
        
        return kpi_map.get(business_type, {})
    
    def _get_sales_kpis(self) -> Dict:
        """Get sales KPIs"""
        kpis = {
            'revenue': self._find_columns(['sales', 'revenue', 'total']),
            'quantity': self._find_columns(['quantity', 'units', 'count']),
            'customer_count': self._find_columns(['customer', 'client']),
            'average_order_value': 'Calculated from revenue and order count',
            'growth_rate': 'Calculated from time-series data',
        }
        return {k: v for k, v in kpis.items() if v}
    
    def _get_hr_kpis(self) -> Dict:
        """Get HR KPIs"""
        kpis = {
            'headcount': self._find_columns(['employee', 'staff']),
            'salary': self._find_columns(['salary', 'compensation']),
            'performance': self._find_columns(['performance', 'rating', 'score']),
            'turnover': 'Calculated from hire/exit dates',
            'experience': self._find_columns(['experience', 'tenure']),
        }
        return {k: v for k, v in kpis.items() if v}
    
    def _get_finance_kpis(self) -> Dict:
        """Get finance KPIs"""
        kpis = {
            'total_amount': self._find_columns(['amount', 'value', 'total']),
            'budget': self._find_columns(['budget', 'allocated']),
            'expense': self._find_columns(['expense', 'cost']),
            'variance': 'Calculated from budget vs actual',
            'category_breakdown': self._find_columns(['category', 'department']),
        }
        return {k: v for k, v in kpis.items() if v}
    
    def _get_inventory_kpis(self) -> Dict:
        """Get inventory KPIs"""
        kpis = {
            'stock_level': self._find_columns(['stock', 'quantity', 'inventory']),
            'reorder_point': self._find_columns(['reorder', 'minimum']),
            'supplier': self._find_columns(['supplier', 'vendor']),
            'turnover_rate': 'Calculated from stock movements',
            'stockout_risk': 'Calculated from current vs reorder levels',
        }
        return {k: v for k, v in kpis.items() if v}
    
    def _get_customer_kpis(self) -> Dict:
        """Get customer KPIs"""
        kpis = {
            'customer_count': self._find_columns(['customer', 'client']),
            'total_spent': self._find_columns(['spent', 'value', 'revenue']),
            'purchase_frequency': 'Calculated from transaction history',
            'segment': self._find_columns(['segment', 'category', 'tier']),
            'lifetime_value': 'Calculated from purchase history',
        }
        return {k: v for k, v in kpis.items() if v}
    
    def _get_operations_kpis(self) -> Dict:
        """Get operations KPIs"""
        kpis = {
            'process_count': 'Calculated from transaction count',
            'status': self._find_columns(['status', 'state']),
            'efficiency': self._find_columns(['efficiency', 'utilization']),
            'duration': self._find_columns(['duration', 'time', 'hours']),
            'completion_rate': 'Calculated from completed vs total',
        }
        return {k: v for k, v in kpis.items() if v}
    
    def _find_columns(self, keywords: List[str]) -> Optional[str]:
        """Find columns matching keywords"""
        for col in self.columns:
            if any(kw in col.lower() for kw in keywords):
                return col
        return None
    
    # ==================== PHASE 2.4: Time Dimension Detection ====================
    
    def detect_time_dimension(self) -> Dict[str, any]:
        """
        Detect and analyze time dimension in data
        
        Returns:
            Dict with time dimension info
        """
        if not self.datetime_cols:
            return {'detected': False, 'reason': 'No datetime columns found'}
        
        time_col = self.datetime_cols[0]
        time_data = pd.to_datetime(self.df[time_col], errors='coerce')
        
        # Remove NaT values
        time_data = time_data.dropna()
        
        if len(time_data) == 0:
            return {'detected': False, 'reason': 'No valid datetime values'}
        
        time_range = time_data.max() - time_data.min()
        
        # Determine granularity
        granularity = self._determine_time_granularity(time_data)
        
        return {
            'detected': True,
            'column': time_col,
            'start_date': time_data.min(),
            'end_date': time_data.max(),
            'duration_days': time_range.days,
            'granularity': granularity,
            'unique_periods': len(time_data.unique()),
            'has_seasonality': self._check_seasonality(time_data),
        }
    
    def _determine_time_granularity(self, time_series: pd.Series) -> str:
        """Determine time series granularity"""
        if len(time_series) < 2:
            return 'unknown'
        
        time_diffs = time_series.sort_values().diff().dropna()
        min_diff = time_diffs.min()
        
        if min_diff.days == 0:
            return 'hourly'
        elif min_diff.days == 1:
            return 'daily'
        elif min_diff.days <= 7:
            return 'weekly'
        elif min_diff.days <= 31:
            return 'monthly'
        elif min_diff.days <= 365:
            return 'yearly'
        else:
            return 'irregular'
    
    def _check_seasonality(self, time_series: pd.Series) -> bool:
        """Check if data shows seasonality patterns"""
        if len(time_series) < 12:
            return False
        
        # Simple check: if we have monthly data, check for repeating patterns
        monthly = time_series.dt.month.value_counts()
        return len(monthly) >= 10  # Most months represented
    
    # ==================== PHASE 2.5: Dataset Intelligence Summary ====================
    
    def analyze_complex_relationships(self) -> Dict[str, any]:
        """
        Analyze complex multi-level relationships between columns
        Similar to groupby aggregations and multi-table joins
        """
        relationships = []
        
        # 1. Group by categorical columns and aggregate numeric columns
        for cat_col in self.categorical_cols[:3]:
            for num_col in self.numeric_cols[:3]:
                try:
                    grouped = self.df.groupby(cat_col)[num_col].agg(['mean', 'sum', 'count', 'min', 'max']).reset_index()
                    
                    # Sort by count to get top categories
                    grouped = grouped.nlargest(10, 'count')
                    
                    relationships.append({
                        'type': 'categorical_numeric_aggregation',
                        'groupby_column': cat_col,
                        'aggregate_column': num_col,
                        'top_groups': grouped.to_dict('records'),
                        'insight': f"Top 10 {cat_col} categories by {num_col} performance"
                    })
                except Exception as e:
                    continue
        
        # 2. Multi-column grouping (like ID grouping with multiple aggregations)
        if len(self.categorical_cols) >= 1 and len(self.numeric_cols) >= 2:
            try:
                cat_col = self.categorical_cols[0]
                num_cols = self.numeric_cols[:2]
                
                multi_agg = self.df.groupby(cat_col).agg({
                    num_cols[0]: ['max', 'min', 'mean'],
                    num_cols[1]: ['sum', 'count']
                }).reset_index()
                
                multi_agg.columns = ['_'.join(col).strip('_') for col in multi_agg.columns.values]
                
                relationships.append({
                    'type': 'multi_column_aggregation',
                    'groupby_column': cat_col,
                    'aggregated_columns': num_cols,
                    'sample_data': multi_agg.head(10).to_dict('records'),
                    'insight': f"Multi-dimensional aggregation of {num_cols} grouped by {cat_col}"
                })
            except Exception as e:
                pass
        
        # 3. Cross-tabulation analysis
        if len(self.categorical_cols) >= 2:
            try:
                cat1, cat2 = self.categorical_cols[0], self.categorical_cols[1]
                crosstab = pd.crosstab(self.df[cat1], self.df[cat2])
                
                relationships.append({
                    'type': 'cross_tabulation',
                    'columns': [cat1, cat2],
                    'dimensions': crosstab.shape,
                    'insight': f"Relationship matrix between {cat1} and {cat2}"
                })
            except Exception as e:
                pass
        
        # 4. Conditional aggregations (like status-based filtering then grouping)
        if 'status' in self.column_lower or 'state' in self.column_lower:
            try:
                status_col = None
                for col in self.columns:
                    if 'status' in col.lower() or 'state' in col.lower():
                        status_col = col
                        break
                
                if status_col and len(self.numeric_cols) >= 1:
                    unique_statuses = self.df[status_col].unique()[:5]
                    
                    conditional_aggs = []
                    for status in unique_statuses:
                        filtered = self.df[self.df[status_col] == status]
                        if len(filtered) > 0 and len(self.categorical_cols) > 0:
                            cat_col = self.categorical_cols[0]
                            num_col = self.numeric_cols[0]
                            
                            agg = filtered.groupby(cat_col)[num_col].agg(['min', 'max', 'mean']).reset_index()
                            
                            conditional_aggs.append({
                                'status': status,
                                'count': len(filtered),
                                'top_groups': agg.head(5).to_dict('records')
                            })
                    
                    relationships.append({
                        'type': 'conditional_aggregation',
                        'condition_column': status_col,
                        'results': conditional_aggs,
                        'insight': f"Performance metrics segmented by {status_col}"
                    })
            except Exception as e:
                pass
        
        # 5. Time-based aggregations if datetime exists
        if self.datetime_cols:
            try:
                time_col = self.datetime_cols[0]
                temp_df = self.df.copy()
                temp_df[time_col] = pd.to_datetime(temp_df[time_col], errors='coerce')
                temp_df = temp_df.dropna(subset=[time_col])
                
                if len(temp_df) > 0 and len(self.numeric_cols) > 0:
                    temp_df['year_month'] = temp_df[time_col].dt.to_period('M')
                    
                    time_agg = temp_df.groupby('year_month')[self.numeric_cols[0]].agg(['sum', 'mean', 'count']).reset_index()
                    
                    relationships.append({
                        'type': 'time_series_aggregation',
                        'time_column': time_col,
                        'metric': self.numeric_cols[0],
                        'monthly_trends': time_agg.tail(12).to_dict('records'),
                        'insight': f"Monthly trends for {self.numeric_cols[0]}"
                    })
            except Exception as e:
                pass
        
        return {
            'total_relationships': len(relationships),
            'relationships': relationships
        }
    
    def visualize_categorical_distributions(self) -> List[Dict]:
        """
        Create visualizations for all object-type columns
        Similar to display_object function
        """
        visualizations = []
        
        for col in self.categorical_cols:
            try:
                value_counts = self.df[col].value_counts().head(15)
                
                import plotly.graph_objects as go
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=value_counts.index,
                    y=value_counts.values,
                    text=value_counts.values,
                    textposition='outside',
                    marker_color='lightblue'
                ))
                
                fig.update_layout(
                    title=f"📊 Distribution Analysis: {col} | Total Categories: {len(value_counts)} | Top 15 Shown",
                    xaxis_title=col,
                    yaxis_title="Frequency Count",
                    xaxis_tickangle=-45
                )
                
                visualizations.append({
                    'column': col,
                    'figure': fig,
                    'unique_values': self.df[col].nunique(),
                    'top_value': value_counts.index[0],
                    'top_value_count': value_counts.values[0],
                    'top_value_percentage': (value_counts.values[0] / len(self.df)) * 100
                })
            except Exception as e:
                continue
        
        return visualizations
    
    def generate_intelligence_summary(self) -> Dict[str, any]:
        """
        Generate comprehensive dataset intelligence summary
        
        Returns:
            Complete business context understanding
        """
        business_type = self.infer_business_type()
        entities = self.detect_entities()
        kpis = self.identify_kpis()
        time_dim = self.detect_time_dimension()
        
        return {
            'business_context': {
                'type': business_type['business_type'],
                'confidence': business_type['confidence'],
                'description': business_type['description'],
                'indicators': business_type['indicators'],
            },
            'entities': entities,
            'kpis': kpis,
            'time_dimension': time_dim,
            'dataset_size': {
                'rows': len(self.df),
                'columns': len(self.df.columns),
                'numeric_columns': len(self.numeric_cols),
                'categorical_columns': len(self.categorical_cols),
            },
            'data_quality': {
                'completeness': self._calculate_completeness(),
                'uniqueness': self._calculate_uniqueness(),
            },
            'recommendations': self._generate_recommendations(),
        }
    
    def _calculate_completeness(self) -> float:
        """Calculate data completeness percentage"""
        total_cells = len(self.df) * len(self.df.columns)
        non_null_cells = self.df.notna().sum().sum()
        return (non_null_cells / total_cells) * 100
    
    def _calculate_uniqueness(self) -> Dict[str, float]:
        """Calculate uniqueness for each column"""
        return {
            col: (self.df[col].nunique() / len(self.df)) * 100
            for col in self.df.columns
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate analysis recommendations"""
        recommendations = []
        
        # Check for missing data
        missing_pct = (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 10:
            recommendations.append(f"⚠️ {missing_pct:.1f}% missing data - consider imputation or removal")
        
        # Check for duplicates
        dup_pct = (self.df.duplicated().sum() / len(self.df)) * 100
        if dup_pct > 5:
            recommendations.append(f"⚠️ {dup_pct:.1f}% duplicate rows - consider deduplication")
        
        # Check for imbalanced data
        if self.categorical_cols:
            for col in self.categorical_cols[:3]:
                value_counts = self.df[col].value_counts()
                if len(value_counts) > 0:
                    imbalance = value_counts.iloc[0] / value_counts.iloc[-1]
                    if imbalance > 10:
                        recommendations.append(f"⚠️ Column '{col}' is highly imbalanced")
        
        # Check for outliers
        if self.numeric_cols:
            recommendations.append("✓ Consider outlier analysis for numeric columns")
        
        # Time series recommendation
        if self.datetime_cols:
            recommendations.append("✓ Time-series analysis recommended")
        
        return recommendations
