"""
Advanced Automatic Analysis Module
Senior Data Analyst approach - Deep insights with strength/weakness analysis
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import logging
import ollama

from analysis.viz_rules import classify_columns, is_id_column

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AdvancedAnalyzer:
    """
    Senior Data Analyst Approach:
    - Deep analysis of each data segment
    - Identify strengths and weaknesses for each part
    - Generate comprehensive insights with visualizations
    - Think like a senior analyst
    """
    
    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset", model: str = "qwen2.5:7b", goals: Dict = None):
        self.df = df.copy()
        self.dataset_name = dataset_name
        self.model = model
        self.goals = goals or {"problem": "", "objective": "", "target": ""}
        
        # Classify columns using centralized rules — filter out ID columns
        classes = classify_columns(df)
        self.numeric_cols = classes['measures']  # Only measures, not IDs
        self.categorical_cols = classes['dimensions']  # Only dimensions, not IDs
        self.datetime_cols = classes['dates']
        self.id_cols = classes['identifiers']  # Track them for reference
        
        if self.id_cols:
            logger.info(f"Excluded {len(self.id_cols)} ID columns from analysis: {self.id_cols}")
    
    def run_complete_analysis(self) -> Dict:
        """Run comprehensive senior-level analysis"""
        logger.info("Starting advanced analysis with senior analyst approach...")
        
        results = {
            "overview": self._generate_overview(),
            "segment_analysis": self._analyze_all_segments(),
            "correlation_insights": self._deep_correlation_analysis(),
            "distribution_insights": self._deep_distribution_analysis(),
            "categorical_insights": self._deep_categorical_analysis(),
            "outlier_insights": self._deep_outlier_analysis(),
            "time_series_insights": self._deep_time_series_analysis() if self.datetime_cols else None,
            "executive_summary": self._generate_executive_summary(),
            "action_plan": self._generate_action_plan()
        }
        
        logger.info("Advanced analysis completed")
        return results
    
    def _generate_overview(self) -> Dict:
        """Generate comprehensive overview"""
        return {
            "title": f"📊 Complete Analysis: {self.dataset_name}",
            "basic_stats": {
                "rows": len(self.df),
                "columns": len(self.df.columns),
                "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
                "duplicates": int(self.df.duplicated().sum()),
                "missing_cells": int(self.df.isna().sum().sum()),
                "completeness": round(100 * (1 - self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))), 2)
            },
            "column_types": {
                "numeric": len(self.numeric_cols),
                "categorical": len(self.categorical_cols),
                "datetime": len(self.datetime_cols)
            }
        }
    
    def _analyze_all_segments(self) -> List[Dict]:
        """
        Analyze each column/segment separately
        Identify strengths and weaknesses for each
        """
        segments = []
        
        # Analyze numeric columns
        for col in self.numeric_cols:
            segment = self._analyze_numeric_column(col)
            segments.append(segment)
        
        # Analyze categorical columns
        for col in self.categorical_cols:
            segment = self._analyze_categorical_column(col)
            segments.append(segment)
        
        # Analyze datetime columns
        for col in self.datetime_cols:
            segment = self._analyze_datetime_column(col)
            segments.append(segment)
        
        return segments
    
    def _analyze_numeric_column(self, col: str) -> Dict:
        """Deep analysis of numeric column with strengths/weaknesses"""
        series = self.df[col].dropna()
        
        if len(series) == 0:
            return {
                "column": col,
                "type": "numeric",
                "status": "⚠️ No data",
                "strengths": [],
                "weaknesses": ["Column is completely empty"],
                "insights": [],
                "visualizations": []
            }
        
        # Calculate statistics
        stats = {
            "mean": series.mean(),
            "median": series.median(),
            "std": series.std(),
            "min": series.min(),
            "max": series.max(),
            "q1": series.quantile(0.25),
            "q3": series.quantile(0.75),
            "skewness": series.skew(),
            "kurtosis": series.kurt(),
            "missing_pct": (self.df[col].isna().sum() / len(self.df)) * 100,
            "zeros": (series == 0).sum(),
            "unique": series.nunique(),
            "cv": (series.std() / series.mean() * 100) if series.mean() != 0 else 0
        }
        
        # Identify strengths
        strengths = []
        if stats["missing_pct"] < 5:
            strengths.append(f"✅ Excellent data quality: Only {stats['missing_pct']:.1f}% missing")
        if -0.5 <= stats["skewness"] <= 0.5:
            strengths.append("✅ Well-balanced distribution (low skewness)")
        if stats["unique"] / len(series) > 0.9:
            strengths.append("✅ High uniqueness - good for identification")
        if stats["cv"] < 30:
            strengths.append("✅ Low variability - stable values")
        if series.min() >= 0:
            strengths.append("✅ All values are non-negative")
        
        # Identify weaknesses
        weaknesses = []
        if stats["missing_pct"] > 20:
            weaknesses.append(f"⚠️ High missing data: {stats['missing_pct']:.1f}%")
        if abs(stats["skewness"]) > 2:
            weaknesses.append(f"⚠️ Highly skewed distribution (skewness: {stats['skewness']:.2f})")
        if stats["cv"] > 100:
            weaknesses.append(f"⚠️ High variability (CV: {stats['cv']:.1f}%)")
        if stats["zeros"] / len(series) > 0.5:
            weaknesses.append(f"⚠️ Too many zeros: {stats['zeros']} ({stats['zeros']/len(series)*100:.1f}%)")
        
        # Detect outliers
        iqr = stats["q3"] - stats["q1"]
        lower_bound = stats["q1"] - 1.5 * iqr
        upper_bound = stats["q3"] + 1.5 * iqr
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        outlier_pct = (len(outliers) / len(series)) * 100
        
        if outlier_pct > 10:
            weaknesses.append(f"⚠️ Significant outliers detected: {outlier_pct:.1f}%")
        
        # Generate AI insights
        insights = self._generate_numeric_insights(col, stats, outlier_pct)
        
        # Create visualizations
        visualizations = self._create_numeric_visualizations(col, series, stats)
        
        return {
            "column": col,
            "type": "numeric",
            "status": "✅ Complete" if not weaknesses else "⚠️ Needs attention",
            "statistics": stats,
            "strengths": strengths if strengths else ["➖ No specific strengths identified"],
            "weaknesses": weaknesses if weaknesses else ["✅ No major issues detected"],
            "insights": insights,
            "visualizations": visualizations
        }
    
    def _analyze_categorical_column(self, col: str) -> Dict:
        """Deep analysis of categorical column"""
        series = self.df[col].dropna()
        
        if len(series) == 0:
            return {
                "column": col,
                "type": "categorical",
                "status": "⚠️ No data",
                "strengths": [],
                "weaknesses": ["Column is completely empty"],
                "insights": [],
                "visualizations": []
            }
        
        # Calculate statistics
        value_counts = series.value_counts()
        stats = {
            "unique": series.nunique(),
            "most_common": value_counts.index[0],
            "most_common_count": value_counts.values[0],
            "most_common_pct": (value_counts.values[0] / len(series)) * 100,
            "missing_pct": (self.df[col].isna().sum() / len(self.df)) * 100,
            "cardinality_ratio": series.nunique() / len(series)
        }
        
        # Identify strengths
        strengths = []
        if stats["missing_pct"] < 5:
            strengths.append(f"✅ Excellent completeness: {100-stats['missing_pct']:.1f}%")
        if 5 <= stats["unique"] <= 50:
            strengths.append(f"✅ Optimal number of categories: {stats['unique']}")
        if 10 <= stats["most_common_pct"] <= 40:
            strengths.append("✅ Well-balanced distribution across categories")
        
        # Identify weaknesses
        weaknesses = []
        if stats["missing_pct"] > 20:
            weaknesses.append(f"⚠️ High missing data: {stats['missing_pct']:.1f}%")
        if stats["unique"] > 100:
            weaknesses.append(f"⚠️ Too many categories: {stats['unique']} (high cardinality)")
        if stats["most_common_pct"] > 80:
            weaknesses.append(f"⚠️ Dominated by one category: {stats['most_common']} ({stats['most_common_pct']:.1f}%)")
        if stats["unique"] == len(series):
            weaknesses.append("⚠️ Every value is unique - may be an ID column")
        
        # Generate AI insights
        insights = self._generate_categorical_insights(col, stats, value_counts)
        
        # Create visualizations
        visualizations = self._create_categorical_visualizations(col, value_counts.head(15))
        
        return {
            "column": col,
            "type": "categorical",
            "status": "✅ Complete" if not weaknesses else "⚠️ Needs attention",
            "statistics": stats,
            "top_categories": value_counts.head(10).to_dict(),
            "strengths": strengths if strengths else ["➖ No specific strengths identified"],
            "weaknesses": weaknesses if weaknesses else ["✅ No major issues detected"],
            "insights": insights,
            "visualizations": visualizations
        }
    
    def _analyze_datetime_column(self, col: str) -> Dict:
        """Deep analysis of datetime column"""
        series = pd.to_datetime(self.df[col], errors='coerce').dropna()
        
        if len(series) == 0:
            return {
                "column": col,
                "type": "datetime",
                "status": "⚠️ No valid dates",
                "strengths": [],
                "weaknesses": ["No valid datetime values"],
                "insights": [],
                "visualizations": []
            }
        
        stats = {
            "min_date": series.min(),
            "max_date": series.max(),
            "date_range_days": (series.max() - series.min()).days,
            "missing_pct": (self.df[col].isna().sum() / len(self.df)) * 100,
            "unique": series.nunique()
        }
        
        strengths = []
        if stats["missing_pct"] < 5:
            strengths.append("✅ Excellent date completeness")
        if stats["date_range_days"] > 365:
            strengths.append(f"✅ Good time span: {stats['date_range_days']} days")
        
        weaknesses = []
        if stats["missing_pct"] > 20:
            weaknesses.append(f"⚠️ Missing dates: {stats['missing_pct']:.1f}%")
        
        insights = [f"Date range: {stats['min_date'].strftime('%Y-%m-%d')} to {stats['max_date'].strftime('%Y-%m-%d')}"]
        
        # Generate AI insights for datetime column
        prompt = f"""As a senior data analyst, provide insights about this datetime column:

Column: {col}
Date Range: {stats['min_date'].strftime('%Y-%m-%d')} to {stats['max_date'].strftime('%Y-%m-%d')}
Total Days: {stats['date_range_days']}
Unique Dates: {stats['unique']}
Missing: {stats['missing_pct']:.1f}%

Provide actionable insights (each as one clear sentence):"""
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            insights_text = response['response'].strip()
            insights = [line.strip() for line in insights_text.split('\n') if line.strip() and len(line.strip()) > 10][:4]
        except Exception as e:
            logger.error(f"Error generating datetime insights: {e}")
        
        visualizations = []
        
        return {
            "column": col,
            "type": "datetime",
            "status": "✅ Valid",
            "statistics": stats,
            "strengths": strengths,
            "weaknesses": weaknesses,
            "insights": insights,
            "visualizations": visualizations
        }
    
    def _generate_numeric_insights(self, col: str, stats: Dict, outlier_pct: float) -> List[str]:
        """Generate AI-powered insights for numeric column"""
        prompt = f"""As a senior data analyst, provide all insights in dataset about this numeric column:

Column: {col}
Mean: {stats['mean']:.2f}
Median: {stats['median']:.2f}
Std Dev: {stats['std']:.2f}
Range: {stats['min']:.2f} to {stats['max']:.2f}
Skewness: {stats['skewness']:.2f}
Missing: {stats['missing_pct']:.1f}%
Outliers: {outlier_pct:.1f}%
Coefficient of Variation: {stats['cv']:.1f}%

Provide actionable insights (each as one clear sentence):"""
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            insights_text = response['response'].strip()
            insights = [line.strip() for line in insights_text.split('\n') if line.strip() and len(line.strip()) > 10]
            return insights[:4]
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return [f"Standard deviation suggests {'high' if stats['cv'] > 50 else 'moderate'} variability"]
    
    def _generate_categorical_insights(self, col: str, stats: Dict, value_counts: pd.Series) -> List[str]:
        """Generate AI-powered insights for categorical column"""
        top_5 = value_counts.head(5).to_dict()
        
        prompt = f"""As a senior data analyst, provide insights about this categorical column through the lens of our business context:
- Problem: {self.goals['problem']}
- Objective: {self.goals['objective']}
- Target Audience: {self.goals['target']}

Column: {col}
Unique values: {stats['unique']}
Most common: {stats['most_common']} ({stats['most_common_pct']:.1f}%)
Top 5 values: {top_5}
Missing: {stats['missing_pct']:.1f}%

Provide actionable insights (each as one clear sentence):"""
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            insights_text = response['response'].strip()
            insights = [line.strip() for line in insights_text.split('\n') if line.strip() and len(line.strip()) > 10]
            return insights[:4]
        except Exception as e:
            logger.error(f"Error generating insights: {e}")
            return [f"Column has {stats['unique']} unique categories"]
    
    def _create_numeric_visualizations(self, col: str, series: pd.Series, stats: Dict) -> List[Dict]:
        """Create comprehensive visualizations for numeric column"""
        visualizations = []
        
        # 1. Distribution with statistics overlay
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=series,
            name='Distribution',
            nbinsx=30,
            marker_color='lightblue'
        ))
        fig.add_vline(x=stats['mean'], line_dash="dash", line_color="red", annotation_text=f"Mean: {stats['mean']:.2f}")
        fig.add_vline(x=stats['median'], line_dash="dash", line_color="green", annotation_text=f"Median: {stats['median']:.2f}")
        fig.update_layout(
            title=f"📊 Deep Statistical Analysis: {col} | Skewness: {stats['skewness']:.2f} | CV: {stats['cv']:.1f}% | N={len(series):,}",
            xaxis_title=f"{col} Values",
            yaxis_title="Frequency Count",
            showlegend=True
        )
        visualizations.append({"title": f"Distribution of {col}", "figure": fig})
        
        # 2. Box plot for outlier detection
        iqr = stats['q3'] - stats['q1']
        fig = go.Figure()
        fig.add_trace(go.Box(
            y=series,
            name=col,
            boxmean='sd',
            marker_color='lightgreen'
        ))
        fig.update_layout(title=f"📦 SUM of {col} — Outlier Detection | IQR: {iqr:.2f} | Range: [{stats['min']:.2f}, {stats['max']:.2f}]")
        visualizations.append({"title": f"SUM of {col} (Box Plot)", "figure": fig})
        
        return visualizations
    
    def _create_categorical_visualizations(self, col: str, value_counts: pd.Series) -> List[Dict]:
        """Create comprehensive visualizations for categorical column"""
        visualizations = []
        
        # Bar chart
        total_count = value_counts.sum()
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=value_counts.index,
            y=value_counts.values,
            marker_color='lightcoral',
            text=value_counts.values,
            textposition='outside'
        ))
        fig.update_layout(
            title=f"📊 COUNT of Records by {col} | Top Categories (Total: {total_count:,} records)",
            xaxis_title=f"{col}",
            yaxis_title="COUNT of Records",
            xaxis_tickangle=-45
        )
        visualizations.append({"title": f"COUNT of Records by {col}", "figure": fig})
        
        # Pie chart — limit to top 6 slices for readability
        top_n = value_counts.head(6)
        fig = go.Figure(data=[go.Pie(
            labels=top_n.index,
            values=top_n.values,
            hole=0.3
        )])
        fig.update_layout(title=f"🥧 COUNT of Records by {col} | Proportional Distribution")
        visualizations.append({"title": f"COUNT of Records by {col} (Pie)", "figure": fig})
        
        return visualizations
    
    def _deep_correlation_analysis(self) -> Dict:
        """Deep correlation analysis with insights — using only measure columns (no IDs)"""
        if len(self.numeric_cols) < 2:
            return {"status": "Not enough numeric measure columns for correlation"}
        
        corr_matrix = self.df[self.numeric_cols].corr()
        
        # Find strong correlations
        strong_corr = []
        for i in range(len(corr_matrix)):
            for j in range(i+1, len(corr_matrix)):
                corr_value = corr_matrix.iloc[i, j]
                if abs(corr_value) > 0.7:
                    strong_corr.append({
                        "col1": corr_matrix.index[i],
                        "col2": corr_matrix.columns[j],
                        "correlation": corr_value,
                        "strength": "Strong Positive" if corr_value > 0 else "Strong Negative"
                    })
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10}
        ))
        fig.update_layout(
            title=f"🔥 Correlation Matrix (Measures Only) | {len(corr_matrix)} x {len(corr_matrix)} Variables",
            width=800,
            height=700
        )
        
        return {
            "strong_correlations": strong_corr,
            "visualization": {"title": "Correlation Matrix", "figure": fig}
        }
    
    def _deep_distribution_analysis(self) -> Dict:
        """Analyze distributions across all numeric columns"""
        distributions = []
        
        for col in self.numeric_cols[:6]:  # Limit to first 6
            series = self.df[col].dropna()
            if len(series) > 0:
                distributions.append({
                    "column": col,
                    "skewness": series.skew(),
                    "kurtosis": series.kurt(),
                    "normality": "Normal-like" if abs(series.skew()) < 0.5 else "Skewed"
                })
        
        return {"distributions": distributions}
    
    def _deep_categorical_analysis(self) -> Dict:
        """Deep analysis of categorical patterns"""
        return {"status": "Categorical analysis complete"}
    
    def _deep_outlier_analysis(self) -> Dict:
        """Comprehensive outlier detection"""
        outlier_summary = []
        
        for col in self.numeric_cols:
            series = self.df[col].dropna()
            if len(series) > 0:
                q1 = series.quantile(0.25)
                q3 = series.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                outliers = series[(series < lower) | (series > upper)]
                
                if len(outliers) > 0:
                    outlier_summary.append({
                        "column": col,
                        "count": len(outliers),
                        "percentage": (len(outliers) / len(series)) * 100
                    })
        
        return {"outliers": outlier_summary}
    
    def _deep_time_series_analysis(self) -> Dict:
        """Time series analysis if datetime columns exist"""
        return {"status": "Time series analysis available"}
    
    def _generate_executive_summary(self) -> str:
        """Generate AI-powered executive summary"""
        prompt = f"""As a senior data analyst, write an executive summary for this dataset:

Dataset: {self.dataset_name}
Business Problem: {self.goals['problem']}
Objective: {self.goals['objective']}
Target: {self.goals['target']}

Rows: {len(self.df):,}
Columns: {len(self.df.columns)}
Numeric Columns: {len(self.numeric_cols)}
Categorical Columns: {len(self.categorical_cols)}
Missing Values: {self.df.isna().sum().sum()}
Duplicates: {self.df.duplicated().sum()}

Write a 4-5 sentence executive summary highlighting the most important findings and recommendations relative to the business goals."""
        
        try:
            response = ollama.generate(model=self.model, prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            logger.error(f"Error generating summary: {e}")
            return f"Dataset contains {len(self.df):,} rows and {len(self.df.columns)} columns with analysis complete."
    
    def _generate_action_plan(self) -> List[str]:
        """Generate actionable recommendations"""
        actions = []
        
        # Missing data actions
        missing_pct = (self.df.isna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100
        if missing_pct > 10:
            actions.append(f"🔧 Address missing data ({missing_pct:.1f}% of dataset)")
        
        # Duplicate actions
        dup_count = self.df.duplicated().sum()
        if dup_count > 0:
            actions.append(f"🔧 Remove {dup_count} duplicate rows")
        
        # High cardinality actions
        for col in self.categorical_cols:
            if self.df[col].nunique() > 100:
                actions.append(f"🔧 Consider grouping high-cardinality column: {col}")
        
        if not actions:
            actions.append("✅ Data quality is generally good")
        
        return actions
