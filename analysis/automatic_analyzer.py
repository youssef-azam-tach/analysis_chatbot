"""
Automatic Analysis Module
Performs complete automatic analysis with insights and visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Tuple
import logging
import ollama
from analysis.eda import EDAAnalyzer
from analysis.visualization import Visualizer
from analysis.business_intelligence import BusinessIntelligence

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutomaticAnalyzer:
    """
    Perform comprehensive automatic analysis on datasets
    Generates insights, visualizations, and recommendations without user interaction
    """
    
    def __init__(self, df: pd.DataFrame, dataset_name: str = "dataset", model: str = "qwen2.5:7b"):
        """
        Initialize automatic analyzer
        
        Args:
            df: DataFrame to analyze
            dataset_name: Name of the dataset
            model: LLM model for insights generation
        """
        self.df = df
        self.dataset_name = dataset_name
        self.model = model
        self.analysis_results = {}
        self.graphs = []
        self.insights = []
        
        # Initialize analyzers
        self.eda_analyzer = EDAAnalyzer(df)
        self.visualizer = Visualizer(df)
        self.bi_analyzer = BusinessIntelligence(df)
    
    def run_complete_analysis(self) -> Dict:
        """
        Run complete automatic analysis
        Returns comprehensive analysis results with graphs and insights
        """
        logger.info(f"Starting automatic analysis for {self.dataset_name}")
        
        results = {
            "dataset_name": self.dataset_name,
            "basic_info": self._analyze_basic_info(),
            "business_context": self._analyze_business_context(),
            "data_patterns": self._analyze_patterns(),
            "key_insights": self._generate_insights(),
            "visualizations": self._generate_visualizations(),
            "recommendations": self._generate_recommendations(),
            "summary": self._generate_summary()
        }
        
        self.analysis_results = results
        logger.info("Automatic analysis completed")
        
        return results
    
    def _analyze_basic_info(self) -> Dict:
        """Analyze basic dataset information"""
        logger.info("Analyzing basic information...")
        
        return {
            "rows": len(self.df),
            "columns": len(self.df.columns),
            "memory_mb": round(self.df.memory_usage(deep=True).sum() / 1024**2, 2),
            "duplicates": int(self.df.duplicated().sum()),
            "missing_values": int(self.df.isna().sum().sum()),
            "completeness": round((self.df.notna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100, 2),
            "numeric_columns": len(self.df.select_dtypes(include=[np.number]).columns),
            "categorical_columns": len(self.df.select_dtypes(include=['object', 'category']).columns),
            "datetime_columns": len(self.df.select_dtypes(include=['datetime64']).columns)
        }
    
    def _analyze_business_context(self) -> Dict:
        """Analyze business context using Business Intelligence module"""
        logger.info("Analyzing business context...")
        
        try:
            business_type_info = self.bi_analyzer.infer_business_type()
            entities = self.bi_analyzer.identify_entities()
            kpis = self.bi_analyzer.suggest_kpis()
            time_dim = self.bi_analyzer.identify_time_dimension()
            
            return {
                "business_type": business_type_info.get("business_type", "unknown"),
                "confidence": business_type_info.get("confidence", 0),
                "description": business_type_info.get("description", ""),
                "entities": entities,
                "kpis": kpis,
                "time_dimension": time_dim
            }
        except Exception as e:
            logger.error(f"Error in business context analysis: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_patterns(self) -> Dict:
        """Analyze data patterns and distributions"""
        logger.info("Analyzing data patterns...")
        
        patterns = {
            "correlations": [],
            "outliers": {},
            "distributions": {},
            "trends": []
        }
        
        # Correlation analysis
        corr_matrix = self.eda_analyzer.get_correlation_matrix()
        if not corr_matrix.empty:
            high_corr = self.eda_analyzer.get_high_correlations(threshold=0.7)
            patterns["correlations"] = high_corr
        
        # Outlier detection
        patterns["outliers"] = self.eda_analyzer.detect_outliers(method="iqr")
        
        # Distribution analysis for numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:5]
        for col in numeric_cols:
            patterns["distributions"][col] = {
                "mean": float(self.df[col].mean()),
                "median": float(self.df[col].median()),
                "std": float(self.df[col].std()),
                "skewness": float(self.df[col].skew()),
                "kurtosis": float(self.df[col].kurtosis())
            }
        
        return patterns
    
    def _generate_insights(self) -> List[Dict]:
        """Generate insights using LLM"""
        logger.info("Generating insights with LLM...")
        
        # Prepare data summary for LLM
        data_summary = self._prepare_data_summary()
        
        prompt = f"""You are a data analyst expert. Analyze this dataset and provide key insights.

{data_summary}

Task: Provide 5-7 key insights from this data. For each insight:
1. What pattern or finding did you discover?
2. Why is it important?
3. What action should be taken?

Format as JSON:
{{
  "insights": [
    {{
      "title": "Brief insight title",
      "finding": "What you discovered",
      "importance": "Why it matters",
      "action": "Recommended action",
      "priority": "high|medium|low"
    }}
  ]
}}"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            import re
            
            content = response['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                insights = result.get("insights", [])
                self.insights = insights
                return insights
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating insights: {str(e)}")
            return []
    
    def _generate_visualizations(self) -> List[Dict]:
        """Generate automatic visualizations"""
        logger.info("Generating visualizations...")
        
        visualizations = []
        
        # 1. Correlation heatmap (if numeric columns exist)
        if len(self.df.select_dtypes(include=[np.number]).columns) > 1:
            try:
                fig = self.visualizer.correlation_heatmap()
                if fig:
                    visualizations.append({
                        "type": "correlation_heatmap",
                        "title": "Correlation Matrix",
                        "figure": fig,
                        "insight": "Shows relationships between numeric variables"
                    })
            except Exception as e:
                logger.warning(f"Could not create correlation heatmap: {str(e)}")
        
        # 2. Distribution plots for top 3 numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:3]
        for col in numeric_cols:
            try:
                fig = self.visualizer.distribution_plot(col)
                visualizations.append({
                    "type": "distribution",
                    "title": f"Distribution of {col}",
                    "figure": fig,
                    "column": col
                })
            except Exception as e:
                logger.warning(f"Could not create distribution for {col}: {str(e)}")
        
        # 3. Top categories for categorical columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns[:3]
        for col in categorical_cols:
            try:
                top_10 = self.df[col].value_counts().head(10)
                if not top_10.empty:
                    fig = px.bar(
                        x=top_10.values,
                        y=top_10.index,
                        orientation='h',
                        title=f"Top 10 {col}",
                        labels={'x': 'Count', 'y': col}
                    )
                    visualizations.append({
                        "type": "categorical_bar",
                        "title": f"Top 10 {col}",
                        "figure": fig,
                        "column": col
                    })
            except Exception as e:
                logger.warning(f"Could not create bar chart for {col}: {str(e)}")
        
        # 4. Missing values visualization
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            try:
                fig = px.bar(
                    x=missing[missing > 0].values,
                    y=missing[missing > 0].index,
                    orientation='h',
                    title="Missing Values by Column",
                    labels={'x': 'Missing Count', 'y': 'Column'}
                )
                visualizations.append({
                    "type": "missing_values",
                    "title": "Missing Values Analysis",
                    "figure": fig
                })
            except Exception as e:
                logger.warning(f"Could not create missing values chart: {str(e)}")
        
        # 5. Outlier boxplots for numeric columns
        numeric_cols_outliers = self.df.select_dtypes(include=[np.number]).columns[:4]
        if len(numeric_cols_outliers) > 0:
            try:
                fig = make_subplots(
                    rows=1, cols=len(numeric_cols_outliers),
                    subplot_titles=[col for col in numeric_cols_outliers]
                )
                
                for idx, col in enumerate(numeric_cols_outliers, 1):
                    fig.add_trace(
                        go.Box(y=self.df[col], name=col),
                        row=1, col=idx
                    )
                
                fig.update_layout(title="Outlier Detection - Boxplots", showlegend=False)
                visualizations.append({
                    "type": "outlier_boxplots",
                    "title": "Outlier Detection",
                    "figure": fig
                })
            except Exception as e:
                logger.warning(f"Could not create outlier boxplots: {str(e)}")
        
        self.graphs = visualizations
        logger.info(f"Generated {len(visualizations)} visualizations")
        
        return visualizations
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate recommendations using LLM"""
        logger.info("Generating recommendations...")
        
        data_summary = self._prepare_data_summary()
        insights_summary = "\n".join([f"- {ins.get('title', '')}: {ins.get('finding', '')}" for ins in self.insights])
        
        prompt = f"""You are a data quality and analysis expert. Based on this data analysis, provide recommendations.

{data_summary}

Key Insights:
{insights_summary}

Task: Provide 5-7 actionable recommendations for:
1. Data quality improvements
2. Further analysis directions
3. Business actions
4. Visualization improvements

Format as JSON:
{{
  "recommendations": [
    {{
      "category": "data_quality|analysis|business|visualization",
      "title": "Recommendation title",
      "description": "Detailed recommendation",
      "priority": "high|medium|low",
      "effort": "low|medium|high"
    }}
  ]
}}"""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            import re
            
            content = response['message']['content']
            json_match = re.search(r'\{.*\}', content, re.DOTALL)
            
            if json_match:
                result = json.loads(json_match.group())
                return result.get("recommendations", [])
            else:
                return []
                
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
            return []
    
    def _generate_summary(self) -> str:
        """Generate overall summary using LLM"""
        logger.info("Generating executive summary...")
        
        data_summary = self._prepare_data_summary()
        insights_text = "\n".join([f"- {ins.get('title', '')}" for ins in self.insights])
        
        prompt = f"""You are an executive analyst. Create a brief, impactful summary of this data analysis.

{data_summary}

Key Insights:
{insights_text}

Task: Write a 2-3 paragraph executive summary that:
1. Describes what the data represents
2. Highlights the most important findings
3. Suggests key next steps

Use clear, business-friendly language."""
        
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary"
    
    def _prepare_data_summary(self) -> str:
        """Prepare comprehensive data summary for LLM"""
        summary = f"""DATASET: {self.dataset_name}

BASIC INFO:
- Rows: {len(self.df)}
- Columns: {len(self.df.columns)}
- Memory: {round(self.df.memory_usage(deep=True).sum() / 1024**2, 2)} MB
- Duplicates: {self.df.duplicated().sum()}
- Missing Values: {self.df.isna().sum().sum()}
- Completeness: {round((self.df.notna().sum().sum() / (len(self.df) * len(self.df.columns))) * 100, 2)}%

COLUMNS:
"""
        
        for col in self.df.columns[:15]:  # Limit to first 15 columns
            summary += f"- {col} ({self.df[col].dtype}): "
            summary += f"{self.df[col].notna().sum()} non-null, "
            summary += f"{self.df[col].nunique()} unique values\n"
            
            if pd.api.types.is_numeric_dtype(self.df[col]):
                summary += f"  Range: [{self.df[col].min():.2f}, {self.df[col].max():.2f}], "
                summary += f"Mean: {self.df[col].mean():.2f}\n"
        
        return summary
    
    def export_report(self, format: str = "markdown") -> str:
        """
        Export analysis report in specified format
        
        Args:
            format: 'markdown' or 'html'
        
        Returns:
            Formatted report string
        """
        if format == "markdown":
            return self._export_markdown()
        elif format == "html":
            return self._export_html()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_markdown(self) -> str:
        """Export analysis as markdown"""
        report = f"""# Automatic Analysis Report: {self.dataset_name}

## Executive Summary

{self.analysis_results.get('summary', 'No summary available')}

## Dataset Overview

- **Rows**: {self.analysis_results['basic_info']['rows']}
- **Columns**: {self.analysis_results['basic_info']['columns']}
- **Memory**: {self.analysis_results['basic_info']['memory_mb']} MB
- **Completeness**: {self.analysis_results['basic_info']['completeness']}%
- **Duplicates**: {self.analysis_results['basic_info']['duplicates']}
- **Missing Values**: {self.analysis_results['basic_info']['missing_values']}

## Business Context

**Business Type**: {self.analysis_results['business_context'].get('business_type', 'Unknown')}
**Confidence**: {self.analysis_results['business_context'].get('confidence', 0):.1%}

{self.analysis_results['business_context'].get('description', '')}

## Key Insights

"""
        
        for idx, insight in enumerate(self.analysis_results.get('key_insights', []), 1):
            report += f"""
### {idx}. {insight.get('title', 'Untitled')}

**Finding**: {insight.get('finding', '')}

**Importance**: {insight.get('importance', '')}

**Recommended Action**: {insight.get('action', '')}

**Priority**: {insight.get('priority', 'medium').upper()}

---
"""
        
        report += "\n## Recommendations\n\n"
        
        for idx, rec in enumerate(self.analysis_results.get('recommendations', []), 1):
            report += f"""
### {idx}. {rec.get('title', 'Untitled')} 
**Category**: {rec.get('category', 'general')}  
**Priority**: {rec.get('priority', 'medium').upper()}  
**Effort**: {rec.get('effort', 'medium').upper()}

{rec.get('description', '')}

---
"""
        
        return report
    
    def _export_html(self) -> str:
        """Export analysis as HTML"""
        # TODO: Implement HTML export
        return "<html><body><h1>HTML Export Coming Soon</h1></body></html>"
