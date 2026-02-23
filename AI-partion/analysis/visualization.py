"""
Visualization Toolkit
Create various charts and visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Optional, List


class Visualizer:
    """Create visualizations from dataframe"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def line_chart(self, x: str, y: str, title: str = "Line Chart") -> go.Figure:
        """Create line chart"""
        fig = px.line(self.df, x=x, y=y, title=title, markers=True)
        fig.update_layout(hovermode='x unified')
        return fig
    
    def bar_chart(self, x: str, y: str, title: str = "Bar Chart", orientation: str = "v") -> go.Figure:
        """Create bar chart"""
        if orientation == "h":
            fig = px.bar(self.df, y=x, x=y, title=title, orientation='h')
        else:
            fig = px.bar(self.df, x=x, y=y, title=title)
        return fig
    
    def pie_chart(self, values: str, names: str, title: str = "Pie Chart") -> go.Figure:
        """Create pie chart"""
        fig = px.pie(self.df, values=values, names=names, title=title)
        return fig
    
    def histogram(self, column: str, nbins: int = 30, title: str = "Histogram") -> go.Figure:
        """Create histogram"""
        fig = px.histogram(self.df, x=column, nbins=nbins, title=title)
        return fig
    
    def boxplot(self, y: str, x: Optional[str] = None, title: str = "Boxplot") -> go.Figure:
        """Create boxplot"""
        fig = px.box(self.df, y=y, x=x, title=title)
        return fig
    
    def scatter_plot(self, x: str, y: str, color: Optional[str] = None, size: Optional[str] = None, 
                     title: str = "Scatter Plot") -> go.Figure:
        """Create scatter plot"""
        fig = px.scatter(self.df, x=x, y=y, color=color, size=size, title=title)
        return fig
    
    def heatmap(self, data: pd.DataFrame, title: str = "Heatmap") -> go.Figure:
        """Create heatmap"""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdBu',
            zmid=0
        ))
        fig.update_layout(title=title)
        return fig
    
    def correlation_heatmap(self, title: str = "Correlation Matrix") -> go.Figure:
        """Create correlation heatmap"""
        if len(self.numeric_cols) == 0:
            return None
        
        corr_matrix = self.df[self.numeric_cols].corr()
        return self.heatmap(corr_matrix, title=title)
    
    def distribution_plot(self, column: str, title: str = "Distribution") -> go.Figure:
        """Create distribution plot with histogram and KDE"""
        fig = px.histogram(self.df, x=column, nbins=30, title=title, marginal="box")
        return fig
    
    def trend_plot(self, x: str, y: str, title: str = "Trend Analysis") -> go.Figure:
        """Create trend plot with moving average"""
        df_sorted = self.df.sort_values(x)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df_sorted[x], y=df_sorted[y], mode='lines', name='Actual'))
        
        # Add moving average if enough data points
        if len(df_sorted) > 7:
            ma = df_sorted[y].rolling(window=7).mean()
            fig.add_trace(go.Scatter(x=df_sorted[x], y=ma, mode='lines', name='7-Day MA'))
        
        fig.update_layout(title=title, hovermode='x unified')
        return fig
    
    def comparison_plot(self, columns: List[str], title: str = "Comparison") -> go.Figure:
        """Create comparison plot for multiple columns"""
        fig = go.Figure()
        
        for col in columns:
            if col in self.numeric_cols:
                fig.add_trace(go.Box(y=self.df[col], name=col))
        
        fig.update_layout(title=title, boxmode='group')
        return fig
    
    def multi_histogram(self, columns: List[str], title: str = "Multi Histogram") -> go.Figure:
        """Create multiple histograms"""
        fig = make_subplots(
            rows=len(columns), cols=1,
            subplot_titles=columns
        )
        
        for i, col in enumerate(columns, 1):
            fig.add_trace(
                go.Histogram(x=self.df[col], name=col, nbinsx=30),
                row=i, col=1
            )
        
        fig.update_layout(title=title, height=300*len(columns), showlegend=False)
        return fig
    
    def get_chart_types(self) -> List[str]:
        """Get available chart types"""
        return [
            "Line Chart",
            "Bar Chart",
            "Pie Chart",
            "Histogram",
            "Boxplot",
            "Scatter Plot",
            "Distribution",
            "Trend",
            "Comparison",
        ]
