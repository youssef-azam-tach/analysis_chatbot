"""
Enhanced LLM Chatbot with Graph Generation
Intelligent chatbot that can generate and send visualizations with answers
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Tuple, Optional
import logging
import ollama
import re
import json

from models.data_to_text import DataToText
from analysis.visualization import Visualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancedChatbot:
    """
    Enhanced chatbot with automatic graph generation capabilities
    Can detect when visualizations would be helpful and generate them
    """
    
    def __init__(self, df: pd.DataFrame, model: str = "qwen2.5:7b"):
        """
        Initialize enhanced chatbot
        
        Args:
            df: DataFrame to analyze
            model: LLM model name
        """
        self.df = df
        self.model = model
        self.data_context = None
        self.conversation_history = []
        self.visualizer = Visualizer(df)
        
        # Generate data context
        self._generate_data_context()
    
    def _sanitize_for_json(self, obj):
        """
        Convert numpy types to native Python types for JSON serialization
        """
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Series):
            return obj.tolist()
        elif isinstance(obj, pd.Index):
            return obj.tolist()
        elif hasattr(obj, 'dtype'):
            return str(obj)
        return obj
    
    def _generate_data_context(self):
        """Generate comprehensive data context for the chatbot"""
        try:
            data_to_text = DataToText(self.df)
            
            # Get all data representations
            column_descriptions = data_to_text.get_column_descriptions()
            summary_stats = data_to_text.get_summary_statistics_text()
            sample_rows = data_to_text.rows_to_sentences(limit=50)
            
            # Combine into context
            self.data_context = f"""
=== DATASET OVERVIEW ===
Total Rows: {len(self.df)}
Total Columns: {len(self.df.columns)}
Columns: {', '.join(self.df.columns.tolist())}
Memory Usage: {self.df.memory_usage(deep=True).sum() / 1024**2:.2f} MB

=== COLUMN DESCRIPTIONS ===
{column_descriptions}

=== SUMMARY STATISTICS ===
{summary_stats}

=== SAMPLE DATA ===
{chr(10).join(sample_rows[:20])}

=== DATA QUALITY ===
Missing Values: {self.df.isna().sum().sum()}
Duplicate Rows: {self.df.duplicated().sum()}
Completeness: {(self.df.notna().sum().sum() / (len(self.df) * len(self.df.columns)) * 100):.1f}%

=== AVAILABLE COLUMNS FOR VISUALIZATION ===
Numeric: {', '.join(self.df.select_dtypes(include=[np.number]).columns.tolist())}
Categorical: {', '.join(self.df.select_dtypes(include=['object', 'category']).columns.tolist()[:10])}
"""
            
            logger.info("✅ Enhanced data context loaded for chatbot")
            return True
        
        except Exception as e:
            logger.error(f"Error generating data context: {str(e)}")
            return False
    
    def chat(self, user_message: str, temperature: float = 0.7) -> Dict:
        """
        Enhanced chat with automatic visualization detection
        
        Args:
            user_message: User's question
            temperature: LLM temperature
            
        Returns:
            Dict with answer, graphs (if any), and metadata
        """
        try:
            # Step 1: Analyze if question needs visualization
            viz_needed = self._should_generate_visualization(user_message)
            
            # Step 2: Generate answer with visualization intent
            prompt = self._build_prompt(user_message, viz_needed)
            
            response = ollama.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful data analyst assistant."},
                    {"role": "user", "content": prompt}
                ],
                options={"temperature": temperature}
            )
            
            answer = response['message']['content']
            
            # Step 3: Generate visualizations if needed
            graphs = []
            if viz_needed:
                graphs = self._generate_visualizations_for_answer(user_message, answer)
            
            # Step 4: Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "graphs": graphs,  # Store actual graph objects
                "has_visualization": len(graphs) > 0
            })
            
            return {
                "answer": answer,
                "graphs": graphs,
                "has_visualization": len(graphs) > 0
            }
            
        except Exception as e:
            logger.error(f"Error in enhanced chat: {str(e)}")
            return {
                "answer": f"Error processing question: {str(e)}",
                "graphs": [],
                "has_visualization": False
            }
    
    def _should_generate_visualization(self, question: str) -> bool:
        """
        Determine if the question would benefit from visualization
        """
        viz_keywords = [
            'show', 'plot', 'graph', 'chart', 'visualize', 'display', 'trend',
            'distribution', 'comparison', 'compare', 'top', 'bottom', 'highest',
            'lowest', 'correlation', 'relationship', 'pattern', 'analyze', 'breakdown'
        ]
        
        question_lower = question.lower()
        
        # Check for visualization keywords
        for keyword in viz_keywords:
            if keyword in question_lower:
                return True
        
        # Check for question patterns that usually need graphs
        patterns = [
            r'how\s+many',
            r'what\s+is\s+the\s+(top|best|worst)',
            r'which\s+\w+\s+has',
            r'compare',
            r'difference\s+between',
            r'trend',
            r'over\s+time'
        ]
        
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def _build_prompt(self, question: str, needs_viz: bool) -> str:
        """Build prompt for LLM based on question and visualization need"""
        
        base_prompt = f"""{self.data_context}

=== USER QUESTION ===
{question}

=== INSTRUCTIONS ===
Answer the user's question based on the dataset provided above.
- Be specific and use actual numbers from the data
- Provide clear, actionable insights
- Format your answer in markdown
"""
        
        if needs_viz:
            base_prompt += """
- I will automatically generate relevant visualizations for your answer
- Mention in your answer that visualizations are being generated
- Suggest which columns would be good to visualize
"""
        
        return base_prompt
    
    def _generate_visualizations_for_answer(self, question: str, answer: str) -> List[Dict]:
        """
        Generate appropriate visualizations based on question and answer
        """
        graphs = []
        question_lower = question.lower()
        
        try:
            # 1. Top/Best/Worst queries -> Bar chart
            if any(word in question_lower for word in ['top', 'best', 'worst', 'highest', 'lowest']):
                graphs.extend(self._create_ranking_charts(question))
            
            # 2. Comparison queries -> Bar/Line chart
            if any(word in question_lower for word in ['compare', 'comparison', 'difference', 'vs', 'versus']):
                graphs.extend(self._create_comparison_charts(question))
            
            # 3. Trend/Time queries -> Line chart
            if any(word in question_lower for word in ['trend', 'over time', 'timeline', 'history', 'trand']):
                graphs.extend(self._create_trend_charts())
                # If no datetime columns, create alternative trend visualizations
                if len(graphs) == 0:
                    logger.info("No datetime columns found, creating alternative trend visualizations")
                    graphs.extend(self._create_ranking_charts(question))
            
            # 4. Distribution queries -> Histogram/Box plot
            if any(word in question_lower for word in ['distribution', 'spread', 'range', 'variance']):
                graphs.extend(self._create_distribution_charts(question))
            
            # 5. Correlation queries -> Scatter plot
            if any(word in question_lower for word in ['correlation', 'relationship', 'related', 'connection']):
                graphs.extend(self._create_correlation_charts())
            
            # 6. Summary/Overview -> Multiple charts
            if any(word in question_lower for word in ['summary', 'overview', 'analyze', 'analysis']):
                graphs.extend(self._create_summary_charts())
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
        
        return graphs
    
    def _create_ranking_charts(self, question: str) -> List[Dict]:
        """Create ranking/top-N charts"""
        graphs = []
        
        # Find categorical and numeric columns
        categorical_cols = self.df.select_dtypes(include=['object', 'category']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Try to create top-N chart for first categorical column
        if len(categorical_cols) > 0 and len(numeric_cols) > 0:
            cat_col = categorical_cols[0]
            num_col = numeric_cols[0]
            
            # Aggregate and get top 10
            top_data = self.df.groupby(cat_col)[num_col].sum().nlargest(10)
            
            if not top_data.empty:
                fig = px.bar(
                    x=self._sanitize_for_json(top_data.values),
                    y=self._sanitize_for_json(top_data.index),
                    orientation='h',
                    title=f"Top 10 {cat_col} by {num_col}",
                    labels={'x': str(num_col), 'y': str(cat_col)}
                )
                
                graphs.append({
                    "type": "ranking_bar",
                    "title": f"Top 10 {cat_col}",
                    "figure": fig
                })
        
        return graphs
    
    def _create_comparison_charts(self, question: str) -> List[Dict]:
        """Create comparison charts"""
        graphs = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:3]
        
        if len(numeric_cols) >= 2:
            # Create grouped bar chart
            fig = go.Figure()
            for col in numeric_cols:
                fig.add_trace(go.Bar(
                    name=str(col),
                    x=[str(col)],
                    y=[float(self.df[col].mean())]
                ))
            
            fig.update_layout(
                title="Comparison of Numeric Columns",
                barmode='group'
            )
            
            graphs.append({
                "type": "comparison_bar",
                "title": "Column Comparison",
                "figure": fig
            })
        
        return graphs
    
    def _create_trend_charts(self) -> List[Dict]:
        """Create trend/time-series charts"""
        graphs = []
        
        # Look for datetime columns
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            date_col = datetime_cols[0]
            num_col = numeric_cols[0]
            
            # Sort by date and create line chart
            df_sorted = self.df.sort_values(date_col).copy()
            # Convert to native Python types
            df_sorted[date_col] = pd.to_datetime(df_sorted[date_col])
            df_sorted[num_col] = df_sorted[num_col].astype(float)
            
            fig = px.line(
                df_sorted,
                x=date_col,
                y=num_col,
                title=f"Trend of {num_col} over time",
                markers=True
            )
            
            graphs.append({
                "type": "trend_line",
                "title": f"{num_col} Trend",
                "figure": fig
            })
        
        return graphs
    
    def _create_distribution_charts(self, question: str) -> List[Dict]:
        """Create distribution charts"""
        graphs = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns[:2]
        
        for col in numeric_cols:
            # Histogram - create a clean copy with native types
            df_clean = self.df[[col]].copy()
            df_clean[col] = df_clean[col].astype(float)
            
            fig = px.histogram(
                df_clean,
                x=col,
                nbins=30,
                title=f"Distribution of {col}",
                marginal="box"
            )
            
            graphs.append({
                "type": "histogram",
                "title": f"{col} Distribution",
                "figure": fig
            })
        
        return graphs
    
    def _create_correlation_charts(self) -> List[Dict]:
        """Create correlation charts"""
        graphs = []
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) >= 2:
            # Correlation heatmap
            corr_matrix = self.df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=self._sanitize_for_json(corr_matrix.values),
                x=self._sanitize_for_json(corr_matrix.columns),
                y=self._sanitize_for_json(corr_matrix.index),
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(title="Correlation Matrix")
            
            graphs.append({
                "type": "correlation_heatmap",
                "title": "Correlation Analysis",
                "figure": fig
            })
            
            # Scatter plot for top 2 numeric columns
            if len(numeric_cols) >= 2:
                # Create clean dataframe with native types
                df_scatter = self.df[[numeric_cols[0], numeric_cols[1]]].copy()
                df_scatter[numeric_cols[0]] = df_scatter[numeric_cols[0]].astype(float)
                df_scatter[numeric_cols[1]] = df_scatter[numeric_cols[1]].astype(float)
                
                fig = px.scatter(
                    df_scatter,
                    x=numeric_cols[0],
                    y=numeric_cols[1],
                    title=f"{numeric_cols[0]} vs {numeric_cols[1]}",
                    trendline="ols"
                )
                
                graphs.append({
                    "type": "scatter",
                    "title": "Relationship Plot",
                    "figure": fig
                })
        
        return graphs
    
    def _create_summary_charts(self) -> List[Dict]:
        """Create summary overview charts"""
        graphs = []
        
        # 1. Missing values chart
        missing = self.df.isnull().sum()
        if missing.sum() > 0:
            fig = px.bar(
                x=self._sanitize_for_json(missing[missing > 0].values),
                y=self._sanitize_for_json(missing[missing > 0].index),
                orientation='h',
                title="Missing Values by Column"
            )
            graphs.append({
                "type": "missing_values",
                "title": "Missing Values",
                "figure": fig
            })
        
        # 2. Data types distribution
        dtype_counts = self.df.dtypes.value_counts()
        fig = px.pie(
            values=self._sanitize_for_json(dtype_counts.values),
            names=[str(name) for name in dtype_counts.index],
            title="Data Types Distribution"
        )
        graphs.append({
            "type": "data_types",
            "title": "Column Types",
            "figure": fig
        })
        
        return graphs
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def set_model(self, model: str):
        """Change LLM model"""
        self.model = model
