"""
Hybrid Chatbot Module
Combines PandasAgentChatbot (accurate calculations) with visualization capabilities
Best of both worlds: accurate answers + helpful visualizations
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Optional, Union, Any
import logging
import re

from models.pandas_agent_chatbot import PandasAgentChatbot
from analysis.visualization import Visualizer
from analysis.viz_rules import (
    classify_columns, is_id_column, is_date_column,
    validate_chart_spec, format_chart_title,
    get_measures, get_dimensions, get_dates,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HybridChatbot:
    """
    Hybrid chatbot combining pandas agent with visualizations
    
    Workflow:
    1. Use PandasAgentChatbot to get accurate answer (with code execution)
    2. Detect if visualization would be helpful
    3. Generate appropriate visualizations
    4. Return answer + graphs together
    
    Benefits:
    - ✅ Accurate calculations (no hallucination)
    - ✅ Helpful visualizations
    - ✅ Code execution transparency
    - ✅ Conversational interface
    """
    
    def __init__(self, df: Union[pd.DataFrame, Dict[str, pd.DataFrame], List[pd.DataFrame]], model: str = "qwen2.5:7b"):
        """
        Initialize hybrid chatbot
        
        Args:
            df: DataFrame(s) to analyze
            model: LLM model name
        """
        # Handle df for visualization (use first one as primary or combined if needed)
        if isinstance(df, dict):
            self.df = list(df.values())[0] if df else pd.DataFrame()
            self.dfs = df
        elif isinstance(df, list):
            self.df = df[0] if df else pd.DataFrame()
            self.dfs = df
        else:
            self.df = df
            self.dfs = [df]
            
        self.model = model
        
        # Initialize components
        self.agent = PandasAgentChatbot(df, model)
        self.visualizer = Visualizer(self.df)
        
        # Conversation history (combines agent + visualizations)
        self.conversation_history = []
        
        logger.info(f"✅ HybridChatbot initialized with {len(self.dfs) if isinstance(self.dfs, (list, dict)) else 1} datasets")
    
    def chat(self, user_message: str, temperature: float = 0.0, context: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Chat with hybrid approach: agent for answer + visualizations
        
        Args:
            user_message: User's question
            temperature: LLM temperature (default 0.0 for accuracy)
            
        Returns:
            Dict with 'answer', 'graphs', 'has_visualization', 'agent_response'
        """
        try:
            # Step 1: Get accurate answer from pandas agent
            logger.info(f"Step 1: Asking pandas agent: {user_message}")
            agent_response = self.agent.ask(user_message, temperature=temperature, context=context)
            
            if not agent_response['success']:
                # Agent failed - return error without visualizations
                return {
                    "answer": agent_response['answer'],
                    "graphs": [],
                    "has_visualization": False,
                    "agent_response": agent_response,
                    "error": agent_response.get('error')
                }
            
            answer = agent_response['answer']
            
            # Step 2: Detect if visualization would be helpful
            logger.info("Step 2: Detecting if visualization needed")
            viz_needed = self._should_generate_visualization(user_message)
            
            # Step 3: Generate visualizations if needed
            graphs = []
            if viz_needed:
                logger.info("Step 3: Generating visualizations")
                graphs = self._generate_visualizations_for_question(user_message, answer)
            
            # Step 4: Store in conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            self.conversation_history.append({
                "role": "assistant",
                "content": answer,
                "graphs": graphs,
                "has_visualization": len(graphs) > 0,
                "agent_response": agent_response
            })
            
            logger.info(f"✅ Response ready: {len(graphs)} visualizations generated")
            
            return {
                "answer": answer,
                "graphs": graphs,
                "has_visualization": len(graphs) > 0,
                "agent_response": agent_response,
                "analysis_context": context or {},
                "error": None
            }
            
        except Exception as e:
            error_msg = f"Error in hybrid chat: {str(e)}"
            logger.error(error_msg)
            
            return {
                "answer": f"⚠️ Error: {str(e)}",
                "graphs": [],
                "has_visualization": False,
                "agent_response": None,
                "error": str(e)
            }
    
    def _should_generate_visualization(self, question: str) -> bool:
        """
        Determine if the question would benefit from visualization
        Uses smarter detection for data-relevant visualizations only
        
        Args:
            question: User's question
            
        Returns:
            True if visualization would be helpful and data-appropriate
        """
        question_lower = question.lower()
        
        # Questions that should NOT get visualizations (simple value queries)
        no_viz_patterns = [
            r'what\s+is\s+the\s+total',
            r'what\s+is\s+the\s+sum',
            r'what\s+is\s+the\s+average',
            r'what\s+is\s+the\s+mean',
            r'what\s+is\s+the\s+count',
            r'how\s+many\s+total',
            r'how\s+much\s+total',
            r'calculate\s+the\s+total',
            r'what\s+is\s+the\s+max',
            r'what\s+is\s+the\s+min',
        ]
        
        for pattern in no_viz_patterns:
            if re.search(pattern, question_lower):
                return False
        
        # Keywords that SHOULD trigger visualizations
        viz_keywords = [
            'show', 'plot', 'graph', 'chart', 'visualize', 'display', 'trend',
            'distribution', 'comparison', 'compare', 'top', 'bottom', 'highest',
            'lowest', 'pattern', 'breakdown', 'important', 'best', 'worst',
            'over time', 'across', 'between'
        ]
        
        # Check for visualization keywords
        for keyword in viz_keywords:
            if keyword in question_lower:
                return True
        
        # Check for question patterns that usually need graphs
        patterns = [
            r'what\s+is\s+the\s+(top|best|worst|highest|lowest|most|important)',
            r'which\s+\w+\s+(has|is|are)\s+(the\s+)?(highest|lowest|best|most|important)',
            r'compare',
            r'difference\s+between',
            r'trend\s+of',
            r'over\s+time',
            r'breakdown',
            r'distribution\s+of',
            r'correlation\s+between'
        ]
        
        for pattern in patterns:
            if re.search(pattern, question_lower):
                return True
        
        return False
    
    def _generate_visualizations_for_question(self, question: str, answer: str) -> List[Dict]:
        """
        Generate appropriate and POWERFUL visualizations based on question
        Only creates visualizations that add real value to understanding the data
        
        Args:
            question: User's question
            answer: Agent's answer
            
        Returns:
            List of powerful graph dictionaries
        """
        graphs = []
        question_lower = question.lower()
        
        try:
            # Track if we've created any visualization to avoid defaults
            created_viz = False
            
            # 1. Important/Top/Best/Worst/Highest/Lowest queries -> Bar chart (most powerful for rankings)
            if any(word in question_lower for word in ['important', 'top', 'best', 'worst', 'highest', 'lowest', 'most', 'least']):
                ranking_graphs = self._create_ranking_charts(question, answer)
                if ranking_graphs:
                    graphs.extend(ranking_graphs)
                    created_viz = True
                    logger.info(f"✅ Created {len(ranking_graphs)} ranking visualization(s)")
            
            # 2. Trend/Time/Over Time queries -> Line chart (powerful for temporal patterns)
            if any(word in question_lower for word in ['trend', 'over time', 'timeline', 'history']):
                trend_graphs = self._create_trend_charts(question)
                if trend_graphs:
                    graphs.extend(trend_graphs)
                    created_viz = True
                    logger.info(f"✅ Created {len(trend_graphs)} trend visualization(s)")
            
            # 3. Comparison queries -> Grouped Bar chart (powerful for side-by-side comparison)
            if any(word in question_lower for word in ['compare', 'comparison', 'difference', 'vs', 'versus', 'between']):
                comparison_graphs = self._create_comparison_charts(question, answer)
                if comparison_graphs:
                    graphs.extend(comparison_graphs)
                    created_viz = True
                    logger.info(f"✅ Created {len(comparison_graphs)} comparison visualization(s)")
            
            # 4. Distribution queries -> Histogram + Box plot (powerful for understanding spread)
            if any(word in question_lower for word in ['distribution', 'spread', 'range', 'variance', 'outlier']):
                dist_graphs = self._create_distribution_charts(question)
                if dist_graphs:
                    graphs.extend(dist_graphs)
                    created_viz = True
                    logger.info(f"✅ Created {len(dist_graphs)} distribution visualization(s)")
            
            # 5. Breakdown queries -> Pie/Bar chart (powerful for composition)
            if any(word in question_lower for word in ['breakdown', 'composition', 'proportion', 'percentage', 'share']):
                breakdown_graphs = self._create_breakdown_charts(question)
                if breakdown_graphs:
                    graphs.extend(breakdown_graphs)
                    created_viz = True
                    logger.info(f"✅ Created {len(breakdown_graphs)} breakdown visualization(s)")
            
            # 6. Correlation queries ONLY when explicitly asked -> Heatmap
            if any(phrase in question_lower for phrase in ['correlation', 'relationship between', 'related to', 'affect']):
                corr_graphs = self._create_correlation_charts(question)
                if corr_graphs:
                    graphs.extend(corr_graphs)
                    created_viz = True
                    logger.info(f"✅ Created {len(corr_graphs)} correlation visualization(s)")
            
            # If no specific visualization matched but question explicitly asks for visualization
            if not created_viz and any(word in question_lower for word in ['show', 'plot', 'chart', 'graph', 'visualize']):
                # Create intelligent default visualizations based on data types
                default_graphs = self._create_smart_default_charts()
                if default_graphs:
                    graphs.extend(default_graphs)
                    logger.info(f"✅ Created {len(default_graphs)} smart default visualization(s)")
            
        except Exception as e:
            logger.error(f"❌ Error generating visualizations: {str(e)}")
        
        # Limit to top 3 most powerful visualizations to avoid overwhelming
        if len(graphs) > 3:
            logger.info(f"⚠️ Limiting from {len(graphs)} to 3 most relevant visualizations")
            graphs = graphs[:3]
        
        return graphs
    
    def _sanitize_for_json(self, obj):
        """Convert numpy types to native Python types"""
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
        return obj

    def _is_id_like_column(self, series: pd.Series, col_name: str) -> bool:
        """Delegate to centralized viz_rules"""
        return is_id_column(series, col_name)

    def _numeric_measures(self) -> List[str]:
        """Get measure columns using centralized rules"""
        return get_measures(self.df)

    def _categorical_dimensions(self) -> List[str]:
        """Get dimension columns using centralized rules"""
        return get_dimensions(self.df)
    
    def _date_columns(self) -> List[str]:
        """Get date columns using centralized rules"""
        return get_dates(self.df)
    
    def _create_ranking_charts(self, question: str, answer: str) -> List[Dict]:
        """Create powerful ranking/top-N charts with validation"""
        graphs = []
        
        try:
            categorical_cols = self._categorical_dimensions()
            numeric_cols = self._numeric_measures()
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Better column detection from question
                for col in categorical_cols:
                    if col.lower() in question.lower():
                        cat_col = col
                        break
                
                for col in numeric_cols:
                    if col.lower() in question.lower():
                        num_col = col
                        break
                
                # Validate chart spec
                valid, reason = validate_chart_spec(self.df, 'bar', cat_col, num_col)
                if not valid:
                    logger.warning(f"Ranking chart blocked: {reason}")
                    return graphs
                
                # Determine if we want top or bottom
                n = 10
                if 'bottom' in question.lower() or 'worst' in question.lower() or 'lowest' in question.lower() or 'least' in question.lower():
                    top_data = self.df.groupby(cat_col)[num_col].sum().nsmallest(n)
                    title_prefix = "Bottom"
                else:
                    top_data = self.df.groupby(cat_col)[num_col].sum().nlargest(n)
                    title_prefix = "Top"
                
                if not top_data.empty and len(top_data) > 1:
                    title = f"{title_prefix} {len(top_data)} — SUM of {num_col} by {cat_col}"
                    fig = px.bar(
                        x=self._sanitize_for_json(top_data.values),
                        y=self._sanitize_for_json(top_data.index),
                        orientation='h',
                        title=title,
                        labels={'x': f'SUM of {num_col}', 'y': str(cat_col)},
                        color=self._sanitize_for_json(top_data.values),
                        color_continuous_scale='Blues'
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        height=400,
                        margin=dict(l=150)
                    )
                    
                    graphs.append({
                        "type": "ranking_bar",
                        "title": title,
                        "figure": fig
                    })
        except Exception as e:
            logger.error(f"Error creating ranking chart: {str(e)}")
        
        return graphs
    
    def _create_comparison_charts(self, question: str, answer: str) -> List[Dict]:
        """Create comparison charts with proper validation"""
        graphs = []
        
        try:
            numeric_cols = self._numeric_measures()
            categorical_cols = self._categorical_dimensions()
            
            # Try grouped comparison if we have categories
            if len(categorical_cols) > 0 and len(numeric_cols) >= 1:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                # Validate
                valid, reason = validate_chart_spec(self.df, 'bar', cat_col, num_col)
                if not valid:
                    logger.warning(f"Comparison chart blocked: {reason}")
                    return graphs
                
                top_cats = self.df[cat_col].value_counts().head(8).index
                df_filtered = self.df[self.df[cat_col].isin(top_cats)]
                
                if len(df_filtered) > 0:
                    agg_data = df_filtered.groupby(cat_col)[num_col].agg(['sum', 'mean', 'count']).reset_index()
                    
                    fig = go.Figure()
                    fig.add_trace(go.Bar(
                        name=f'SUM of {num_col}',
                        x=self._sanitize_for_json(agg_data[cat_col]),
                        y=self._sanitize_for_json(agg_data['sum']),
                        marker_color='lightblue'
                    ))
                    fig.add_trace(go.Bar(
                        name=f'AVG of {num_col}',
                        x=self._sanitize_for_json(agg_data[cat_col]),
                        y=self._sanitize_for_json(agg_data['mean']),
                        marker_color='darkblue'
                    ))
                    
                    fig.update_layout(
                        title=f"Comparison: SUM vs AVG of {num_col} by {cat_col}",
                        barmode='group',
                        height=450,
                        xaxis_tickangle=-45
                    )
                    
                    graphs.append({
                        "type": "comparison_grouped_bar",
                        "title": f"SUM vs AVG of {num_col} by {cat_col}",
                        "figure": fig
                    })
            
            # Numeric columns comparison
            elif len(numeric_cols) >= 2:
                fig = go.Figure()
                for col in numeric_cols[:4]:
                    fig.add_trace(go.Bar(
                        name=str(col),
                        x=['AVG', 'MEDIAN', 'MAX'],
                        y=[
                            float(self.df[col].mean()),
                            float(self.df[col].median()),
                            float(self.df[col].max())
                        ]
                    ))
                
                fig.update_layout(
                    title="Statistical Comparison of Business Measures",
                    barmode='group',
                    height=450
                )
                
                graphs.append({
                    "type": "comparison_stats",
                    "title": "Measures Statistical Comparison",
                    "figure": fig
                })
        except Exception as e:
            logger.error(f"Error creating comparison chart: {str(e)}")
        
        return graphs
    
    def _create_trend_charts(self, question: str) -> List[Dict]:
        """Create trend/time-series charts — ONLY with date columns on X-axis"""
        graphs = []
        
        try:
            # Use centralized date detection
            datetime_cols = self._date_columns()
            
            # Also check datetime dtype columns as fallback
            if not datetime_cols:
                for col in self.df.columns:
                    if pd.api.types.is_datetime64_any_dtype(self.df[col]):
                        if not is_id_column(self.df[col], col):
                            datetime_cols.append(col)
                # Also try date-like string columns
                if not datetime_cols:
                    for col in self.df.columns:
                        if any(word in col.lower() for word in ['date', 'time', 'year', 'month', 'day']):
                            try:
                                pd.to_datetime(self.df[col].dropna().head(20))
                                datetime_cols.append(col)
                            except:
                                pass
            
            numeric_cols = self._numeric_measures()
            
            if len(datetime_cols) > 0 and len(numeric_cols) > 0:
                date_col = datetime_cols[0]
                
                # Try to detect numeric column from question
                num_col = numeric_cols[0]
                for col in numeric_cols:
                    if col.lower() in question.lower():
                        num_col = col
                        break
                
                # Validate: line chart MUST have time on X-axis
                valid, reason = validate_chart_spec(self.df, 'line', date_col, num_col)
                if not valid:
                    logger.warning(f"Trend chart blocked: {reason}")
                    return graphs
                
                df_sorted = self.df.sort_values(date_col).copy()
                df_sorted[date_col] = pd.to_datetime(df_sorted[date_col], errors='coerce')
                df_sorted = df_sorted.dropna(subset=[date_col])
                df_sorted[num_col] = df_sorted[num_col].astype(float)
                
                title = f"SUM of {num_col} Trend Over {date_col}"
                fig = px.line(
                    df_sorted,
                    x=date_col,
                    y=num_col,
                    title=title,
                    markers=True
                )
                
                fig.update_traces(line=dict(width=3))
                fig.update_layout(
                    hovermode='x unified',
                    height=450
                )
                
                graphs.append({
                    "type": "trend_line",
                    "title": title,
                    "figure": fig
                })
            else:
                logger.info("No date columns found — cannot create line chart (line charts require time-based data)")
        except Exception as e:
            logger.error(f"Error creating trend chart: {str(e)}")
        
        return graphs
    
    def _create_breakdown_charts(self, question: str) -> List[Dict]:
        """Create breakdown/composition charts with validation"""
        graphs = []
        
        try:
            categorical_cols = self._categorical_dimensions()
            numeric_cols = self._numeric_measures()
            
            if len(categorical_cols) > 0:
                cat_col = categorical_cols[0]
                
                # Detect column from question
                for col in categorical_cols:
                    if col.lower() in question.lower():
                        cat_col = col
                        break
                
                # Check: pie chart max 6 categories, otherwise use bar
                n_unique = self.df[cat_col].nunique()
                
                if len(numeric_cols) > 0:
                    num_col = numeric_cols[0]
                    breakdown_data = self.df.groupby(cat_col)[num_col].sum().nlargest(10)
                    
                    if not breakdown_data.empty:
                        if n_unique <= 6:
                            # Pie chart for small number of categories
                            valid, reason = validate_chart_spec(self.df, 'pie', cat_col, num_col)
                            if valid:
                                fig = px.pie(
                                    values=self._sanitize_for_json(breakdown_data.values),
                                    names=self._sanitize_for_json(breakdown_data.index),
                                    title=f"SUM of {num_col} by {cat_col}",
                                    hole=0.3
                                )
                                fig.update_traces(textposition='inside', textinfo='percent+label')
                                fig.update_layout(height=450)
                                graphs.append({
                                    "type": "breakdown_pie",
                                    "title": f"SUM of {num_col} by {cat_col}",
                                    "figure": fig
                                })
                        else:
                            # Bar chart for many categories
                            valid, reason = validate_chart_spec(self.df, 'bar', cat_col, num_col)
                            if valid:
                                fig = px.bar(
                                    x=self._sanitize_for_json(breakdown_data.values),
                                    y=self._sanitize_for_json(breakdown_data.index),
                                    orientation='h',
                                    title=f"SUM of {num_col} Breakdown by {cat_col} (Top 10)",
                                    labels={'x': f'SUM of {num_col}', 'y': cat_col}
                                )
                                fig.update_layout(height=450)
                                graphs.append({
                                    "type": "breakdown_bar",
                                    "title": f"SUM of {num_col} by {cat_col}",
                                    "figure": fig
                                })
                else:
                    # Count breakdown
                    breakdown_data = self.df[cat_col].value_counts().head(10)
                    
                    if not breakdown_data.empty:
                        if n_unique <= 6:
                            fig = px.pie(
                                values=self._sanitize_for_json(breakdown_data.values),
                                names=self._sanitize_for_json(breakdown_data.index),
                                title=f"COUNT Distribution by {cat_col}",
                                hole=0.3
                            )
                            fig.update_traces(textposition='inside', textinfo='percent+label')
                        else:
                            fig = px.bar(
                                x=self._sanitize_for_json(breakdown_data.values),
                                y=self._sanitize_for_json(breakdown_data.index),
                                orientation='h',
                                title=f"COUNT Distribution by {cat_col} (Top 10)",
                                labels={'x': 'COUNT', 'y': cat_col}
                            )
                        fig.update_layout(height=450)
                        graphs.append({
                            "type": "breakdown_count",
                            "title": f"COUNT by {cat_col}",
                            "figure": fig
                        })
        except Exception as e:
            logger.error(f"Error creating breakdown chart: {str(e)}")
        
        return graphs
    
    def _create_smart_default_charts(self) -> List[Dict]:
        """Create smart default visualizations with proper validation"""
        graphs = []
        
        try:
            numeric_cols = self._numeric_measures()
            categorical_cols = self._categorical_dimensions()
            
            if len(categorical_cols) > 0 and len(numeric_cols) > 0:
                cat_col = categorical_cols[0]
                num_col = numeric_cols[0]
                
                valid, reason = validate_chart_spec(self.df, 'bar', cat_col, num_col)
                if not valid:
                    logger.warning(f"Smart default chart blocked: {reason}")
                    return graphs
                
                top_data = self.df.groupby(cat_col)[num_col].sum().nlargest(10)
                
                if not top_data.empty:
                    title = f"SUM of {num_col} by {cat_col} (Top 10)"
                    fig = px.bar(
                        x=self._sanitize_for_json(top_data.values),
                        y=self._sanitize_for_json(top_data.index),
                        orientation='h',
                        title=title,
                        labels={'x': f'SUM of {num_col}', 'y': cat_col},
                        color=self._sanitize_for_json(top_data.values),
                        color_continuous_scale='Viridis'
                    )
                    
                    fig.update_layout(showlegend=False, height=400)
                    
                    graphs.append({
                        "type": "default_bar",
                        "title": title,
                        "figure": fig
                    })
        except Exception as e:
            logger.error(f"Error creating smart default chart: {str(e)}")
        
        return graphs
    
    def _create_trend_charts_old(self) -> List[Dict]:
        """Create trend/time-series charts"""
        graphs = []
        
        datetime_cols = self.df.select_dtypes(include=['datetime64']).columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        if len(datetime_cols) > 0 and len(numeric_cols) > 0:
            date_col = datetime_cols[0]
            num_col = numeric_cols[0]
            
            df_sorted = self.df.sort_values(date_col).copy()
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
        """Create distribution charts using only measure columns"""
        graphs = []
        
        numeric_cols = self._numeric_measures()[:2]
        
        for col in numeric_cols:
            valid, reason = validate_chart_spec(self.df, 'histogram', col, None)
            if not valid:
                logger.warning(f"Distribution chart for {col} blocked: {reason}")
                continue
            
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
                "title": f"Distribution of {col}",
                "figure": fig
            })
        
        return graphs
    
    def _create_correlation_charts(self, question: str) -> List[Dict]:
        """Create correlation charts using only measure columns (no IDs)"""
        graphs = []
        
        numeric_cols = self._numeric_measures()
        
        if len(numeric_cols) >= 2:
            corr_matrix = self.df[numeric_cols].corr()
            
            fig = go.Figure(data=go.Heatmap(
                z=self._sanitize_for_json(corr_matrix.values),
                x=self._sanitize_for_json(corr_matrix.columns),
                y=self._sanitize_for_json(corr_matrix.index),
                colorscale='RdBu',
                zmid=0
            ))
            
            fig.update_layout(title="Correlation Matrix (Business Measures Only)")
            
            graphs.append({
                "type": "correlation_heatmap",
                "title": "Correlation Analysis (Measures Only)",
                "figure": fig
            })
        
        return graphs
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.agent.clear_history()
        logger.info("Conversation history cleared")
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def set_model(self, model: str):
        """Change LLM model"""
        self.model = model
        self.agent.set_model(model)
        logger.info(f"Model changed to: {model}")
