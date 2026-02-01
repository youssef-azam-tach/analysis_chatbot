"""
LLM Chatbot Module
Direct LLM-based chatbot trained on data context
No RAG - uses data-to-text conversion for context
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import streamlit as st

try:
    import ollama
except ImportError:
    st.error("Ollama not installed. Please install it with: pip install ollama")

from models.data_to_text import DataToText


class LLMChatbot:
    """
    Direct LLM chatbot trained on data
    
    Features:
    - No RAG complexity
    - Direct data-to-text conversion
    - Full data context in every query
    - Streaming responses
    """
    
    def __init__(self, df: pd.DataFrame, model: str = "qwen2.5:7b"):
        """
        Initialize chatbot with data
        
        Args:
            df: DataFrame to analyze
            model: LLM model name
        """
        self.df = df
        self.model = model
        self.data_context = None
        self.conversation_history = []
        
        # Generate data context
        self._generate_data_context()
    
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
"""
            
            st.success("✅ Data context loaded for chatbot")
            return True
        
        except Exception as e:
            st.error(f"Error generating data context: {str(e)}")
            return False
    
    def chat(self, user_message: str, temperature: float = 0.7, 
             max_tokens: int = 1024) -> str:
        """
        Send message to chatbot and get response
        
        Args:
            user_message: User's question
            temperature: LLM temperature (0-1)
            max_tokens: Maximum response length
            
        Returns:
            Chatbot response
        """
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Build system prompt
            system_prompt = """You are an expert data analyst assistant. 
Your role is to:
1. Answer questions about the provided dataset
2. Provide insights and analysis
3. Suggest visualizations and further analysis
4. Explain data patterns and relationships
5. Give actionable recommendations

Always base your answers on the actual data provided.
If you don't have information to answer a question, say so clearly.
Be concise but thorough in your explanations."""
            
            # Build conversation context
            messages = [
                {
                    "role": "system",
                    "content": system_prompt + "\n\n" + self.data_context
                }
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-5:]:  # Keep last 5 messages for context
                messages.append(msg)
            
            # Generate response
            response = ollama.chat(
                model=self.model,
                messages=messages,
                stream=False,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            )
            
            assistant_message = response["message"]["content"]
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message
            })
            
            return assistant_message
        
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            return error_msg
    
    def chat_streaming(self, user_message: str, temperature: float = 0.7,
                      max_tokens: int = 1024):
        """
        Send message and stream response
        
        Args:
            user_message: User's question
            temperature: LLM temperature
            max_tokens: Maximum response length
            
        Yields:
            Response chunks
        """
        try:
            # Add to conversation history
            self.conversation_history.append({
                "role": "user",
                "content": user_message
            })
            
            # Build system prompt
            system_prompt = """You are an expert data analyst assistant. 
Your role is to:
1. Answer questions about the provided dataset
2. Provide insights and analysis
3. Suggest visualizations and further analysis
4. Explain data patterns and relationships
5. Give actionable recommendations

Always base your answers on the actual data provided.
If you don't have information to answer a question, say so clearly.
Be concise but thorough in your explanations."""
            
            # Build conversation context
            messages = [
                {
                    "role": "system",
                    "content": system_prompt + "\n\n" + self.data_context
                }
            ]
            
            # Add conversation history
            for msg in self.conversation_history[-5:]:
                messages.append(msg)
            
            # Stream response
            full_response = ""
            for chunk in ollama.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens,
                }
            ):
                content = chunk["message"]["content"]
                full_response += content
                yield content
            
            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
        
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            st.error(error_msg)
            yield error_msg
    
    def get_data_summary(self) -> str:
        """Get summary of the data"""
        return self.data_context
    
    def get_conversation_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
    
    def set_model(self, model: str):
        """Change the LLM model"""
        self.model = model
    
    def get_available_models(self) -> List[str]:
        """Get list of available models"""
        try:
            response = ollama.list()
            models = [model["name"].split(":")[0] for model in response["models"]]
            return list(set(models))
        except Exception as e:
            st.error(f"Error fetching models: {str(e)}")
            return ["qwen2.5:7b"]
    
    def analyze_column(self, column_name: str) -> str:
        """
        Analyze a specific column
        
        Args:
            column_name: Name of column to analyze
            
        Returns:
            Analysis of the column
        """
        if column_name not in self.df.columns:
            return f"Column '{column_name}' not found in dataset"
        
        col_data = self.df[column_name]
        
        analysis = f"""
Column: {column_name}
Data Type: {col_data.dtype}
Non-Null Count: {col_data.notna().sum()}
Null Count: {col_data.isna().sum()}
Unique Values: {col_data.nunique()}
"""
        
        if pd.api.types.is_numeric_dtype(col_data):
            analysis += f"""
Mean: {col_data.mean():.2f}
Median: {col_data.median():.2f}
Std Dev: {col_data.std():.2f}
Min: {col_data.min():.2f}
Max: {col_data.max():.2f}
"""
        else:
            top_values = col_data.value_counts().head(5)
            analysis += f"\nTop Values:\n"
            for val, count in top_values.items():
                analysis += f"  {val}: {count}\n"
        
        return analysis
    
    def ask_about_column(self, column_name: str, question: str) -> str:
        """
        Ask a specific question about a column
        
        Args:
            column_name: Column to analyze
            question: Question about the column
            
        Returns:
            Answer
        """
        column_analysis = self.analyze_column(column_name)
        
        prompt = f"""Based on this column analysis:
{column_analysis}

Answer this question: {question}"""
        
        return self.chat(prompt)
    
    def suggest_analysis(self) -> str:
        """
        Suggest analysis based on the data
        
        Returns:
            Analysis suggestions
        """
        prompt = """Based on the dataset provided, what are the top 5 analyses or visualizations you would recommend? 
Please be specific and explain why each would be valuable."""
        
        return self.chat(prompt)
    
    def detect_insights(self) -> str:
        """
        Automatically detect and report key insights
        
        Returns:
            Key insights from the data
        """
        prompt = """Analyze the dataset and identify the top 5 key insights or patterns. 
For each insight, explain:
1. What the insight is
2. Why it's important
3. What action could be taken based on this insight"""
        
        return self.chat(prompt)
