"""
Pandas Agent Chatbot Module
Uses LangChain's create_pandas_dataframe_agent for accurate data analysis
Prevents hallucination by executing Python code on actual DataFrame
"""

import pandas as pd
import logging
import numpy as np
import re
from typing import Dict, List, Optional, Union
from langchain_ollama import ChatOllama

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PandasAgentChatbot:
    """
    Simple pandas DataFrame chatbot using direct code execution
    
    Features:
    - Executes Python code on DataFrames for accurate results
    - Prevents hallucination through code execution
    - Works reliably with local LLMs
    """
    
    def __init__(self, df: Union[pd.DataFrame, List[pd.DataFrame], Dict[str, pd.DataFrame]], model: str = "qwen2.5:7b"):
        """
        Initialize pandas agent chatbot
        
        Args:
            df: DataFrame(s) to analyze. Can be a single DF, a list of DFs, or a dict {name: df}.
            model: LLM model name for Ollama
        """
        self.model = model
        self.conversation_history = []
        
        # Handle different input types for df
        if isinstance(df, dict):
            self.dfs = list(df.values())
            self.df_names = list(df.keys())
        elif isinstance(df, list):
            self.dfs = df
            self.df_names = [f"df{i+1}" for i in range(len(df))]
        else:
            self.dfs = [df]
            self.df_names = ["df"]
            
        # For compatibility with existing single-df code
        self.df = self.dfs[0]
        
        # Initialize Ollama LLM
        self.llm = ChatOllama(
            model=self.model,
            temperature=0.0,
        )
        
        logger.info(f"✅ PandasAgentChatbot initialized with {len(self.dfs)} dataframes using model: {model}")
    
    def _get_df_info(self) -> str:
        """Get DataFrame info for prompt"""
        info_parts = []
        for i, (name, df) in enumerate(zip(self.df_names, self.dfs)):
            var_name = "df" if len(self.dfs) == 1 else f"df{i+1}"
            cols = list(df.columns)
            info_parts.append(f"DataFrame `{var_name}`: {df.shape[0]} rows, {df.shape[1]} columns\nColumns: {cols}")
        return "\n\n".join(info_parts)
    
    def _extract_code(self, text: str) -> Optional[str]:
        """Extract Python code from LLM response"""
        # Try to find code in ```python blocks
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            return matches[0].strip()
        
        # Try ``` blocks
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, text, re.DOTALL)
        if matches:
            code = matches[0].strip()
            # Check if it looks like Python (not SQL)
            if not code.upper().startswith('SELECT') and not code.upper().startswith('WITH'):
                return code
        
        return None
    
    def _execute_code(self, code: str) -> str:
        """Execute Python code safely on DataFrames"""
        try:
            # Create execution environment
            local_vars = {'pd': pd, 'np': np}
            
            # Add dataframes to environment
            if len(self.dfs) == 1:
                local_vars['df'] = self.dfs[0]
            else:
                for i, dframe in enumerate(self.dfs):
                    local_vars[f'df{i+1}'] = dframe
            
            # Capture print output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = buffer = io.StringIO()
            
            # Execute code
            exec(code, local_vars)
            
            # Get output
            output = buffer.getvalue()
            sys.stdout = old_stdout
            
            # If no print output, try to get the last expression result
            if not output.strip():
                try:
                    result = eval(code.split('\n')[-1], local_vars)
                    if result is not None:
                        output = str(result)
                except:
                    pass
            
            return output if output.strip() else "Code executed successfully (no output)"
            
        except Exception as e:
            return f"Error executing code: {str(e)}"
    
    def ask(self, question: str, temperature: Optional[float] = None) -> Dict:
        """
        Ask a question and get answer based on DataFrame analysis
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Build prompt
            df_info = self._get_df_info()
            
            prompt = f"""You are a data analyst. Analyze the data and answer the question.

AVAILABLE DATA:
{df_info}

RULES:
1. Write Python pandas code to answer the question
2. Put your code in ```python ``` blocks
3. Use print() to show results
4. NEVER write SQL - only Python pandas code
5. Use 'df' for single DataFrame or 'df1', 'df2' for multiple

QUESTION: {question}

Write the Python code to answer this, then explain the result."""

            # Get LLM response
            response = self.llm.invoke(prompt)
            llm_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract and execute code if present
            code = self._extract_code(llm_text)
            code_output = ""
            
            if code:
                logger.info(f"Executing code: {code[:100]}...")
                code_output = self._execute_code(code)
                logger.info(f"Code output: {code_output[:200]}...")
                
                # Get final answer with code results
                final_prompt = f"""Based on the data analysis:

Question: {question}

Code executed:
```python
{code}
```

Result:
{code_output}

Provide a clear, concise answer based on these actual results. Be specific with numbers from the output."""

                final_response = self.llm.invoke(final_prompt)
                answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
            else:
                # No code found, use LLM response directly
                answer = llm_text
            
            # Store in conversation history
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": answer, "code": code, "output": code_output})
            
            return {
                "answer": answer,
                "code": code,
                "output": code_output,
                "error": None,
                "success": True
            }
            
        except Exception as e:
            error_msg = f"Error processing question: {str(e)}"
            logger.error(error_msg)
            
            self.conversation_history.append({"role": "user", "content": question})
            self.conversation_history.append({"role": "assistant", "content": f"❌ {error_msg}", "type": "error"})
            
            return {
                "answer": f"⚠️ I encountered an error: {str(e)}",
                "error": str(e),
                "success": False
            }
    
    def get_history(self) -> List[Dict]:
        """Get conversation history"""
        return self.conversation_history
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        logger.info("Conversation history cleared")
    
    def set_model(self, model: str):
        """Change the LLM model"""
        self.model = model
        self.llm = ChatOllama(model=model, temperature=0.0)
        logger.info(f"Model changed to: {model}")
    
    def get_dataframe_info(self) -> Dict:
        """Get basic information about the DataFrame"""
        return {
            "count": len(self.dfs),
            "df_names": self.df_names,
            "total_rows": sum(df.shape[0] for df in self.dfs)
        }
