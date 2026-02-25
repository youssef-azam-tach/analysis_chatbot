"""
Pandas Agent Chatbot Module
Uses LangChain's create_pandas_dataframe_agent for accurate data analysis
Prevents hallucination by executing Python code on actual DataFrame
"""

import pandas as pd
import logging
import numpy as np
import re
from typing import Dict, List, Optional, Union, Any
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

    def _relationships_text(self, context: Optional[Dict[str, Any]]) -> str:
        if not context:
            return "No explicit relationships provided."
        rels = context.get("manual_relationships") or []
        if not rels:
            return "No explicit relationships provided. Infer joins carefully from key-like columns and business semantics."
        lines = []
        for rel in rels[:25]:
            lines.append(
                f"- {rel.get('file1')}[{rel.get('column1')}] -> {rel.get('file2')}[{rel.get('column2')}] ({rel.get('relationship_type', 'related')})"
            )
        return "\n".join(lines)

    def _analysis_instruction(self, context: Optional[Dict[str, Any]]) -> str:
        mode = str((context or {}).get("analysis_mode") or "balanced").lower()
        if mode == "deep":
            return (
                "Use deep analysis: validate assumptions, test multiple groupings, include trend/comparison checks, "
                "and surface risks/limitations explicitly."
            )
        if mode == "fast":
            return "Use fast analysis: prioritize direct calculation with minimal overhead and concise explanation."
        return "Use balanced analysis: accurate calculations with concise but high-value interpretation."

    def _response_style_instruction(self, context: Optional[Dict[str, Any]]) -> str:
        style = str((context or {}).get("response_style") or "executive").lower()
        if style == "technical":
            return "Respond in technical analyst style with clear formulas/logic and computed evidence."
        if style == "deep":
            return "Respond in expert style with sections: Executive Answer, Evidence, Method, Assumptions, Next Actions."
        return "Respond in executive style: sharp answer first, then key evidence and recommendation."
    
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

    def _strip_markdown_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = str(text)
        cleaned = re.sub(r"```[\s\S]*?```", "", cleaned)
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\*\*(.*?)\*\*", r"\1", cleaned)
        cleaned = re.sub(r"\*(.*?)\*", r"\1", cleaned)
        cleaned = re.sub(r"`([^`]*)`", r"\1", cleaned)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"^\s*\d+[\.)]\s+", "", cleaned, flags=re.MULTILINE)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
        return cleaned.strip()
    
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
    
    def ask(self, question: str, temperature: Optional[float] = None, context: Optional[Dict[str, Any]] = None) -> Dict:
        """
        Ask a question and get answer based on DataFrame analysis
        """
        try:
            logger.info(f"Processing question: {question}")
            
            # Build prompt
            df_info = self._get_df_info()
            
            relationships_text = self._relationships_text(context)
            analysis_instruction = self._analysis_instruction(context)
            response_style_instruction = self._response_style_instruction(context)
            prefer_joins = bool((context or {}).get("prefer_joins", True))
            use_all_tables = bool((context or {}).get("use_all_tables", True))

            prompt = f"""You are a senior Data Engineer + Data Analyst + Data Scientist.
Analyze the data rigorously and answer the user question with evidence.

AVAILABLE DATA:
{df_info}

RELATIONSHIPS / JOIN HINTS:
{relationships_text}

RULES:
1. Write Python pandas code to answer the question.
2. Put your code in one ```python ``` block
3. Use print() to show results
4. NEVER write SQL - only Python pandas code
5. Use 'df' for single DataFrame or 'df1', 'df2' for multiple
6. {'Use all available tables and join when needed to answer correctly.' if use_all_tables else 'Use current working table only unless absolutely necessary.'}
7. {'Prefer explicit joins based on provided relationships before heuristic joins.' if prefer_joins else 'Use joins only when essential.'}
8. Never aggregate obvious identifier keys as business measures.
9. Validate join result quality (row counts, null rates in critical columns) before concluding.
10. Think like a professional analytics team member; avoid generic responses.
11. If the answer cannot be computed from available data, print exactly: INSUFFICIENT_DATA

ANALYSIS MODE:
{analysis_instruction}

RESPONSE STYLE:
{response_style_instruction}

QUESTION: {question}

Return ONLY the Python code block, no explanation text."""

            # Get LLM response
            response = self.llm.invoke(prompt)
            llm_text = response.content if hasattr(response, 'content') else str(response)
            
            # Extract and execute code if present
            code = self._extract_code(llm_text)
            if not code:
                retry_prompt = prompt + "\n\nReturn ONLY one Python code block that computes the answer."
                retry_response = self.llm.invoke(retry_prompt)
                retry_text = retry_response.content if hasattr(retry_response, 'content') else str(retry_response)
                code = self._extract_code(retry_text)
            code_output = ""

            if not code:
                safe_answer = (
                    "I could not produce a reliable computed answer from the available tables/columns. "
                    "Please rephrase the question with exact field names or clarify the required metric."
                )
                self.conversation_history.append({"role": "user", "content": question})
                self.conversation_history.append({"role": "assistant", "content": safe_answer, "code": None, "output": ""})
                return {
                    "answer": safe_answer,
                    "code": None,
                    "output": "",
                    "error": None,
                    "success": True
                }
            
            if code:
                logger.info(f"Executing code: {code[:100]}...")
                code_output = self._execute_code(code)
                logger.info(f"Code output: {code_output[:200]}...")
                
                # Get final answer with code results
                final_prompt = f"""Based on the executed analysis, produce a strong professional plain-text answer.

Question: {question}

Code executed:
```python
{code}
```

Result:
{code_output}

Requirements:
- Be explicit and numeric; cite concrete values from output.
- If multiple tables were used, mention join logic briefly.
- If uncertainty exists, state it clearly.
- End with one actionable recommendation.
- Do NOT use markdown symbols (#, *, -, backticks, code blocks).

Style:
{response_style_instruction}

Provide the final answer now."""

                final_response = self.llm.invoke(final_prompt)
                answer = final_response.content if hasattr(final_response, 'content') else str(final_response)
                answer = self._strip_markdown_text(answer)
            else:
                answer = self._strip_markdown_text(llm_text)
            
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
