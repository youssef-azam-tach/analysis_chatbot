import pandas as pd
from typing import Dict, List, Any
import json
import plotly.graph_objects as go
from models.hybrid_chatbot import HybridChatbot

class DashboardBuilder:
    def __init__(self, df: pd.DataFrame, goals: Dict[str, str] = None):
        self.df = df
        self.goals = goals or {"problem": "", "objective": "", "target": ""}
        self.chatbot = HybridChatbot(df)
        
    def suggest_kpis(self) -> List[Dict[str, Any]]:
        """Suggest key performance indicators based on goals"""
        prompt = f"""
        Based on the following business context:
        - Problem: {self.goals['problem']}
        - Objective: {self.goals['objective']}
        - Target Audience: {self.goals['target']}
        
        Suggest 4 critical Key Performance Indicators (KPIs) that should be tracked on a dashboard for this dataset.
        Format your response as a list of dictionaries with 'label' and 'description'.
        """
        # For simplicity, we use the chatbot to get suggestions
        # In a real app, we might parse JSON
        response = self.chatbot.chat(prompt)
        return response["answer"]

    def build_summary_dashboard(self) -> List[Dict[str, Any]]:
        """Generate a set of recommended charts for a dashboard"""
        prompt = f"""
        Analyze the dataset and create a dashboard summary for:
        Problem: {self.goals['problem']}
        Objective: {self.goals['objective']}
        
        Generate 2-3 most important visualizations that address these goals.
        """
        response = self.chatbot.chat(prompt)
        pinned_items = []
        
        for graph in response.get("graphs", []):
            plotly_json = graph.get("plotly_json")
            if not plotly_json:
                figure_obj = graph.get("figure")
                if hasattr(figure_obj, "to_json"):
                    try:
                        plotly_json = figure_obj.to_json()
                    except Exception:
                        plotly_json = None

            pinned_items.append({
                "title": graph["title"],
                "plotly_json": plotly_json,
                "type": "chart"
            })
            
        return pinned_items
