import pandas as pd
from models.pandas_agent_chatbot import PandasAgentChatbot
import os

# Create dummy data
df1 = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Sales': [100, 200, 300]
})

df2 = pd.DataFrame({
    'Product': ['A', 'B', 'C'],
    'Stock': [10, 20, 30]
})

# Initialize agent with multiple DFs
agent = PandasAgentChatbot({'SalesData': df1, 'StockData': df2})

# Test query
question = "Compare sales in df1 with stock in df2 for product B"
print(f"Question: {question}")
response = agent.ask(question)
print(f"Answer: {response['answer']}")
