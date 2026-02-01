"""
Create sample Excel files for testing
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

os.makedirs("./data", exist_ok=True)

# Sample 1: Sales Data
print("Creating Sales Data...")
dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
sales_data = {
    'Date': np.random.choice(dates, 365),
    'Product': np.random.choice(['Laptop', 'Desktop', 'Tablet', 'Phone', 'Monitor'], 365),
    'Category': np.random.choice(['Electronics', 'Accessories'], 365),
    'Quantity': np.random.randint(1, 50, 365),
    'Unit_Price': np.random.uniform(100, 2000, 365),
    'Customer_ID': np.random.randint(1000, 2000, 365),
    'Region': np.random.choice(['North', 'South', 'East', 'West'], 365),
}
sales_data['Total_Sales'] = sales_data['Quantity'] * sales_data['Unit_Price']
df_sales = pd.DataFrame(sales_data)
df_sales.to_excel('./data/sample_sales.xlsx', index=False)
print("✅ Created sample_sales.xlsx")

# Sample 2: HR Data
print("Creating HR Data...")
hr_data = {
    'Employee_ID': range(1001, 1101),
    'Name': [f'Employee_{i}' for i in range(1, 101)],
    'Department': np.random.choice(['HR', 'IT', 'Finance', 'Sales', 'Marketing'], 100),
    'Salary': np.random.randint(30000, 150000, 100),
    'Years_Experience': np.random.randint(0, 30, 100),
    'Performance_Score': np.random.uniform(1, 5, 100),
    'Hire_Date': [datetime(2020, 1, 1) + timedelta(days=int(x)) for x in np.random.uniform(0, 1460, 100)],
    'Status': np.random.choice(['Active', 'Inactive'], 100),
}
df_hr = pd.DataFrame(hr_data)
df_hr.to_excel('./data/sample_hr.xlsx', index=False)
print("✅ Created sample_hr.xlsx")

# Sample 3: Finance Data
print("Creating Finance Data...")
finance_data = {
    'Transaction_ID': range(5001, 5201),
    'Date': pd.date_range(start='2024-01-01', periods=200, freq='D'),
    'Amount': np.random.uniform(100, 10000, 200),
    'Category': np.random.choice(['Revenue', 'Expense', 'Investment'], 200),
    'Department': np.random.choice(['Operations', 'Marketing', 'R&D', 'Admin'], 200),
    'Status': np.random.choice(['Approved', 'Pending', 'Rejected'], 200),
    'Description': [f'Transaction_{i}' for i in range(1, 201)],
}
df_finance = pd.DataFrame(finance_data)
df_finance.to_excel('./data/sample_finance.xlsx', index=False)
print("✅ Created sample_finance.xlsx")

# Sample 4: Inventory Data
print("Creating Inventory Data...")
inventory_data = {
    'Product_ID': range(2001, 2101),
    'Product_Name': [f'Product_{i}' for i in range(1, 101)],
    'Category': np.random.choice(['Electronics', 'Accessories', 'Software'], 100),
    'Stock_Quantity': np.random.randint(0, 1000, 100),
    'Reorder_Level': np.random.randint(10, 100, 100),
    'Unit_Cost': np.random.uniform(10, 500, 100),
    'Supplier': np.random.choice(['Supplier_A', 'Supplier_B', 'Supplier_C'], 100),
    'Last_Restock': pd.date_range(start='2024-01-01', periods=100, freq='D'),
}
df_inventory = pd.DataFrame(inventory_data)
df_inventory.to_excel('./data/sample_inventory.xlsx', index=False)
print("✅ Created sample_inventory.xlsx")

# Sample 5: Customer Data
print("Creating Customer Data...")
customer_data = {
    'Customer_ID': range(3001, 3201),
    'Name': [f'Customer_{i}' for i in range(1, 201)],
    'Email': [f'customer_{i}@example.com' for i in range(1, 201)],
    'Phone': [f'555-{np.random.randint(1000, 9999)}' for _ in range(200)],
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], 200),
    'Country': 'USA',
    'Total_Purchases': np.random.randint(1, 100, 200),
    'Total_Spent': np.random.uniform(100, 50000, 200),
    'Last_Purchase_Date': pd.date_range(start='2023-01-01', end='2024-12-31', periods=200),
    'Customer_Segment': np.random.choice(['Premium', 'Standard', 'Basic'], 200),
}
df_customers = pd.DataFrame(customer_data)
df_customers.to_excel('./data/sample_customers.xlsx', index=False)
print("✅ Created sample_customers.xlsx")

print("\n✅ All sample files created successfully!")
print("Files location: ./data/")
