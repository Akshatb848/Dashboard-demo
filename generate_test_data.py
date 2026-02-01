"""
Test Data Generator for AI Analytics Dashboard
Creates various test datasets to validate all features
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_sales_data(n_rows=1000):
    """Generate sales transaction data"""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(n_rows)]
    
    data = {
        'order_id': [f'ORD-{i:06d}' for i in range(1, n_rows + 1)],
        'order_date': dates,
        'customer_id': [f'CUST-{i:04d}' for i in np.random.randint(1, 500, n_rows)],
        'product_category': np.random.choice(['Electronics', 'Clothing', 'Home', 'Books'], n_rows),
        'region': np.random.choice(['North', 'South', 'East', 'West'], n_rows),
        'revenue': np.random.gamma(100, 2, n_rows),
        'cost': np.random.gamma(60, 2, n_rows),
        'quantity': np.random.randint(1, 10, n_rows),
        'discount_percent': np.random.uniform(0, 20, n_rows),
        'shipping_days': np.random.randint(1, 7, n_rows)
    }
    
    df = pd.DataFrame(data)
    df['profit'] = df['revenue'] - df['cost']
    df['profit_margin'] = (df['profit'] / df['revenue'] * 100).round(2)
    
    return df

def generate_financial_data(n_rows=500):
    """Generate financial metrics data"""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=n_rows, freq='D')
    
    # Generate base trend with seasonality
    trend = np.linspace(100000, 150000, n_rows)
    seasonality = 10000 * np.sin(2 * np.pi * np.arange(n_rows) / 365)
    noise = np.random.normal(0, 5000, n_rows)
    
    revenue = trend + seasonality + noise
    
    data = {
        'date': dates,
        'revenue': revenue,
        'expenses': revenue * np.random.uniform(0.6, 0.8, n_rows),
        'cash_flow': np.random.normal(50000, 10000, n_rows),
        'accounts_receivable': np.random.uniform(20000, 100000, n_rows),
        'accounts_payable': np.random.uniform(15000, 80000, n_rows),
        'inventory_value': np.random.uniform(50000, 200000, n_rows),
        'debt_ratio': np.random.uniform(0.2, 0.6, n_rows),
        'quick_ratio': np.random.uniform(0.8, 2.0, n_rows)
    }
    
    df = pd.DataFrame(data)
    df['net_income'] = df['revenue'] - df['expenses']
    df['profit_margin_pct'] = (df['net_income'] / df['revenue'] * 100).round(2)
    
    return df

def generate_hr_data(n_rows=300):
    """Generate HR/workforce data"""
    np.random.seed(42)
    
    departments = ['Engineering', 'Sales', 'Marketing', 'Finance', 'HR', 'Operations']
    job_levels = ['Junior', 'Mid', 'Senior', 'Lead', 'Manager']
    
    data = {
        'employee_id': [f'EMP-{i:05d}' for i in range(1, n_rows + 1)],
        'department': np.random.choice(departments, n_rows),
        'job_level': np.random.choice(job_levels, n_rows),
        'hire_date': pd.date_range('2015-01-01', periods=n_rows, freq='W'),
        'age': np.random.randint(22, 65, n_rows),
        'salary': np.random.gamma(80, 1000, n_rows),
        'performance_score': np.random.uniform(60, 100, n_rows),
        'satisfaction_score': np.random.uniform(50, 100, n_rows),
        'years_experience': np.random.randint(0, 20, n_rows),
        'training_hours': np.random.randint(0, 100, n_rows),
        'sick_days': np.random.poisson(5, n_rows),
        'overtime_hours': np.random.poisson(10, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    return df

def generate_ecommerce_data(n_rows=2000):
    """Generate e-commerce web analytics data"""
    np.random.seed(42)
    
    start_date = datetime(2023, 1, 1)
    dates = pd.date_range(start_date, periods=n_rows, freq='H')
    
    data = {
        'timestamp': dates,
        'page_views': np.random.poisson(100, n_rows),
        'unique_visitors': np.random.poisson(70, n_rows),
        'sessions': np.random.poisson(85, n_rows),
        'bounce_rate_pct': np.random.uniform(30, 70, n_rows),
        'avg_session_duration_sec': np.random.gamma(180, 2, n_rows),
        'conversion_rate_pct': np.random.uniform(1, 5, n_rows),
        'cart_additions': np.random.poisson(15, n_rows),
        'checkouts': np.random.poisson(10, n_rows),
        'revenue': np.random.gamma(500, 5, n_rows),
        'ads_clicks': np.random.poisson(50, n_rows),
        'ads_spend': np.random.gamma(100, 2, n_rows),
        'organic_traffic_pct': np.random.uniform(40, 80, n_rows)
    }
    
    df = pd.DataFrame(data)
    df['conversion_value'] = df['revenue'] / df['unique_visitors']
    df['roas'] = (df['revenue'] / df['ads_spend']).replace([np.inf, -np.inf], 0)
    
    return df

def generate_dirty_data(n_rows=500):
    """Generate intentionally messy data to test robustness"""
    np.random.seed(42)
    
    data = {
        'Mixed Date': pd.date_range('2023-01-01', periods=n_rows, freq='D').tolist() + [None] * 20,
        'Numeric with Nulls': list(np.random.randn(n_rows)) + [None] * 20,
        'Text Column': ['Value' + str(i) for i in range(n_rows)] + [''] * 20,
        'Almost All Nulls': [1, 2, 3] + [None] * (n_rows + 17),
        'All Same Value': ['Constant'] * (n_rows + 20),
        'High Cardinality': [f'ID-{i}' for i in range(n_rows + 20)],
        'Low Cardinality': np.random.choice(['A', 'B', 'C'], n_rows + 20),
        'Outliers Present': list(np.random.normal(100, 10, n_rows)) + [1000, -1000] * 10,
        'Zero Variance': [42.0] * (n_rows + 20),
        'Mixed Types': [1, 'two', 3.0, 'four', 5] * ((n_rows + 20) // 5)
    }
    
    df = pd.DataFrame(data)
    
    return df

def save_test_datasets(output_dir='test_data'):
    """Save all test datasets"""
    import os
    
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = {
        'sales_data.csv': generate_sales_data(),
        'financial_data.csv': generate_financial_data(),
        'hr_data.csv': generate_hr_data(),
        'ecommerce_data.csv': generate_ecommerce_data(),
        'dirty_data.csv': generate_dirty_data()
    }
    
    for filename, df in datasets.items():
        filepath = os.path.join(output_dir, filename)
        df.to_csv(filepath, index=False)
        print(f"✅ Generated: {filepath} ({len(df)} rows, {len(df.columns)} columns)")
    
    print(f"\n📁 All test datasets saved to '{output_dir}/' directory")
    
    return datasets

if __name__ == "__main__":
    print("🔧 Generating Test Datasets...\n")
    save_test_datasets()
