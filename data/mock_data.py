"""
Mock Data Generator
-------------------
Generates synthetic retail/supply-chain data for testing the Decision Intelligence Platform.
In production, this would connect to Google BigQuery.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_dataset(n_days=180, seed=42):
    """
    Generate mock sales, inventory, and market data.
    
    Args:
        n_days (int): Number of historical days to generate
        seed (int): Random seed for reproducibility
    
    Returns:
        pd.DataFrame: DataFrame with date, demand, inventory_level, price
    """
    np.random.seed(seed)
    
    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=n_days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Generate demand with trend + seasonality + noise
    trend = np.linspace(100, 150, len(dates))
    seasonality = 30 * np.sin(np.arange(len(dates)) * 2 * np.pi / 30)
    noise = np.random.normal(0, 10, len(dates))
    demand = trend + seasonality + noise
    demand = np.maximum(demand, 10)  # Ensure positive demand
    
    # Generate inventory (starts at 200, changes based on demand)
    inventory = [200]
    for d in demand[1:]:
        # Restock when low, otherwise deplete
        current = inventory[-1]
        if current < 50:
            current += np.random.uniform(100, 150)  # Restock event
        current -= d / 30  # Depletion based on demand
        inventory.append(max(current, 0))
    
    # Add random anomalies (supply disruptions)
    anomaly_indices = np.random.choice(len(dates), size=max(1, len(dates) // 30), replace=False)
    for idx in anomaly_indices:
        inventory[idx] *= np.random.uniform(0.4, 0.7)  # Sudden drop
    
    # Generate price (inversely correlated with inventory)
    price = 50 - (np.array(inventory) / max(inventory)) * 20 + np.random.normal(0, 2, len(dates))
    price = np.maximum(price, 10)
    
    return pd.DataFrame({
        'date': dates,
        'demand': demand,
        'inventory_level': inventory,
        'price': price
    })


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(10))
    print(f"\nGenerated {len(df)} days of mock data")
