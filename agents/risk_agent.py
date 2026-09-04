"""
Risk Agent
----------
Detects anomalies in supply chain data (inventory, demand) using IsolationForest.
Flags supply disruptions, low stock alerts, and unusual patterns.
"""

import numpy as np
from sklearn.ensemble import IsolationForest


class RiskAgent:
    """
    Detects supply chain anomalies and risks.
    Uses IsolationForest for unsupervised outlier detection.
    """
    
    def __init__(self, contamination=0.05, low_stock_threshold=50):
        """
        Args:
            contamination (float): Expected proportion of anomalies (0-1)
            low_stock_threshold (int): Inventory level below which to alert
        """
        self.contamination = contamination
        self.low_stock_threshold = low_stock_threshold
        self.model = None
    
    def run(self, df):
        """
        Run the risk agent on a DataFrame.
        
        Args:
            df (pd.DataFrame): Must have 'date' and 'inventory_level' columns
        
        Returns:
            dict: Risk analysis with anomalies, alerts, and risk level
        """
        inventory = df['inventory_level'].values.reshape(-1, 1)
        dates = df['date'].values
        
        # Train IsolationForest
        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=42,
            n_estimators=100
        )
        predictions = self.model.fit_predict(inventory)
        
        # Extract anomalies (-1 = anomaly, 1 = normal)
        anomaly_indices = np.where(predictions == -1)[0]
        anomaly_dates = [dates[i].strftime('%Y-%m-%d') for i in anomaly_indices]
        
        # Check for low stock
        current_inventory = inventory[-1, 0]
        low_stock_alert = current_inventory < self.low_stock_threshold
        
        # Calculate risk level
        num_anomalies = len(anomaly_indices)
        anomaly_rate = num_anomalies / len(inventory)
        
        if low_stock_alert or anomaly_rate > 0.1:
            risk_level = "HIGH"
        elif anomaly_rate > 0.05:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        # Anomaly severity (how far from normal)
        anomaly_severity = []
        if len(anomaly_indices) > 0:
            anomaly_values = inventory[anomaly_indices]
            mean_val = np.mean(inventory)
            severity = np.abs(anomaly_values - mean_val) / (mean_val + 1e-6)
            anomaly_severity = severity.flatten().tolist()
        
        return {
            "risk_level": risk_level,
            "anomaly_dates": anomaly_dates,
            "anomaly_count": num_anomalies,
            "anomaly_rate": float(anomaly_rate),
            "low_stock_alert": bool(low_stock_alert),
            "current_inventory": float(current_inventory),
            "threshold": self.low_stock_threshold,
            "anomaly_severity": anomaly_severity,
        }


if __name__ == "__main__":
    from data.mock_data import generate_dataset
    
    df = generate_dataset()
    agent = RiskAgent()
    result = agent.run(df)
    
    print(f"Risk Level: {result['risk_level']}")
    print(f"Anomalies Detected: {result['anomaly_count']}")
    print(f"Current Inventory: {result['current_inventory']:.1f}")
    print(f"Low Stock Alert: {result['low_stock_alert']}")
