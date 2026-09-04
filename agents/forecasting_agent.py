"""
Forecasting Agent
-----------------
Predicts demand for the next N days using Holt's exponential smoothing.
In production, this integrates with NVIDIA NIM for GPU-accelerated inference.
"""

import numpy as np
from datetime import datetime, timedelta


class ForecastingAgent:
    """
    Forecasts demand using exponential smoothing.
    Designed as a drop-in replacement for GPU-accelerated models (NVIDIA NIM).
    """
    
    def __init__(self, horizon=7, alpha=0.3, beta=0.1):
        """
        Args:
            horizon (int): Days ahead to forecast
            alpha (float): Level smoothing parameter (0-1)
            beta (float): Trend smoothing parameter (0-1)
        """
        self.horizon = horizon
        self.alpha = alpha
        self.beta = beta
    
    def _holt_exponential_smoothing(self, series):
        """
        Implement Holt's exponential smoothing for trend-based forecasting.
        
        Args:
            series (np.array): Historical demand values
        
        Returns:
            tuple: (level, trend, forecast_values)
        """
        # Initialize
        level = series[0]
        trend = series[1] - series[0]
        forecasts = []
        
        # Smooth historical data
        for value in series[1:]:
            prev_level = level
            level = self.alpha * value + (1 - self.alpha) * (level + trend)
            trend = self.beta * (level - prev_level) + (1 - self.beta) * trend
        
        # Forecast ahead
        for i in range(1, self.horizon + 1):
            forecast = level + i * trend
            forecasts.append(max(forecast, 0))  # Ensure non-negative
        
        return level, trend, forecasts
    
    def predict_gpu_accelerated(self, series):
        """
        Stub for NVIDIA NIM GPU acceleration.
        In production, this would call a remote NIM endpoint.
        For now, falls back to CPU-based smoothing.
        
        Args:
            series (np.array): Historical demand
        
        Returns:
            list: GPU-accelerated forecast (currently CPU fallback)
        """
        # TODO: Replace with actual NVIDIA NIM call
        # response = requests.post(f"{NIM_ENDPOINT}/v1/forecast", json={"data": series})
        # return response.json()["forecast"]
        
        return self._holt_exponential_smoothing(series)[2]
    
    def run(self, df):
        """
        Run the forecasting agent on a DataFrame.
        
        Args:
            df (pd.DataFrame): Must have 'date' and 'demand' columns
        
        Returns:
            dict: Forecast results with values, dates, and metadata
        """
        demand_series = df['demand'].values
        last_date = df['date'].iloc[-1]
        
        # Compute forecast
        level, trend, forecast_values = self._holt_exponential_smoothing(demand_series)
        
        # Generate future dates
        forecast_dates = [
            (last_date + timedelta(days=i)).strftime('%Y-%m-%d')
            for i in range(1, self.horizon + 1)
        ]
        
        # Determine trend direction
        trend_direction = "INCREASING" if trend > 0 else "DECREASING"
        
        # Compute confidence (inverse of variance)
        variance = np.var(demand_series[-30:])  # Recent 30 days
        confidence = 1.0 / (1.0 + variance / 100)
        
        return {
            "trend": trend_direction,
            "forecast_values": forecast_values,
            "forecast_dates": forecast_dates,
            "confidence": confidence,
            "level": level,
            "trend_slope": trend,
            "horizon": self.horizon,
        }


if __name__ == "__main__":
    from data.mock_data import generate_dataset
    
    df = generate_dataset()
    agent = ForecastingAgent(horizon=7)
    result = agent.run(df)
    
    print(f"Trend: {result['trend']}")
    print(f"Next 7 days forecast: {result['forecast_values']}")
    print(f"Confidence: {result['confidence']:.2%}")
