"""
Recommendation Agent
--------------------
Synthesizes forecasting and risk signals into a single explainable business decision.
Uses Claude LLM for reasoning when available, falls back to deterministic rules.
"""

import os
import json


class RecommendationAgent:
    """
    Generates business recommendations by synthesizing forecast + risk data.
    Falls back to rule-based reasoning if LLM is unavailable.
    """
    
    def __init__(self, use_llm=True):
        """
        Args:
            use_llm (bool): Try to use Claude API if ANTHROPIC_API_KEY is set
        """
        self.use_llm = use_llm and os.getenv("ANTHROPIC_API_KEY")
        self.client = None
        
        if self.use_llm:
            try:
                from anthropic import Anthropic
                self.client = Anthropic()
            except ImportError:
                self.use_llm = False
    
    def _rule_based_decision(self, forecast, risk):
        """
        Deterministic fallback: synthesize decisions using business rules.
        
        Args:
            forecast (dict): Output from ForecastingAgent
            risk (dict): Output from RiskAgent
        
        Returns:
            dict: Decision, confidence, explanation, reasoning_mode
        """
        trend = forecast["trend"]
        risk_level = risk["risk_level"]
        low_stock = risk["low_stock_alert"]
        confidence = forecast["confidence"]
        anomaly_rate = risk["anomaly_rate"]
        
        # Decision logic
        if low_stock:
            decision = "URGENT_RESTOCK"
            explanation = (
                f"Current inventory is critically low ({risk['current_inventory']:.0f} units). "
                f"Immediate restocking required regardless of forecast."
            )
        elif risk_level == "HIGH":
            decision = "INCREASE_BUFFER_STOCK"
            explanation = (
                f"High supply chain risk detected ({anomaly_rate:.1%} anomaly rate). "
                f"Increase safety stock to mitigate disruptions."
            )
        elif trend == "INCREASING" and risk_level == "LOW":
            decision = "PREPARE_FOR_SURGE"
            explanation = (
                f"Demand is increasing and supply is stable. "
                f"Pre-position inventory to capture demand surge (confidence: {confidence:.0%})."
            )
        elif trend == "DECREASING":
            decision = "OPTIMIZE_INVENTORY"
            explanation = (
                f"Demand is decreasing. Optimize inventory levels downward "
                f"to free up capital and reduce holding costs."
            )
        else:
            decision = "MAINTAIN_STATUS_QUO"
            explanation = (
                f"Demand trend is stable and supply chain is healthy. "
                f"Maintain current inventory and order levels."
            )
        
        base_confidence = 0.7 if risk_level == "LOW" else 0.5
        return {
            "decision": decision,
            "confidence": base_confidence,
            "explanation": explanation,
            "reasoning_mode": "rule_based",
        }
    
    def _llm_decision(self, forecast, risk):
        """
        Use Claude to synthesize forecast + risk into recommendation.
        
        Args:
            forecast (dict): Output from ForecastingAgent
            risk (dict): Output from RiskAgent
        
        Returns:
            dict: Decision, confidence, explanation, reasoning_mode
        """
        prompt = f"""
You are a supply chain optimization expert. Analyze the following signals and provide a single, actionable business decision.

**Forecasting Signal:**
- Trend: {forecast['trend']}
- Next {forecast['horizon']} days forecast: {[round(v, 1) for v in forecast['forecast_values']]}
- Confidence: {forecast['confidence']:.0%}

**Risk Signal:**
- Risk Level: {risk['risk_level']}
- Anomalies Detected: {risk['anomaly_count']} ({risk['anomaly_rate']:.1%} of data)
- Current Inventory: {risk['current_inventory']:.0f} units
- Low Stock Alert: {risk['low_stock_alert']}

Provide a JSON response with exactly this structure (no markdown, pure JSON):
{{
  "decision": "DECISION_NAME (one of: URGENT_RESTOCK, INCREASE_BUFFER_STOCK, PREPARE_FOR_SURGE, OPTIMIZE_INVENTORY, MAINTAIN_STATUS_QUO)",
  "confidence": 0.75,
  "explanation": "Clear, concise business explanation (2-3 sentences)"
}}
"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Parse JSON response
            response_text = message.content[0].text
            result = json.loads(response_text)
            result["reasoning_mode"] = "llm"
            return result
        
        except Exception as e:
            # Fallback to rule-based if LLM fails
            print(f"LLM error, falling back to rule-based: {e}")
            return self._rule_based_decision(forecast, risk)
    
    def run(self, forecast, risk):
        """
        Generate recommendation by synthesizing forecast + risk.
        
        Args:
            forecast (dict): Output from ForecastingAgent
            risk (dict): Output from RiskAgent
        
        Returns:
            dict: Recommendation with decision, explanation, confidence
        """
        if self.use_llm:
            return self._llm_decision(forecast, risk)
        else:
            return self._rule_based_decision(forecast, risk)


if __name__ == "__main__":
    from data.mock_data import generate_dataset
    from forecasting_agent import ForecastingAgent
    from risk_agent import RiskAgent
    
    df = generate_dataset()
    
    forecaster = ForecastingAgent(horizon=7)
    forecast = forecaster.run(df)
    
    risk_agent = RiskAgent()
    risk = risk_agent.run(df)
    
    recommender = RecommendationAgent(use_llm=False)  # Use rule-based for testing
    recommendation = recommender.run(forecast, risk)
    
    print(f"Decision: {recommendation['decision']}")
    print(f"Confidence: {recommendation['confidence']:.0%}")
    print(f"Explanation: {recommendation['explanation']}")
