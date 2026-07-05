"""
Orchestrator
------------
Coordinates the multi-agent pipeline:
  DataLayer -> ForecastingAgent -> RiskAgent -> RecommendationAgent -> Decision

This mirrors an agent-graph pattern (comparable to LangGraph/CrewAI)
but implemented with plain Python for zero-dependency reliability
in a hackathon judging environment.
"""

from data.mock_data import generate_dataset
from agents.forecasting_agent import ForecastingAgent
from agents.risk_agent import RiskAgent
from agents.recommendation_agent import RecommendationAgent


class DecisionIntelligenceOrchestrator:
    def __init__(self):
        self.forecasting_agent = ForecastingAgent(horizon=7)
        self.risk_agent = RiskAgent()
        self.recommendation_agent = RecommendationAgent()

    def run_pipeline(self, df=None) -> dict:
        if df is None:
            df = generate_dataset()

        forecast_result = self.forecasting_agent.run(df)
        risk_result = self.risk_agent.run(df)
        recommendation_result = self.recommendation_agent.run(forecast_result, risk_result)

        return {
            "forecast": forecast_result,
            "risk": risk_result,
            "recommendation": recommendation_result,
            "raw_data_tail": df.tail(10).to_dict(orient="records"),
        }


if __name__ == "__main__":
    import json

    orchestrator = DecisionIntelligenceOrchestrator()
    output = orchestrator.run_pipeline()
    print(json.dumps(
        {k: v for k, v in output.items() if k != "raw_data_tail"},
        indent=2,
    ))
