# AI-Powered Decision Intelligence Platform

**Google Cloud Gen AI Academy — Cohort 2 Hackathon (APAC Edition)**
**Sponsors: Google Cloud × NVIDIA**

## 🎯 Problem Statement

Retail and supply-chain businesses generate huge volumes of sales,
inventory, and market data — but decisions (when to reorder, when to
flag a risk, how to price) are still made manually and reactively.

This platform uses a **multi-agent AI system** to continuously
analyze that data and produce a single, **explainable decision** —
not just a prediction, but a *reason* a human can trust and act on.

## 🏗️ Architecture

```
                ┌─────────────────────────┐
                │   Data Layer            │
                │   (Google BigQuery /    │
                │    mock streaming data) │
                └────────────┬────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    │
┌───────────────┐   ┌───────────────┐             │
│ Forecasting   │   │ Risk Agent    │             │
│ Agent         │   │ (anomaly      │             │
│ (Holt trend / │   │  detection —  │             │
│  NVIDIA NIM   │   │  IsolationForest)           │
│  in prod)     │   └───────┬───────┘             │
└───────┬───────┘           │                     │
        └─────────┬─────────┘                     │
                  ▼                               │
          ┌───────────────────┐                   │
          │ Recommendation    │◄──────────────────┘
          │ Agent (LLM-based  │
          │ synthesis + expl.)│
          └─────────┬─────────┘
                    ▼
          ┌───────────────────┐
          │  Decision + Why   │
          │  (Streamlit UI)   │
          └───────────────────┘
```

### Agents

| Agent | Role | Tech |
|---|---|---|
| **ForecastingAgent** | Predicts demand for the next N days | Holt's exponential smoothing (numpy); swappable for GPU model via **NVIDIA NIM/Triton** |
| **RiskAgent** | Detects anomalies (supply disruption, stock risk) | IsolationForest (scikit-learn) |
| **RecommendationAgent** | Synthesizes both signals into one explainable decision | Claude LLM reasoning, with deterministic rule-based fallback |

### Sponsor integration
- **Google Cloud**: Data layer designed for BigQuery ingestion; deployable to Vertex AI / Cloud Run (see comments in `data/mock_data.py` and `agents/forecasting_agent.py`)
- **NVIDIA**: `ForecastingAgent.predict_gpu_accelerated()` is a stubbed integration point for an NVIDIA NIM microservice, for GPU-accelerated inference in production

## 🚀 Running the Demo

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

Optional — enable real LLM reasoning (else falls back to rule-based, which always works offline):
```bash
export ANTHROPIC_API_KEY=your_key_here
```

Or run the pipeline headless:
```bash
python orchestrator.py
```

## 📁 Project Structure

```
decision_intelligence_platform/
├── agents/
│   ├── forecasting_agent.py
│   ├── risk_agent.py
│   └── recommendation_agent.py
├── data/
│   └── mock_data.py
├── orchestrator.py
├── dashboard.py
├── requirements.txt
└── README.md
```

## 🔮 Future Scope
- Replace mock data with live BigQuery streaming pipeline
- Deploy ForecastingAgent as a GPU-served NVIDIA NIM endpoint
- Add a fourth "Pricing Agent" for dynamic pricing decisions
- Multi-tenant dashboard for different business units
