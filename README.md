🚀 DCA-AI-Management

AI-Driven Debt Collection Management System

An end-to-end AI-powered Debt Collection Agency (DCA) Management platform that predicts recovery probability, prioritizes accounts, and enables operational decision-making through a complete ML pipeline, REST APIs, and a working UI.

🔹 Built for the FedEx SMART Hackathon
🔹 Designed for real-world deployability & scalability

🏆 Why This Solution Stands Out (For Judges)

✔ End-to-End ML Lifecycle (Data → Model → API → UI)
✔ Business-aligned AI prioritization logic
✔ Handles imbalanced real-world data
✔ Production-ready backend with FastAPI
✔ Mandatory Basic Working UI included
✔ Clear alignment with DCA & logistics use cases

🎯 Problem Statement

Debt Collection Agencies face challenges in:

Identifying high-priority recovery cases

Managing large volumes of accounts

Ensuring SLA compliance

Reducing manual decision-making

💡 Our Solution

An AI-based decision support system that:

Predicts debt recovery probability

Assigns risk & priority levels

Supports single & batch predictions

Provides a simple UI for operational use

🧠 System Architecture (End-to-End Pipeline)
Raw Account Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
XGBoost Model Training
      ↓
Model Evaluation (F1 Score)
      ↓
Model Persistence
      ↓
FastAPI Inference Layer
      ↓
Web UI for DCA Agents


🔁 Machine Learning Pipeline
1️⃣ Data Ingestion

Account-level structured data

Numerical behavioral features

2️⃣ Preprocessing

Missing value handling

Feature scaling

Class imbalance correction (scale_pos_weight)

3️⃣ Feature Engineering

Urgency score

Priority ranking

Risk segmentation

4️⃣ Model Training

XGBoost (Gradient Boosting)

Hyperparameter tuning

5-fold cross-validation

5️⃣ Evaluation

Primary Metric: F1 Score

Accuracy & scenario-based validation

6️⃣ Deployment

Trained model stored in /models

Served via FastAPI

Consumed by UI

🤖 Model Performance
✅ Final Selected Model: XGBoost
| Metric       | Score    |
| ------------ | -------- |
| F1 Score     | **0.64** |
| Accuracy     | 64%      |
| CV Stability | ±0.03    |

Model Comparison
| Model               | F1 Score            |
| ------------------- | ------------------- |
| **XGBoost**         | **0.64 (Selected)** |
| Random Forest       | 0.63                |
| Logistic Regression | 0.58                |


📌 Reason for Selection:
Best balance of precision & recall on imbalanced recovery data

🚦 Risk & Priority Logic
| Recovery Probability | Risk Level | Action               |
| -------------------- | ---------- | -------------------- |
| < 0.3                | High       | Immediate collection |
| 0.3 – 0.7            | Medium     | Standard follow-up   |
| > 0.7                | Low        | Monitor              |

🖥️ Basic Working UI (MANDATORY ✔)

A lightweight operational UI for DCA agents.

UI Capabilities

Enter customer debt details

Trigger AI prediction

View:

Recovery probability

Risk level

Priority recommendation

Tech Stack

HTML

CSS

JavaScript (Fetch API)

FastAPI backend

📁 Location:

ui/
 ├── index.html
 ├── style.css
 └── script.js

🔗 API Endpoints
| Method | Endpoint            | Purpose           |
| ------ | ------------------- | ----------------- |
| GET    | `/`                 | API status        |
| GET    | `/health`           | System health     |
| GET    | `/model_info`       | Model metadata    |
| POST   | `/predict_recovery` | Single prediction |
| POST   | `/predict_batch`    | Batch prediction  |

📦 Project Structure
DCA-AI-Management/
│
├── src/
│   ├── dca_model.py     # ML pipeline & training
│   ├── api.py           # FastAPI inference service
│   └── extract_pdf.py
│
├── models/
│   └── xgboost_model.pkl
│
├── ui/                  # Basic working UI
│
├── tests/
├── docs/
├── requirements.txt
└── README.md

⚙️ Installation & Run
git clone <repo-url>
cd DCA-AI-Management
pip install -r requirements.txt
python src/dca_model.py
python src/api.py


Open UI:

ui/index.html


🧪 Innovation & Future Scope

Integration with live DCA databases

Advanced dashboards (React / Power BI)

Automated SLA breach alerts

Multi-model ensemble learning

Role-based access control

✅ Final Compliance Checklist

✔ Code
✔ Model
✔ ML Pipeline
✔ REST APIs
✔ Basic Working UI
✔ Business relevance
✔ Deployment-ready design

🏁 Conclusion

DCA-AI-Management delivers a complete, practical, and scalable AI solution for debt recovery operations, combining machine learning, backend services, and a usable UI — fully aligned with hackathon expectations and real-world constraints.
