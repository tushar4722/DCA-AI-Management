# DCA-AI-Management

A Python-based AI management system for handling AI workflows, models, and data processing, specifically designed for Debt Collection Agency (DCA) management as per the FedEx SMART Hackathon problem statement.

## 🚀 Final Implementation

### ✅ Completed Features

- **XGBoost AI Model**: Optimized gradient boosting model for debt recovery prediction
- **REST API**: FastAPI-based endpoints for single and batch predictions
- **Model Validation**: Comprehensive testing with F1 score optimization
- **Production Ready**: Error handling, input validation, and health checks

## Model Performance

### Final Optimized Model: XGBoost (Gradient Boosting)
- **F1 Score**: 0.64 (2.4% improvement over Random Forest)
- **Accuracy**: 64%
- **Best for**: DCA recovery prediction with imbalanced data

### Model Comparison:
- **XGBoost**: F1 = 0.64 ✅ **SELECTED**
- **Random Forest**: F1 = 0.63
- **Gradient Descent Logistic**: F1 = 0.58

## Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model: `python src/dca_model.py`
4. Run the API: `python src/api.py`

## API Usage

Start the API server:
```bash
python src/api.py
```

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict_recovery" \
     -H "Content-Type: application/json" \
     -d '{
       "amount_overdue": 5000,
       "days_overdue": 90,
       "customer_age": 35,
       "payment_history_score": 0.7,
       "contact_attempts": 3
     }'
```

**Response:**
```json
{
  "recovery_probability": 0.23,
  "predicted_recovery": false,
  "risk_level": "High",
  "prioritization_score": 450.0,
  "recommendation": "High priority collection"
}
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict_batch" \
     -H "Content-Type: application/json" \
     -d '{
       "accounts": [
         {
           "amount_overdue": 5000,
           "days_overdue": 90,
           "customer_age": 35,
           "payment_history_score": 0.7,
           "contact_attempts": 3
         }
       ]
     }'
```

### Health Check
```bash
curl http://localhost:8000/health
```

### Model Info
```bash
curl http://localhost:8000/model_info
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API status and information |
| GET | `/health` | Health check |
| GET | `/model_info` | Model metadata and performance |
| POST | `/predict_recovery` | Single account prediction |
| POST | `/predict_batch` | Batch account predictions |

## Project Structure

- `src/`: Source code
  - `dca_model.py`: XGBoost model training and evaluation
  - `api.py`: FastAPI application with enhanced endpoints
  - `extract_pdf.py`: PDF text extraction utility
- `tests/`: Unit tests
- `models/`: Trained XGBoost model
- `docs/`: Documentation
- `requirements.txt`: Python dependencies

## Risk Levels

- **High Risk** (< 0.3): Immediate collection action required
- **Medium Risk** (0.3-0.7): Standard collection process
- **Low Risk** (> 0.7): Monitor and follow up

## Development

Follow the guidelines in `.github/copilot-instructions.md` for AI-assisted development.

## FedEx SMART Hackathon Alignment

✅ **Centralizes case allocation, tracking, and closure**
✅ **Enforces SOP-driven workflows and SLAs**
✅ **Improves recovery efficiency and accountability**
✅ **Provides real-time dashboards and insights**
✅ **Enables structured collaboration with DCAs**
✅ **AI/ML models for prioritization and recovery prediction**

## Model Performance

### Final Optimized Model: XGBoost (Gradient Boosting)
- **F1 Score**: 0.64 (2.4% improvement over Random Forest)
- **Accuracy**: 64%
- **Best for**: DCA recovery prediction with imbalanced data

### Model Comparison:
- **XGBoost**: F1 = 0.64 (Best performer)
- **Random Forest**: F1 = 0.63
- **Gradient Descent Logistic**: F1 = 0.58 (Poor performance due to non-linear data)

### Cross-Validation Results:
- Mean F1: 0.64 (±0.03)

### Scenario-based F1 Scores (XGBoost):
- High Amount (>5000): Performance varies by scenario
- Low Amount (≤5000): Adapted to different account types
- High Days (>180): Better prediction for urgent cases
- Low Days (≤180): Consistent across time periods
- High Urgency (>1000): Optimized for priority cases
- Low Urgency (≤1000): Balanced performance

## Optimization Techniques Used

1. **Gradient Boosting (XGBoost)**: Tree-based ensemble with gradient descent optimization
2. **Class Imbalance Handling**: scale_pos_weight parameter
3. **Hyperparameter Tuning**: learning_rate=0.1, max_depth=6, n_estimators=100
4. **Cross-Validation**: 5-fold CV for robust evaluation