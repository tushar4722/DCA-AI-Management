# DCA-AI-Management Copilot Instructions

## Project Overview
DCA-AI-Management is a Python-based project focused on managing AI-related components and workflows for Debt Collection Agency (DCA) operations, specifically addressing the FedEx SMART Hackathon challenge to reimagine DCA management through digital and AI solutions.

## Architecture
- **Major Components**: Core modules in `src/` (e.g., `dca_model.py` for AI recovery prediction, `api.py` for FastAPI serving), tests in `tests/`, docs in `docs/`.
- **Service Boundaries**: Modular structure with separate directories for source, tests, and documentation; API layer for external integrations.
- **Data Flows**: Data processing handled via pandas and scikit-learn; AI models via RandomForest for recovery prediction.
- **Structural Decisions**: Standard Python project layout for maintainability and scalability, with models saved in `models/` directory.

## Developer Workflows
- **Build Process**: Install dependencies with `pip install -r requirements.txt`.
- **Testing**: Run tests with `python -m unittest discover tests/`.
- **Debugging**: Use Python debugger (pdb) or IDE tools; run main script with `python src/main.py`.
- **Key Commands**: `python src/dca_model.py` to train the AI model; `python src/api.py` to start the API server.

## Project Conventions
- **Code Style**: Follow PEP 8; use type hints where possible.
- **Naming Patterns**: Descriptive names for AI components (e.g., `dca_model.py`, `predict_recovery` endpoint).
- **File Organization**: `src/` for code, `tests/` for unit tests, `docs/` for documentation, `models/` for trained models.

## Integration Points
- **External Dependencies**: Listed in `requirements.txt` (numpy, pandas, scikit-learn, openai, fastapi, uvicorn).
- **Cross-Component Communication**: API-based with FastAPI for web services; model predictions via REST endpoints.

## Key Files/Directories
- `.github/copilot-instructions.md`: This file for AI guidance.
- `src/dca_model.py`: AI model for predicting debt recovery probability.
- `src/api.py`: FastAPI application for model serving.
- `models/`: Directory for saved trained models.
- Future: `src/data_processor.py`, `src/workflow_manager.py` as project develops.