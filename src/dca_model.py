import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import joblib

class DCAModel:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.weights = None
        self.bias = None

    def generate_synthetic_data(self, n_samples=1000):
        """Generate synthetic data for overdue accounts."""
        np.random.seed(42)
        data = {
            'account_id': range(1, n_samples + 1),
            'amount_overdue': np.random.uniform(100, 10000, n_samples),
            'days_overdue': np.random.randint(30, 365, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'payment_history_score': np.random.uniform(0, 1, n_samples),
            'contact_attempts': np.random.randint(0, 10, n_samples),
            'recovered': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% recovery rate
        }
        return pd.DataFrame(data)

    def sigmoid(self, z):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-z))

    def fit_gradient_descent_mini_batch(self, X, y, learning_rate=0.01, epochs=1000, batch_size=32, lambda_reg=0.001):
        """Train logistic regression using mini-batch gradient descent with learning rate decay."""
        X_scaled = self.scaler.fit_transform(X)
        y_array = y.values if hasattr(y, 'values') else y  # Convert to numpy array
        m, n = X_scaled.shape
        self.weights = np.zeros(n)
        self.bias = 0

        n_batches = m // batch_size

        for epoch in range(epochs):
            # Learning rate decay
            lr = learning_rate / (1 + 0.01 * epoch)

            # Shuffle data
            indices = np.random.permutation(m)
            X_shuffled = X_scaled[indices]
            y_shuffled = y_array[indices]

            for batch in range(n_batches):
                start = batch * batch_size
                end = start + batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward pass
                z = np.dot(X_batch, self.weights) + self.bias
                y_pred = self.sigmoid(z)

                # Compute gradients with L2 regularization
                dw = (1/batch_size) * (np.dot(X_batch.T, (y_pred - y_batch)) + lambda_reg * self.weights)
                db = (1/batch_size) * np.sum(y_pred - y_batch)

                # Update parameters
                self.weights -= lr * dw
                self.bias -= lr * db

            # Early stopping
            if epoch > 50 and np.linalg.norm(dw) < 1e-4:
                break

    def predict_proba_gd(self, X):
        """Predict probabilities using gradient descent model."""
        X_scaled = self.scaler.transform(X)
        z = np.dot(X_scaled, self.weights) + self.bias
        return self.sigmoid(z)

    def predict_gd(self, X, threshold=0.5):
        """Predict classes using gradient descent model."""
        proba = self.predict_proba_gd(X)
        return (proba >= threshold).astype(int)
        """Generate synthetic data for overdue accounts."""
        np.random.seed(42)
        data = {
            'account_id': range(1, n_samples + 1),
            'amount_overdue': np.random.uniform(100, 10000, n_samples),
            'days_overdue': np.random.randint(30, 365, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'payment_history_score': np.random.uniform(0, 1, n_samples),
            'contact_attempts': np.random.randint(0, 10, n_samples),
            'recovered': np.random.choice([0, 1], n_samples, p=[0.7, 0.3])  # 30% recovery rate
        }
        return pd.DataFrame(data)

    def preprocess_data(self, df):
        """Preprocess the data for modeling."""
        # Feature engineering
        df['amount_per_day'] = df['amount_overdue'] / df['days_overdue']
        df['urgency_score'] = df['days_overdue'] * df['amount_overdue'] / 1000

        features = ['amount_overdue', 'days_overdue', 'customer_age',
                   'payment_history_score', 'contact_attempts', 'amount_per_day', 'urgency_score']
        X = df[features]
        y = df['recovered']
        return X, y

    def cross_validate_model(self, X, y, cv=5):
        """Perform cross-validation and return F1 scores."""
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        f1_scores = cross_val_score(model, X, y, cv=cv, scoring='f1_weighted')
        return f1_scores

    def train_model(self, X_train, y_train):
        """Train the Random Forest model."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(X_train, y_train)

    def train_model_xgb(self, X_train, y_train):
        """Train using XGBoost (gradient boosting)."""
        self.model_xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            scale_pos_weight=len(y_train) / sum(y_train),  # Handle imbalance
            random_state=42
        )
        self.model_xgb.fit(X_train, y_train)

    def train_model_gd(self, X_train, y_train, learning_rate=0.01, epochs=1000):
        """Train using mini-batch gradient descent."""
        self.fit_gradient_descent_mini_batch(X_train, y_train, learning_rate, epochs)

    def evaluate_model_xgb(self, X_test, y_test):
        """Evaluate XGBoost model."""
        y_pred = self.model_xgb.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        return accuracy, f1, report

    def evaluate_model_gd(self, X_test, y_test):
        """Evaluate gradient descent model."""
        y_pred = self.predict_gd(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        return accuracy, f1, report

    def evaluate_model(self, X_test, y_test):
        """Evaluate the Random Forest model performance."""
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred)
        return accuracy, f1, report

    def test_scenarios_gd(self, X_test, y_test):
        """Test GD model on different scenarios and calculate F1 scores."""
        scenarios = {
            'High Amount': X_test[X_test['amount_overdue'] > 5000],
            'Low Amount': X_test[X_test['amount_overdue'] <= 5000],
            'High Days': X_test[X_test['days_overdue'] > 180],
            'Low Days': X_test[X_test['days_overdue'] <= 180],
            'High Urgency': X_test[X_test['urgency_score'] > 1000],
            'Low Urgency': X_test[X_test['urgency_score'] <= 1000]
        }
        
        results = {}
        for scenario, X_scenario in scenarios.items():
            if len(X_scenario) > 0:
                y_scenario = y_test.loc[X_scenario.index]
                y_pred = self.predict_gd(X_scenario)
                f1 = f1_score(y_scenario, y_pred, average='weighted')
                results[scenario] = f1
        return results

    def save_model(self, filepath):
        """Save the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet.")
        joblib.dump(self.model, filepath)

    def load_model(self, filepath):
        """Load a trained model."""
        self.model = joblib.load(filepath)

if __name__ == "__main__":
    dca_model = DCAModel()

    # Generate and preprocess data
    df = dca_model.generate_synthetic_data()
    X, y = dca_model.preprocess_data(df)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Cross-validate
    f1_scores = dca_model.cross_validate_model(X, y)
    print(f"Cross-Validation F1 Scores: {f1_scores}")
    print(f"Mean F1 Score: {f1_scores.mean():.2f} (+/- {f1_scores.std() * 2:.2f})")

    # Train Random Forest model
    print("=== Random Forest Model ===")
    dca_model.train_model(X_train, y_train)
    accuracy_rf, f1_rf, report_rf = dca_model.evaluate_model(X_test, y_test)
    print(f"RF Model Accuracy: {accuracy_rf:.2f}")
    print(f"RF Model F1 Score (Weighted): {f1_rf:.2f}")
    print("RF Classification Report:")
    print(report_rf)

    # Train Gradient Descent model
    print("\n=== Gradient Descent Logistic Regression ===")
    dca_model.train_model_gd(X_train, y_train, learning_rate=0.1, epochs=2000)
    accuracy_gd, f1_gd, report_gd = dca_model.evaluate_model_gd(X_test, y_test)
    print(f"GD Model Accuracy: {accuracy_gd:.2f}")
    print(f"GD Model F1 Score (Weighted): {f1_gd:.2f}")
    print("GD Classification Report:")
    print(report_gd)

    # Train XGBoost model
    print("\n=== XGBoost (Gradient Boosting) ===")
    dca_model.train_model_xgb(X_train, y_train)
    accuracy_xgb, f1_xgb, report_xgb = dca_model.evaluate_model_xgb(X_test, y_test)
    print(f"XGB Model Accuracy: {accuracy_xgb:.2f}")
    print(f"XGB Model F1 Score (Weighted): {f1_xgb:.2f}")
    print("XGB Classification Report:")
    print(report_xgb)

    # Compare all models
    print(f"\n=== Model Comparison ===")
    print(f"Random Forest F1: {f1_rf:.3f}")
    print(f"Gradient Descent F1: {f1_gd:.3f}")
    print(f"XGBoost F1: {f1_xgb:.3f}")
    print(f"GD vs RF: {((f1_gd - f1_rf) / f1_rf * 100):.1f}%")
    print(f"XGB vs RF: {((f1_xgb - f1_rf) / f1_rf * 100):.1f}%")

    # Test scenarios with best model
    best_model = "RF"
    best_f1 = f1_rf
    if f1_xgb > best_f1:
        best_model = "XGB"
        best_f1 = f1_xgb
    if f1_gd > best_f1:
        best_model = "GD"
        best_f1 = f1_gd

    print(f"\nBest model: {best_model} with F1: {best_f1:.3f}")

    # Save the best model
    if best_model == "XGB":
        print("XGBoost performs best - saving XGB model")
        dca_model.model = dca_model.model_xgb
        dca_model.save_model('models/dca_recovery_model.pkl')
    elif best_model == "GD":
        print("Gradient Descent performs best - saving GD model")
        dca_model.save_model('models/dca_recovery_model.pkl')
    else:
        print("Random Forest performs best - keeping RF model")