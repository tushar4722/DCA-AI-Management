import unittest
from src.main import main
import requests
import time

class TestMain(unittest.TestCase):
    def test_main(self):
        # TODO: Add actual tests
        self.assertTrue(True)

class TestAPI(unittest.TestCase):
    def test_predict_recovery(self):
        try:
            url = "http://localhost:8000/predict_recovery"
            data = {
                "amount_overdue": 5000,
                "days_overdue": 90,
                "customer_age": 35,
                "payment_history_score": 0.7,
                "contact_attempts": 3
            }
            response = requests.post(url, json=data, timeout=5)
            self.assertEqual(response.status_code, 200)
            result = response.json()
            self.assertIn("recovery_probability", result)
            self.assertIn("predicted_recovery", result)
            self.assertIn("prioritization_score", result)
        except requests.exceptions.RequestException:
            self.skipTest("API server not running")

if __name__ == '__main__':
    unittest.main()