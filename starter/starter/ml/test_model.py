"""
This file contains unit tests for the model.py file.
"""
import unittest

from starter.ml.model import train_model, inference, compute_model_metrics
from sklearn.ensemble import RandomForestClassifier


class MyTestCase(unittest.TestCase):
    def test_train_model_type(self):
        """
        This function tests the train_model function.
        """
        # Create dummy data
        X_train = [[1, 2], [3, 4], [5, 6]]
        y_train = [0, 1, 0]

        # Train the model
        model = train_model(X_train, y_train)

        # Test if the trained model is a Random Forest Classifier.
        self.assertEqual(type(model), type(RandomForestClassifier()))

    def test_model_metrics_1(self):
        """
        This function tests the compute_model_metrics function.
        """
        # Create dummy data
        y = [0, 1, 0, 1]
        preds = [0, 0, 0, 0]

        # Compute the model metrics
        precision, recall, fbeta = compute_model_metrics(y, preds)

        # Test if the model metrics are correct.
        self.assertEqual(precision, 1.0)  # zero_division=1
        self.assertEqual(recall, 0.0)
        self.assertEqual(fbeta, 0.0)

    def test_model_metrics_2(self):
        """
        This function tests the compute_model_metrics function.
        """
        # Create dummy data
        y = [0, 1, 0, 1]
        preds = [1, 1, 1, 1]

        # Compute the model metrics
        precision, recall, fbeta = compute_model_metrics(y, preds)

        # Test if the model metrics are correct.
        self.assertEqual(precision, 0.5)
        self.assertEqual(recall, 1.0)
        self.assertEqual(fbeta, 0.6666666666666666)

    def test_inference(self):
        """
        This function tests the inference function with a Random Forest Classifier.
        """
        # Create dummy data
        X = [[1, 2], [3, 4], [5, 6]]
        model = RandomForestClassifier()
        model.fit(X, [0, 1, 0])

        # Run inference
        preds = inference(model, X)

        # Test if the inference is correct.
        self.assertEqual(preds.tolist(), [0, 1, 0])


if __name__ == '__main__':
    unittest.main()
