# Brings in Python’s built-in unit testing framework.
import unittest
# Imports your Flask application instance (app) from your project’s flask_app/app.py.
from flask_app.app import app


# Defines a test class FlaskAppTests which inherits from unittest.TestCase.
# This means all methods that start with test_ will be automatically recognized and executed as unit tests.
class FlaskAppTests(unittest.TestCase):

    # Runs once before all tests in the class.
    @classmethod
    def setUpClass(cls):
        # Creates a test client for the Flask app. This lets you send fake requests (GET/POST) without running the actual server.
        # Saves the test client so that all tests in the class can reuse it.
        cls.client = app.test_client()

    # A test method that checks the / (home page) route.
    def test_home_page(self):
        # Simulates a GET request to the home page.
        response = self.client.get('/')
        # Asserts the page loads successfully (200 OK).
        self.assertEqual(response.status_code, 200)
        # Ensures the returned HTML contains the correct <title> tag.
        # Notice the b'' prefix: the response is in bytes, so the string must also be bytes for comparison.
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    # Tests the /predict route.
    def test_predict_page(self):
        # Simulates submitting the form with a text input "I love this!".
        response = self.client.post('/predict', data=dict(text="I love this!"))
        # Confirms the request succeeded.
        self.assertEqual(response.status_code, 200)
        # Checks that the response contains either "Positive" or "Negative" in the returned HTML.
        # If neither word is found, it fails with the message: "Response should contain either 'Positive' or 'Negative'".
        self.assertTrue(
            b'Positive' in response.data or b'Negative' in response.data,
            "Response should contain either 'Positive' or 'Negative'"
        )

if __name__ == '__main__':
    unittest.main()