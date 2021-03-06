import unittest
from app import app


class ApiRoutesTest(unittest.TestCase):
    def setUp(self):
        # Create a test app every test case can use.
        self.test_app = app.test_client()

    def assert_route(self, response):
        # Check for error.
        self.assertEquals(response.status, "200 OK")

    def make_predict(self, body):

        return self.test_app.post(url='/predict', json=body)


class TestPredict(ApiRoutesTest):

    def test_predict_route(self):

        json_body = {
            'input_text': 'George Washington went to Washington.',
            'model_type': 'tagger'}

        response = self.test_app.post("/predict", json=json_body)
        self.assert_route(response)
    
    def test_sequence_tagger_valid(self):

        json_body = {
            'input_text': 'George Washington went to Washington.',
            'model_type': 'tagger'}

        response = self.test_app.post("/predict", json=json_body)
        