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
        # Make a prediction
        return self.test_app.post(url='/predict', json=body)

class TestPredict(ApiRoutesTest):

    def test_sentence_tagger(self):

        json_body = {
            'input_text': 'George Washington went to Washington.',
            'model_type': 'tagger'}

        with app.test_client() as c:
            rv = c.post("/predict", json=json_body)
            json_data = rv.get_json()
            print(json_data)
            assert rv.status_code == 200