from models.model import Model

from flair.models import TextClassifier
from flair.data import Sentence


class ClassificationModel(Model):
    model = TextClassifier.load('sentiment')

    @classmethod
    def make_prediction(cls, input_text):
        
        sentence = Sentence(input_text)

        # # run classifier over sentence
        cls.model.predict(sentence)

        #extract text and its prediction
        text = sentence.to_plain_string()
        label = sentence.labels[0]
        prediction = {'phrase': text, 'tag': label.value, 'score': round(label.score, 2)}

        return prediction