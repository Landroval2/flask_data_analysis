from models.model import Model

from flair.models import SequenceTagger

from flair.data import Sentence


class SequenceTagger(Model):
    
    model = SequenceTagger.load('ner-fast')

    @staticmethod
    def to_dict(sentence, tag_type: str = None):
        labels = []
        entities = []

        if tag_type:
            entities = [span.to_dict() for span in sentence.get_spans(tag_type)]
            for span_dict in entities:
                span_dict['labels'] = str(span_dict['labels'])

        if sentence.labels:
            labels = [str(l.to_dict()) for l in sentence.labels]

        return {"text": sentence.to_original_text(), "labels": labels, "entities": entities}

    @classmethod
    def make_prediction(cls, input_text):
        
        sentence = Sentence(input_text)

        # # run classifier over sentence
        cls.model.predict(sentence)

        #extract text and its prediction
        prediction = cls.to_dict(sentence, tag_type='ner')

        return prediction