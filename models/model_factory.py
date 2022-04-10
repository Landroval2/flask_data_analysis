from models.model import Model
from models.classification import Classification
from models.sequence_tagger import SequenceTagger


class ModelFactory():
    @staticmethod
    def get_model(model_type: str) -> Model:

        if model_type == 'classification':
            return Classification

        elif model_type == 'tagger':
            return SequenceTagger

        else:
            raise ValueError('Unknown model type')
