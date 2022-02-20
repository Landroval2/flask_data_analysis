from model import Model
from models.classification_model import ClassificationModel


class ModelFactory():
    @staticmethod
    def get_model(model_type: str) -> Model:
        if model_type == 'classification':
            return ClassificationModel
        else:
            pass