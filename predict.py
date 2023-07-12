import pickle
import torch
from cog import BasePredictor, Input
from flair.data import Sentence
from flair.models import SequenceTagger
from typing import Any
import base64
import lz4
# TODO: Use safe pickle?


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # TODO: Better to keep the model in the docker
        #self.tagger = SequenceTagger.load("flair/pos-english")
        self.tagger = SequenceTagger.load("pos-english/pytorch_model.bin")

    @staticmethod
    def str_to_sentence(text: str) -> Sentence:
        assert isinstance(text, str)
        return Sentence(text)

    def predict(self,
            sentences_json: str = Input(description="JSON of sentence strings (or individual sentence string) to POS tag")
    ) -> str:
        with torch.no_grad():
            sentences = json.loads(sentence)
            if isinstance(sentences, list):
                results = []
                for sentence in sentences:
                    sentence = Predictor.str_to_sentence(sentence)
                    self.tagger.predict(sentence)
                results.append(sentence)
            else:
                sentence = Predictor.str_to_sentence(sentence)
                results = sentence
            return base64.b64encode(lz4.frame.compress(pickle.dumps(results)))
