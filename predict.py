import base64
import gzip
import json
import pickle
from typing import Any

# import lz4
# import snappy
import torch
from cog import BasePredictor, Input
from flair.data import Sentence
from flair.models import SequenceTagger

# TODO: Use safe pickle?


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        # TODO: Better to keep the model in the docker
        # self.tagger = SequenceTagger.load("flair/pos-english")
        self.tagger = SequenceTagger.load("pos-english/pytorch_model.bin")

    @staticmethod
    def str_to_sentence(text: str) -> Sentence:
        assert isinstance(text, str)
        return Sentence(text)

    def predict(
        self,
        sentences_json: str = Input(
            description="JSON of sentence strings (or individual sentence string) to POS tag"
        ),
        compression: str = Input(
            description="Compression to use: none (default) / snappy (unimplemented) / lz4 (unimplemented) / gzip",
            default="none",
        ),
    ) -> str:
        with torch.no_grad():
            sentences = json.loads(sentences_json)
            if isinstance(sentences, list):
                results = []
                for sentence in sentences:
                    sentence = Predictor.str_to_sentence(sentence)
                    self.tagger.predict(sentence)
                results.append(sentence)
            elif isinstance(sentences, str):
                sentence = Predictor.str_to_sentence(sentences)
                results = sentence
            else:
                assert False, f"{type(sentences)}"
            pkl = pickle.dumps(results)
            if compression == "none":
                pkl = pkl
            #            elif compression == "snappy":
            #                pkl = snappy.compress(pkl)
            elif commpression == "gzip":
                pkl = gzip.compress(pkl)
            else:
                assert False, f"Unknown compression: {compression}"
            return base64.b64encode(pkl)
