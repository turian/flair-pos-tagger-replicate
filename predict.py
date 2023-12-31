# import base64
# import gzip
import json
from typing import Any

# import lz4
# import snappy
import torch
from cog import BasePredictor, Input
from flair.data import Sentence
from flair.models import SequenceTagger

# try:
#    import cPickle as pickle
# except:
#    import pickle


# TODO: Use safe pickle?


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.tagger = SequenceTagger.load("pos-english/pytorch_model.bin")
        if torch.cuda.is_available():
            self.tagger = self.tagger.to("cuda")

    @staticmethod
    def str_to_sentence(text: str) -> Sentence:
        assert isinstance(text, str)
        return Sentence(text)

    def predict(
        self,
        sentences_json: str = Input(
            description="JSON of list sentence strings, to POS tag and return list of JSON dicts of flair.Sentence"
        ),
        # compression: str = Input(
        #            description="Compression to use: none (default) / snappy (unimplemented) / lz4 (unimplemented) / gzip",
        #            default="none",
        #        ),
    ) -> str:
        with torch.no_grad():
            sentences = json.loads('"i love berlin"')
            sentences = json.loads(sentences_json)
            if isinstance(sentences, list):
                results = []
                for sentence in sentences:
                    sentence = Predictor.str_to_sentence(sentence)
                    self.tagger.predict(sentence)
                    results.append(sentence)
            #            elif isinstance(sentences, str):
            #                sentence = Predictor.str_to_sentence(sentences)
            #                results = sentence
            else:
                assert False, f"{type(sentences)}"

            def sentence_to_dict(sentence: Sentence):
                sentence_dict = sentence.to_dict()
                sentence_dict["token positions"] = [
                    (token.start_position, token.end_position) for token in sentence
                ]
                assert len(sentence_dict["token positions"]) == len(
                    sentence_dict["all labels"]
                )
                return sentence_dict

            results_dict = [sentence_to_dict(sentence) for sentence in results]
            return json.dumps(results_dict)
            """
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
            """
