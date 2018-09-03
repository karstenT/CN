"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

from collections import namedtuple
from sklearn.base import TransformerMixin, BaseEstimator
from cnt.io import load_entities_from_db
from cnt.annotate import annotate_designs
from cnt.model import DesignEstimator
import spacy
from functools import lru_cache
import pandas as pd

NER = namedtuple("NER", ["doc", "subj", "obj"])
mysql_connection = "mysql://cnt:rJnW6m7kZR@localhost:3306/thrakien_cnt"


def make_my_estimator(X):
    entities = {
        "PERSON": load_entities_from_db("nlp_list_person", mysql_connection),
        "OBJECT": load_entities_from_db("nlp_list_obj", mysql_connection),
        "ANIMAL": load_entities_from_db("nlp_list_animal", mysql_connection),
        "PLANT": load_entities_from_db("nlp_list_plant", mysql_connection)
    }
    annotated_designs = annotate_designs(entities, X)
    annotated_designs = annotated_designs[
        annotated_designs.annotations.map(len) > 0]
    my_estimator = DesignEstimator(3)
    my_estimator.fit(annotated_designs, annotated_designs.annotations)
    return my_estimator


class SingleSentenceTransformerMixin:
    def transform(self, X):
        """
        transforms a list of sentences into NER objects
        with sentence = spacy.doc, subj and obj = spacy.span
        
        Parameters
        -----------
        
        X: list of strings

        Returns
        -------

        list of lists of NER objects
        """
        trans = X[self.KEY].map(self.transform_single_sentence)
        return pd.DataFrame({"DesignID": X["DesignID"], "y": trans})


class NERTransformer(SingleSentenceTransformerMixin, TransformerMixin, BaseEstimator):
    KEY = "DesignEng"
    def fit(self, X, y):
        """
        fits the model
        
        Parameters
        -----------
        
        X: list of designs
        y: list of lists of (subj, relation_class_label, obj)
        """
        self.my_estimator = make_my_estimator(X)
        return self
    


    def transform_single_sentence(self, x):
        """
        transforms a sentence into a NER object
        with sentence = spacy.doc, subj and obj = spacy.span
        
        Parameters
        -----------
        
        x: string
        """
        doc = self.my_estimator.predict_single_sentence(x, as_doc=True)
        sent_subj_obj = []
        for subj in filter(lambda span: span.label_ == 'PERSON', doc.ents):
            for obj in filter(lambda span: span.label_ == 'OBJECT', doc.ents):
                sent_subj_obj.append(NER(doc, subj, obj))
        return sent_subj_obj


def path(subj, obj):
    """
    determines the least common ancestor of two nodes
    and prints the whole path between them

    Parameters
    -----------
    
    subj: token
        word in the sentence / node in the tree
        to start the path
    obj: token
        word in the sentence / node in the tree
        to end the path

    Returns
    -------

    list of spacy.Token
    """
    up_from_obj = []
    up_from_subj = []

    current_token = obj
    while True:
        up_from_obj.append(current_token)
        if current_token == current_token.head:
            break
        current_token = current_token.head
    up_from_obj = list(reversed(up_from_obj))

    current_token = subj
    while current_token not in up_from_obj and current_token != current_token.head:
        up_from_subj.append(current_token)
        current_token = current_token.head
    
    try:
        intersection = up_from_obj.index(current_token)
    except ValueError:  # current_token not in up_from_obj
        return []

    path = up_from_subj + up_from_obj[intersection:]
    
    return path


Feature = namedtuple("Feature", ["subj", "path", "obj", "doc", "verbs"])

class FeatureExtractor(SingleSentenceTransformerMixin, TransformerMixin, BaseEstimator):
    KEY = "y"
    def fit(self, X, y):
        return self
    
    def transform_single_sentence(self, x):
        """
        transforms a sentence into a Feature object
        with sentence = spacy.doc, subj and obj = spacy.span

        Parameters
        -----------

        X: list of NER objects
        """
        extracted_paths = []
        for ner in x:
            p = path(ner.subj.root, ner.obj.root)
            verbs = self.extract_verbs_single_sentence(p)
            extracted_paths.append(Feature(ner.subj, p, ner.obj, ner.doc, verbs))
        return extracted_paths

    def extract_verbs_single_sentence(self, p):
        verbs = []
        for token in p:
            if token.pos_ == "VERB":
                verbs.append(token.text)
        return verbs