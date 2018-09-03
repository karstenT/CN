"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

import spacy
import random
from sklearn.base import BaseEstimator, ClassifierMixin
from tqdm import tqdm
import pandas as pd


label_object = 'OBJECT'
label_animal = 'ANIMAL'
label_plant = 'PLANT'


def batches(list_, batch_size):
    """
    Divides a list into several packages whose
    size is determined by the batch_size.

    Parameters
    ---------------

    list_ : list
        The input list
    
    batch_size: int
        The size of the batches
    """
    for i in range(0, len(list_), batch_size):
        yield list_[i:i + batch_size]


class DesignEstimator(BaseEstimator):
    def __init__(self, n_rep, learning_rate=0.001, batch_size=100):
        self.nlp = spacy.load('en')
        self.n_rep = n_rep
        self.learning_rate = learning_rate
        self.batch_size = batch_size
    
    def fit(self, X, y):
        """
        Fits the model / the estimator

        Parameters
        ---------------

        X: list
            list of input sentences
        
        y: list
            the corresponding annotations to the sentences
            e.g. [[(0, 6, 'PERSON')], [(0, 9, 'PERSON')]]
        """
        self.nlp = spacy.load('en')
        zipped = zip(X["DesignEng"],y)
        train_data = list(zipped)        

        other_pipes = [pipe for pipe in self.nlp.pipe_names if pipe != 'ner']
        # train an additional entity type
        if 'ner' not in self.nlp.pipe_names:
            ner = self.nlp.create_pipe('ner')
            self.nlp.add_pipe(ner)
        else:
            ner = self.nlp.get_pipe('ner')
        ner.add_label(label_object)
        ner.add_label(label_animal)
        ner.add_label(label_plant)
        with self.nlp.disable_pipes(*other_pipes):
            optimizer = self.nlp.begin_training()
            optimizer.alpha = self.learning_rate
            for iterations in range(self.n_rep):
                random.shuffle(train_data)
                for batch in list(batches(train_data, self.batch_size)):
                    raw_text, entity_offsets = zip(*batch)
                    self.nlp.update(raw_text,
                                    [{'entities' : ent} for ent in entity_offsets],
                                    sgd=optimizer)
    
    def predict_single_sentence(self, string, as_doc=False):
        """
        Predicts the annotation of the estimator
        on a single sentence

        Parameters
        -----------
        
        string: str
            The sentence that is to be predicted
        
        as_doc = Boolean
            Indicates if a list is a doc object or not;
            required for rendering in SpaCy

        """
        doc = self.nlp(string)
        if as_doc:
            return doc
        l = []
        for ent in doc.ents:
            l.append((ent.start_char, ent.end_char, ent.label_))
        return l

    def predict(self, X, as_doc=False):
        """
        Predicts the annotation of the estimator
        on a list of sentences
        
        Parameters
        -----------
        
        X: data frame
            data frame with designs and designIDs
        
        as_doc = Boolean
            Indicates if a list is a doc object or not;
            required for rendering in SpaCy
        """
        predictions = X["DesignEng"].map(lambda x: self.predict_single_sentence(x, as_doc=as_doc))
        X_pred = pd.DataFrame({"DesignID": X["DesignID"], "y": predictions})
        return X_pred
		
    def predict_single_sentence_clear(self, string, as_doc=False):
        """
        Predicts the annotation of the estimator
        on a single sentence

        Parameters
        -----------
        
        string: str
            The sentence that is to be predicted
        
        as_doc = Boolean
            Indicates if a list is a doc object or not;
            required for rendering in SpaCy

        """
        doc = self.nlp(string)
        if as_doc:
            return doc
        l = []
        for ent in doc.ents:
            l.append((ent.text, ent.label_))
        return l

    def predict_clear(self, X, as_doc=False):
        """
        Predicts the annotation of the estimator
        on a list of sentences
        
        Parameters
        -----------
        
        X: data frame
            data frame with designs and designIDs
        
        as_doc = Boolean
            Indicates if a list is a doc object or not;
            required for rendering in SpaCy
        """
        predictions = X["DesignEng"].map(lambda x: self.predict_single_sentence_clear(x, as_doc=as_doc))
        X_pred = pd.DataFrame({"DesignID": X["DesignID"], "y": predictions})
        return X_pred


class RelationExtractor(BaseEstimator, ClassifierMixin):
    NONEXISTINGRELATION = "nonexisting_relation"
    KEY = "y"

    def __init__(self, pipeline):
        self.pipeline = pipeline   
    
    def fit(self, X, y):
        """
        fits the model
        
        Parameters
        ----------
        
        X: list of lists of Feature objects
        y: list of lists of (subj, relation_class_label, obj)
        """
        X_features = []
        y_for_classification = []
        for list_of_features, list_of_annotations in zip(X["y"], y["y"]):
            dict_of_annotations = {(subj, obj) : label for subj, _, label, obj, _ in list_of_annotations}
            for feature in list_of_features:
                label = dict_of_annotations.get((feature.subj.text, feature.obj.text), self.NONEXISTINGRELATION)
                y_for_classification.append(label)
                X_features.append(feature)
        
        self.pipeline.fit(X_features, y_for_classification)
        
        return self
    
    def predict(self, X):
        """
        predicts the models' output for a list of sentences
        
        Parameters
        ----------
        
        X: list of lists of Feature objects
        """
        trans = X[self.KEY].map(self.predict_single)
        return pd.DataFrame({"DesignID": X["DesignID"], "y": trans})

    def predict_single(self, x):
        """
        predicts the models' output for a single sentence
        
        Parameters
        ----------
        
        X: list of Feature objects
        """
        if len(x) == 0:
            return []
        
        list_of_predicted_relations = self.pipeline.predict(x)
        
        prediction = []
        for feature, rel in zip(x, list_of_predicted_relations):
            if rel != self.NONEXISTINGRELATION:
                prediction.append((feature.subj.text, feature.subj.label_, rel, feature.obj.text, feature.obj.label_))
        return prediction