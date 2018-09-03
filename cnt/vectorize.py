"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

from sklearn.base import TransformerMixin, BaseEstimator
import numpy as np
import gensim
from gensim.models.doc2vec import TaggedDocument

class Doc2Str(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        """
        builds a list of words as input for vectorization
        
        Parameters
        ----------
        
        X: list of Feature objects
        """
        X_vect = [" ".join(map(str, x.doc)) for x in X]
        return X_vect

class Path2Str(TransformerMixin, BaseEstimator):
    def __init__(self, pos=False, dep=False):
        self.pos = pos
        self.dep = dep
    
    def fit(self, X, y):
        return self
    
    def transform_single(self, x):
        if self.pos and not self.dep:
            x_vect = " ".join(map(lambda t: t.text.replace(" ", "_") + "\\" + t.pos_, x.path))
        elif self.dep and not self.pos:
            x_vect = " ".join(map(lambda t: t.text.replace(" ", "_") + "\\" + t.dep_, x.path))
        elif self.pos and self.dep:
            x_vect = " ".join(map(lambda t: t.text.replace(" ", "_") + "\\" + t.pos_ + "\\" + t.dep_, x.path))
        else:
            x_vect = " ".join(map(str, x.path))
        return x_vect
            
    
    def transform(self, X):
        """
        builds a list of words as input for vectorization
        
        Parameters
        ----------
        
        X: list of Feature objects
        """
        return [self.transform_single(x) for x in X]

class Verbs2Str(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        return self
    
    def transform(self, X):
        """
        builds a list of words as input for vectorization
        
        Parameters
        ----------
        
        X: list of Feature objects
        """
        X_vect = [" ".join(map(str, x.verbs)) for x in X]
        return X_vect
    
class AveragedPath2Vec(TransformerMixin, BaseEstimator):
    # vector size on the en_core_web_sm = 384
    vec_dim = 384
    
    def fit(self, X, y):
        return self
    
    def transform_single(self, x):
        """
        Parameters:
        -----------
        
        x: single Feature object
        """
        path_vector = sum((token.vector for token in x.path), np.zeros(self.vec_dim))
        averaged_path_vector = path_vector / len(x.path) if len(x.path) > 0 else path_vector
        return averaged_path_vector
        
    def transform(self, X):
        """
        builds an averaged path vector
        
        Parameters
        ----------
        
        X: list of Feature objects
        """
        return [self.transform_single(x) for x in X]
    
class AveragedRest2Vec(TransformerMixin, BaseEstimator):
    # vector size on the en_core_web_sm = 384
    vec_dim = 384
    
    def fit(self, X, y):
        return self
    
    def transform_single(self, x):
        """
        Parameters:
        -----------
        
        x: single Feature object
        """
        path_vector = sum((token.vector for token in x.path), np.zeros(self.vec_dim))
        doc_vector = sum((token.vector for token in x.doc), np.zeros(self.vec_dim))
        averaged_vector = ((doc_vector - path_vector) / (len(x.doc) - len(x.path))
                           if (len(x.doc) - len(x.path)) > 0 else np.zeros(self.vec_dim))
        return averaged_vector
        
    def transform(self, X):
        """
        builds an averaged path vector
        
        Parameters
        ----------
        
        X: list of Feature objects
        """
        return [self.transform_single(x) for x in X]
        

class Doc2Vec(TransformerMixin, BaseEstimator):
    def fit(self, X, y):
        tokenized_sentences = []
        for x in X:
            tokenized_sentence = []
            for token in x.path:
                tokenized_sentence.append(str(token))
            tokenized_sentences.append(tokenized_sentence)
        
        tagged_documents = []
        i = 0
        for tokenized_sentence in tokenized_sentences:
            tagged_document = TaggedDocument(words = tokenized_sentence, tags=[i])
            tagged_documents.append(tagged_document)
            i += 1 

        self.doc2vec_model = gensim.models.doc2vec.Doc2Vec(alpha=0.001, min_alpha=0.0001, size=300,
                                                           min_count=2, iter=100, sample=0)
        self.doc2vec_model.build_vocab(tagged_documents)
        self.doc2vec_model.train(tagged_documents, 
                            total_examples=self.doc2vec_model.corpus_count, 
                            epochs=self.doc2vec_model.iter)
        
        return self
    
    def transform_single(self, x):
        """
        x: Feature object
        """
        transformed_path = self.doc2vec_model.infer_vector([str(token) for token in x.path])
        return transformed_path 
    
    def transform(self, X):
        """
        X: list of Feature objects
        """
        return [self.transform_single(x) for x in X]