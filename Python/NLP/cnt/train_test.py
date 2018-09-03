"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

import random
from .annotate import annotate_designs, extract_string_from_annotation


def train_test_annotate(entities, extract_entities_from_designs, annotated_designs):
    """
    specific train test split on disjoint sets of names
    """
    train_size=0.75
    train_entities = []
    test_entities = []
    # train estimator with given labels
    for list_of_entities in entities.values():
        random.shuffle(list_of_entities)
        n_training_samples = int(train_size*len(list_of_entities))
        train_entities += list_of_entities[:n_training_samples] 
        test_entities += list_of_entities[n_training_samples:]
    # proof which names are part of dataframe train_entities
    def has_train_entity(list_of_entities):
        for entity in list_of_entities:
            if entity in train_entities:
                return True
        return False
    entity_is_in_train = extract_entities_from_designs.map(has_train_entity)
    train_designs = annotated_designs[entity_is_in_train]
    test_designs = annotated_designs[~entity_is_in_train]
    # create disjoint train and test subsets for fitting the estimator
    X_train = train_designs.DesignEng
    y_train = train_designs.annotations
    X_test = test_designs.DesignEng
    y_test = test_designs.annotations
    return X_train, y_train, X_test, y_test