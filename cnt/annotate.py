"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

import re
import pandas as pd
import numpy as np

def annotate(sentence, label, entities):
    """
    Given the entities, annotate a sentence with a label.
    
    Parameters
    ----------
    
    sentence: str
        The sentence to be annotated.
        
    label: str
        The label, e.g. "PERSON", "ANIMAL", ...
        
    entities: list
        List of entities belonging to the label.
        E.g. ["Aphrodite", "Apollo", ...]
    """
    regex = r'\b' + '(' + "|".join(entities) + ')' + r'\b'
    occurences = re.finditer(regex, sentence)
    annotation = [(match.start(), match.end(), label) for match in occurences]
    return annotation


def annotate_single_design(entities, design):
    """
    Given the entities, annotate a concrete design.
    
    Parameters
    ----------

    entities: dict
        Dictionary whose keys are the labels and whose values
        are the corresponding lists of entities.

    design: str
        The input sentence.
    """
    annotations = []
    for label, entities in entities.items():
        annotations += annotate(design, label, entities)
    annotations = sorted(annotations, key = lambda x : x[0])
    return annotations


def annotate_designs(entities, designs):
    """
    Given the entities, annotate a list of design.
    
    Parameters
    ----------

    entities: dict
        Dictionary whose keys are the labels and whose values
        are the corresponding lists of entities.

    design: list
        List of sentences.
    """

    annotated_designs = pd.DataFrame({
        "DesignEng": designs["DesignEng"], 
        "DesignID": designs["DesignID"],
        "annotations": list(map(lambda x: 
        annotate_single_design(entities, x), designs["DesignEng"]))})
    return annotated_designs

def extract_string_from_annotation(annotations, design):
    """
    Given the annotations, extract the corresponding string
    from a sentence.

    Parameters
    -----------

    annotations: list of list of triples
        E.g. y = [[(0, 5, "PERSON"), (10, 15, "OBJECT")], ...]

    design: str
        The input sentence.
    """
    list_of_strings = []
    for (start, stop, label) in annotations:
        list_of_strings.append(design[start:stop])
    return list_of_strings