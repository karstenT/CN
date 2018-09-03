"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

import pandas as pd

def load_designs():
    """
    loads sentences from CNT database from an SQL table
    """
    designs = pd.read_sql_table("Designs", "mysql://cnt:rJnW6m7kZR@localhost:3306/thrakien_cnt")
    english_designs = designs[["DesignID", "DesignEng"]]
    return english_designs

def load_crro_designs():
    """
    loads sentences from CRRO database from an SQL table
    """
    designs = pd.read_sql_table("designs_crro","mysql://cnt:rJnW6m7kZR@localhost:3306/external_designs")
    english_designs = [sentence for sentence in designs.Design_Eng if sentence is not None]
    english_designs = replace_left_right(english_designs)
    return english_designs

def load_ocre_designs():
    """
    loads sentences from OCRE database from an SQL table
    """
    designs = pd.read_sql_table("designs_ocre","mysql://cnt:rJnW6m7kZR@localhost:3306/thrakien_cnt")
    english_designs = designs[["DesignID", "DesignEng"]]
    return english_designs

def load_entities_from_db(table_name, con):
    """

    Returns
    -------

    List of entities. This list contains
    alternative names and capitalized entities.
    """
    entities = pd.read_sql_table(table_name, con)
    name = entities["name"]
    alternativenames = entities["alternativenames"]
    splitted_alternativenames = alternativenames.fillna("").str.split(",")
    flattened_alternativenames = sum(splitted_alternativenames, [])
    list_of_entities = name.append(pd.Series(flattened_alternativenames))
    return preprocess_entities(list_of_entities)


def load_entities_from_file(filepath):
    """
    opens a local file and generates a list

    also adds uppercase versions of all list entries

    Parameters
    -----------

    filepath : str
        The path to the local file
    """
    with open(filepath) as file:
        entities = file.read().split('\n')
    return preprocess_entities(entities)

def preprocess_entities(entities):
    entities = [entity.strip() for entity in entities]
    entities = [entity for entity in entities if len(entity) > 0]
    capitalized_entities = [entity.capitalize() for entity in entities]
    entities += capitalized_entities
    return entities

def replace_left_right_single_design(design):
    """
    preprocesses the data by replacing r. and l.

    Parameters
    -----------

    design: string
        the input sentence
    """
    a = design.strip()
    b = a.replace(" l.", " left")
    c = b.replace(" r.", " right")
    if not c.endswith("."):
        d = c + "."
    else:
        d = c
    return d

def replace_left_right_list_of_designs(designs):
    """
    Parameters
    ----------

    designs: list of strings
    """
    preprocessed_designs = []
    for design in designs:
        preprocessed_designs.append(replace_left_right_single_design(design))
    return preprocessed_designs

def replace_left_right(design):
    """
    Parameters
    ----------

    design: string or list of strings
    """
    if isinstance(design, str):
        return replace_left_right_single_design(design)
    elif isinstance(design, list):
        return replace_left_right_list_of_designs(design)
    elif isinstance(design, pd.DataFrame):
        res = design.copy()
        res["DesignEng"] = design["DesignEng"].map(replace_left_right_single_design)
        return res
    else:
        raise Exception("replace_left_right only accepts str of list of str as input")