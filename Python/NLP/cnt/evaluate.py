"""
This package contains the implementations discussed in the bachelor thesis 
"Natural Language Processing to enable semantic search on numismatic descriptions".
All code in the cnt package has been developed by Patricia Klinger.
"""

def score_precision_recall(y_true, y_pred):
    """
    calculates precision and recall of the classifier

    Parameters
    ------------
    y_true: list of lists of (subj, relation_class_label, obj)
        The annotations from the gold standards

    y_pred: list of lists of (subj, relation_class_label, obj)
        The predictions made by the estimator
    """
    true = set((sentence_counter, relation) for sentence_counter, relations in enumerate(y_true["y"])
                                       for relation in relations)
    pred = set((sentence_counter, relation) for sentence_counter, relations in enumerate(y_pred["y"])
                                       for relation in relations)
    recall = len(true & pred) / len(true) if len(true) > 0 else 0
    precision = len(true & pred) / len(pred) if len(pred) > 0 else 0
    return precision, recall


def score_accuracy(y_true, y_pred, ignore_order=True):
    """
    Percentage of annotated input data that 
    is labeled correctly by the estimator

    Parameters 
    ------------

    y_true: column of a data frame
        The annotations from the gold standards
    
    y_pred: column of a data frame
        The predictions made by the estimator
    """
    if ignore_order:
        sort = set
    else:
        sort = lambda x: x
    
    counter = 0
    for a, b in zip(y_pred["y"], y_true["y"]):
        if sort(a) == sort(b):
            counter += 1
    accuracy = counter/len(y_pred)
    return accuracy
