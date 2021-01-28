from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
 
class AbstractTransformation(ABC):
    """
    An abstract class for transformations to be applied 
    to input data. 
    """

    @abstractmethod
    def __init__(self, **kwargs):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        """
        pass
    
    @abstractmethod
    def __call__(self, string):
        """
        Apply the transformation to a string input

        Parameters
        ----------
        string : str
            the input string
        """
        pass

    @abstractmethod
    def transform_Xy(self, X, y):
        """
        Apply the transformation to a string input 
        and an int target label

        Parameters
        ----------
        X : str
            the input string
        y : int
            the target label

        Returns
        ----------
        X_ : str
            the transformed string
        y_ : int
            if SIB ==> transformed target label
            if INV ==> the original target label
        """
        pass

    @abstractmethod
    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        """
        See self._get_tran_types()
        """
        pass


    def _get_tran_types(self, tran_types, task_name=None, tran_type=None, label_type=None):
        """
        Defines the task and type of transformation (SIB or INV) 
        to determine the effect on the expected behavior (whether 
        to change the label if SIB, or leave the label alone if INV). 

        Parameters
        ----------
        task_name : str
            Filters the results for the requested task.
        tran_type : str
            Filters the results for the requested trans type,
            which is either 'INV' or 'SIB'.
        label_type : str
            Filters the results for the requested label type,
            which is either 'hard' or 'soft'.

        Returns
        -------
        df : pandas.DataFrame
            A pandas DataFrame containing:
                - task_name : str
                    short description of the task
                - tran_type : str
                    INV == invariant ==> output behavior does 
                    not change
                    SIB == sibylvariant ==> output behavior 
                    changes in some way
                - label_type : str
                    whether to use soft or hard labels
        """
        df = pd.DataFrame.from_dict(tran_types)
        if task_name is not None:
            task_names = set(df.task_name.tolist())
            if task_name not in task_names:
                raise ValueError('The selected task must be one of the following: {}'.format(', '.join(task_names)))
            df = df[df['task_name'] == task_name]
        if tran_type is not None:
            tran_types = set(df.tran_type.tolist())
            if tran_type not in tran_types:
                raise ValueError('The selected tran type must be one of the following: {}'.format(', '.join(tran_types)))
            df = df[df['tran_type'] == tran_type]
        if label_type is not None:
            label_types = set(df.label_type.tolist())
            if label_type not in label_types:
                raise ValueError('The selected label type must be one of the following: {}'.format(', '.join(tran_types)))
            df = df[df['label_type'] == label_type]
        return df

def one_hot_encode(y, nb_classes):
    if not isinstance(y, np.ndarray):
        y = np.expand_dims(np.array(y), 0)
    res = np.eye(nb_classes)[np.array(y).reshape(-1)]
    return res.reshape(list(y.shape)+[nb_classes])

def soften_label(y, num_classes=None):
    if isinstance(y, np.ndarray):
        return y
    if not num_classes:
        num_classes = max(2, y)
    return one_hot_encode(y, num_classes) 

def invert_label(y, soften=False, num_classes=None):
    if soften:
        y = soften_label(y, num_classes)
    if isinstance(y, np.ndarray):
        return (1 - y) / (1 - y).sum()
    else:
        return int(not y)

def interpolate_label(y1, y2, x1=None, x2=None, num_classes=None, y_weights=None):
    if isinstance(y_weights, (list, tuple)):
        mass_y1 = y_weights[0]
        mass_y2 = y_weights[1]
    elif x1 and x2:
        mass_y1 = len(x1) / (len(x1) + len(x2)) 
        mass_y2 = 1 - mass_y1
    else:
        mass_y1 = 1
        mass_y2 = 1    
    y1 = soften_label(y1, num_classes) * mass_y1
    y2 = soften_label(y2, num_classes) * mass_y2
    return (y1 + y2) / (y1 + y2).sum()

def weight_label(y, y_weights=None):
    if not y_weights:
        y_weights = np.ones_like(y)
    y = y * np.array(y_weights)
    return y / y.sum()

def smooth_label(y, factor=0.1):
    if not isinstance(y, np.ndarray):
        y = soften_label(y)
    y = y * (1. - factor)
    y = y + (factor / y.shape[-1])
    return y