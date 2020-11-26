from abc import ABC, abstractmethod
import pandas as pd
 
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
        Apply the transformation using config as support

        Parameters
        ----------
        string : str
            the input string
        """
        pass

    @abstractmethod
    def get_tran_types(self, task_name=None, tran_type=None):
        pass


def _get_tran_types(tran_types, task_name=None, tran_type=None):
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
    return df