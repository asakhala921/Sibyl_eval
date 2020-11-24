from abc import ABC, abstractmethod
 
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
    def _get_tran_types(self, task=None):
        """
        Defines the task and type of transformation (SIB or INV) 
        to determine the effect on the expected behavior (whether 
        to change the label if SIB, or leave the label alone if INV). 

        Parameters
        ----------
        task : str
            Filters the results for the requested task.

        Returns
        -------
        desc : pandas.DataFrame
            A pandas DataFrame containing:
                - task_name : str
                    short description of the task
                - tran_type : str
                    INV == invariant ==> output behavior does 
                    not change
                    SIB == sibylvariant ==> output behavior 
                    changes in some way
        """
        pass