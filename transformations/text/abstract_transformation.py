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