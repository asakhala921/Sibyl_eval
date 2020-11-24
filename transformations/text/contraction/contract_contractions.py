from ..abstract_transformation import AbstractTransformation
import re

class ContractContractions(AbstractTransformation):
    """
    Contracts all known contractions in a string input or 
    returns the original string if none are found. 
    """

    def __init__(self):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        """
        self.reverse_contraction_map = {
            'is not': "isn't", 'are not': "aren't", 'cannot': "can't",
            'could not': "couldn't", 'did not': "didn't", 'does not':
            "doesn't", 'do not': "don't", 'had not': "hadn't", 'has not':
            "hasn't", 'have not': "haven't", 'he is': "he's", 'how did':
            "how'd", 'how is': "how's", 'I would': "I'd", 'I will': "I'll",
            'I am': "I'm", 'i would': "i'd", 'i will': "i'll", 'i am': "i'm",
            'it would': "it'd", 'it will': "it'll", 'it is': "it's",
            'might not': "mightn't", 'must not': "mustn't", 'need not': "needn't",
            'ought not': "oughtn't", 'shall not': "shan't", 'she would': "she'd",
            'she will': "she'll", 'she is': "she's", 'should not': "shouldn't",
            'that would': "that'd", 'that is': "that's", 'there would':
            "there'd", 'there is': "there's", 'they would': "they'd",
            'they will': "they'll", 'they are': "they're", 'was not': "wasn't",
            'we would': "we'd", 'we will': "we'll", 'we are': "we're", 'were not':
            "weren't", 'what are': "what're", 'what is': "what's", 'when is':
            "when's", 'where did': "where'd", 'where is': "where's",
            'who will': "who'll", 'who is': "who's", 'who have': "who've", 'why is':
            "why's", 'will not': "won't", 'would not': "wouldn't", 'you would':
            "you'd", 'you will': "you'll", 'you are': "you're",
        }
    
    def __call__(self, string):
        """Contracts contractions in a string (if any)

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        string
            String with contractions expanded (if any)
        """

        reverse_contraction_pattern = re.compile(r'\b({})\b '.format('|'.join(self.reverse_contraction_map.keys())),
            flags=re.IGNORECASE|re.DOTALL)
        def cont(possible):
            match = possible.group(1)
            first_char = match[0]
            expanded_contraction = self.reverse_contraction_map.get(match, self.reverse_contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:] + ' '
            return expanded_contraction
        return reverse_contraction_pattern.sub(cont, string)

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
        tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['INV', 'INV']
        }
        desc = pd.DataFrame.from_dict(tran_types)
        if task is not None:
            task_names = desc.task_name.tolist():
            if task not in task_names:
                raise ValueError('The selected task must be one of the following:', ', '.join(task_names))
            desc = desc[desc['task_name'] == task]
        return desc