from ..abstract_transformation import AbstractTransformation
import re

class ExpandContractions(AbstractTransformation):
    """
    Expands all known contractions in a string input or 
    returns the original string if none are found. 
    """

    def __init__(self):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        """
        self.contraction_map = {
            "ain't": "is not", "aren't": "are not", "can't": "cannot",
            "can't've": "cannot have", "could've": "could have", "couldn't":
            "could not", "didn't": "did not", "doesn't": "does not", "don't":
            "do not", "hadn't": "had not", "hasn't": "has not", "haven't":
            "have not", "he'd": "he would", "he'd've": "he would have",
            "he'll": "he will", "he's": "he is", "how'd": "how did", "how'd'y":
            "how do you", "how'll": "how will", "how's": "how is",
            "I'd": "I would", "I'll": "I will", "I'm": "I am",
            "I've": "I have", "i'd": "i would", "i'll": "i will",
            "i'm": "i am", "i've": "i have", "isn't": "is not",
            "it'd": "it would", "it'll": "it will", "it's": "it is", "ma'am":
            "madam", "might've": "might have", "mightn't": "might not",
            "must've": "must have", "mustn't": "must not", "needn't":
            "need not", "oughtn't": "ought not", "shan't": "shall not",
            "she'd": "she would", "she'll": "she will", "she's": "she is",
            "should've": "should have", "shouldn't": "should not", "that'd":
            "that would", "that's": "that is", "there'd": "there would",
            "there's": "there is", "they'd": "they would",
            "they'll": "they will", "they're": "they are",
            "they've": "they have", "wasn't": "was not", "we'd": "we would",
            "we'll": "we will", "we're": "we are", "we've": "we have",
            "weren't": "were not", "what're": "what are", "what's": "what is",
            "when's": "when is", "where'd": "where did", "where's": "where is",
            "where've": "where have", "who'll": "who will", "who's": "who is",
            "who've": "who have", "why's": "why is", "won't": "will not",
            "would've": "would have", "wouldn't": "would not",
            "you'd": "you would", "you'd've": "you would have",
            "you'll": "you will", "you're": "you are", "you've": "you have"
        }
    
    def __call__(self, string):
        """Expands contractions in a string (if any)

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        string
            String with contractions expanded (if any)

        """# self.reverse_contraction_map = dict([(y, x) for x, y in self.contraction_map.items()])
        contraction_pattern = re.compile(r'\b({})\b'.format('|'.join(self.contraction_map.keys())),
            flags=re.IGNORECASE|re.DOTALL)

        def expand_match(contraction):
            match = contraction.group(0)
            first_char = match[0]
            expanded_contraction = self.contraction_map.get(match, self.contraction_map.get(match.lower()))
            expanded_contraction = first_char + expanded_contraction[1:]
            return expanded_contraction

        return contraction_pattern.sub(expand_match, string)
    

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