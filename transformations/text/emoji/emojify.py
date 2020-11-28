from ..abstract_transformation import AbstractTransformation, _get_tran_types
from emoji_translate import Translator

class Emojify(AbstractTransformation):
    def __init__(self, exact_match_only=False, randomize=True):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        
        Parameters
        ----------
        exact_match_only : boolean
            Determines whether we find exact matches for
            the emoji's name for replacement. If false,
            approximate matching is used. 
        randomize : boolean
            If true, randomizes approximate matches.
            If false, always picks the first match.
        """
        self.exact_match_only = exact_match_only
        self.randomize = randomize
        self.emo = Translator(self.exact_match_only, self.randomize)

    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The string input

        Returns
        ----------
        ret : str
            The output with as many non-stopwords translated
            to emojis as possible.
        """
        return self.emo.emojify(string)

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['INV', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df

class AddEmoji(Emojify):
    def __init__(self, num=1, polarity=[-1, 1]):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        
        Parameters
        ----------
        num : int
            The number of emojis to append to the end
            of a given string
        polarity : list
            Emoji sentiment is measured in polarity 
            between -1 and +1. This param allows you
            to pick the sentiment range you want.
            - positivev ==> [0.05, 1]    
            - negative ==> [-1, -0.05] 
            - neutral ==> [-0.05, 0.05]
        """
        super().__init__(self) 
        self.num = num
        self.polarity = polarity

    def __call__(self, string):
        """
        Parameters
        ----------
        string : str
            The string input

        Returns
        ----------
        ret : str
            The output with `num` emojis appended
        """
        return string + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))

    def get_tran_types(self, task_name=None, tran_type=None):
        pass

    def sample_emoji_by_polarity(self, p_rng, num=1):
        emojis = self.emo.emojis
        return emojis[emojis['polarity'].apply(
            lambda x: p_rng[0] <= x <= p_rng[1])].sample(num)['char'].values.tolist()

class AddPositiveEmoji(AddEmoji):
    def __init__(self, num=1, polarity=[0.05, 1]):
        super().__init__(self) 
        self.num = num
        self.polarity = polarity

    def __call__(self, string):
        return string + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df

class AddNegativeEmoji(AddEmoji):
    def __init__(self, num=1, polarity=[-1, -0.05]):
        super().__init__(self) 
        self.num = num
        self.polarity = polarity

    def __call__(self, string):
        return string + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df

class AddNeutralEmoji(AddEmoji):
    def __init__(self, num=1, polarity=[-0.05, 0.05]):
        super().__init__(self) 
        self.num = num
        self.polarity = polarity

    def __call__(self, string):
        return string + ' ' + ''.join(self.sample_emoji_by_polarity(self.polarity, self.num))

    def get_tran_types(self, task_name=None, tran_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['INV', 'INV']
        }
        df = _get_tran_types(self.tran_types, task_name, tran_type)
        return df