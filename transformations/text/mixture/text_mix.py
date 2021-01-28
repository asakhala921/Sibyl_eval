from ..abstract_batch_transformation import AbstractBatchTransformation, self._get_tran_types
import numpy as np
import pandas as pd
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def concat_text(np_char1, np_char2):
    np_char1 = np_char1.astype(np.string_)
    np_char2 = np_char2.astype(np.string_)
    sep = np.full_like(np_char1, " ", dtype=np.string_)
    ret = np.char.add(np_char1, sep)
    ret = np.char.add(ret, np_char2)
    return ret

def sent_shuffle(string):
    X = nltk.tokenize.sent_tokenize(str(string))
    np.random.shuffle(X)
    return ' '.join(X)

def word_shuffle(string):
    X = str(string).split(" ")
    np.random.shuffle(X)
    return ' '.join(X)

class TextMix(AbstractBatchTransformation):
    """
    Concatenates two texts together and interpolates 
    the labels
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, batch):
        """
        Parameters
        ----------
        batch : tuple(np.array, np.array)
            The input batch of (X, y) pairs

        Returns
        ----------
        ret : tuple(np.array, np.array)
            : tuple(np.array, np.array, dict)
            The output batch of transformed (X, y)
            pairs. y --> one-hot-encoded, weighted
            by the amount of text drawn from each
            of the inputs. May return metadata dict
            if requested.
        """
        data, targets = batch
        if type(data) == list:
            data = np.array(data, dtype=np.string_)
            targets = np.array(targets)

        # shuffle data, targets
        indices = np.random.permutation(len(data))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        # concatenate data
        textmix = concat_text(data, shuffled_data)

        # create soft target labels by first
        # one-hot-encoding targets if necessary
        if targets.shape[-1] == 1:
            ohe_target1 = pd.get_dummies(targets).to_numpy(dtype=np.float)
            ohe_target2 = pd.get_dummies(shuffled_targets).to_numpy(dtype=np.float)
            classes1 = targets
            classes2 = shuffled_targets
        else:
            ohe_target1 = targets
            ohe_target2 = shuffled_targets
            classes1 = np.argmax(ohe_target1, axis=1)
            classes2 = np.argmax(ohe_target2, axis=1)
        idx = np.arange(len(ohe_target1))

        # calculate length of each data and use that
        # to determine the lambda weight assigned to
        # the index for the target
        len_data = np.char.str_len(data)
        len_shuffled_data = np.char.str_len(shuffled_data)
        lam = len_data / (len_data + len_shuffled_data)

        ohe_target1[idx, classes1] *= lam
        ohe_target2[idx, classes2] *= 1-lam

        ohe_targets = ohe_target1 + ohe_target2

        ret = (textmix, ohe_targets)

        # metadata
        if self.metadata: 
            meta = {'change': True}
            return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'SIB'],
            'label_type': ['soft', 'soft']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df

class SentMix(AbstractBatchTransformation):
    """
    Concatenates two texts together and then mixes the
    sentences in the new string. Interpolates the labels
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, batch):
        """
        Parameters
        ----------
        batch : tuple(np.array, np.array)
            The input batch of (X, y) pairs

        Returns
        ----------
        ret : tuple(np.array, np.array)
            : tuple(np.array, np.array, dict)
            The output batch of transformed (X, y)
            pairs. y --> one-hot-encoded, weighted
            by the amount of text drawn from each
            of the inputs. May return metadata dict
            if requested.
        """
        data, targets = batch
        if type(data) == list:
            data = np.array(data, dtype=np.string_)
            targets = np.array(targets)

        # shuffle data, targets
        indices = np.random.permutation(len(data))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        # concatenate data
        textmix = concat_text(data, shuffled_data)

        # mix sentences
        sent_shuffle_ = np.vectorize(sent_shuffle)
        sentmix = np.apply_along_axis(sent_shuffle_, 0, textmix)

        # create soft target labels by first
        # one-hot-encoding targets if necessary
        if len(targets.shape) == 1:
            ohe_target1 = pd.get_dummies(targets).to_numpy(dtype=np.float)
            ohe_target2 = pd.get_dummies(shuffled_targets).to_numpy(dtype=np.float)
            classes1 = targets
            classes2 = shuffled_targets
        else:
            ohe_target1 = targets
            ohe_target2 = shuffled_targets
            classes1 = np.argmax(ohe_target1, axis=1)
            classes2 = np.argmax(ohe_target2, axis=1)
        idx = np.arange(len(ohe_target1))

        # calculate length of each data and use that
        # to determine the lambda weight assigned to
        # the index for the target
        len_data = np.char.str_len(data)
        len_shuffled_data = np.char.str_len(shuffled_data)
        lam = len_data / (len_data + len_shuffled_data)

        ohe_target1[idx, classes1] *= lam
        ohe_target2[idx, classes2] *= 1-lam

        ohe_targets = ohe_target1 + ohe_target2

        ret = (sentmix, ohe_targets)

        # metadata
        if self.metadata: 
            meta = {'change': True}
            return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'SIB'],
            'label_type': ['soft', 'soft']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df

class WordMix(AbstractBatchTransformation):
    """
    Concatenates two texts together and then mixes the
    words in the new string. Interpolates the labels
    """
    def __init__(self, task=None, meta=False):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        task : str
            the type of task you wish to transform the
            input towards
        """
        self.task = task
        self.metadata = meta
        
    def __call__(self, batch):
        """
        Parameters
        ----------
        batch : tuple(np.array, np.array)
            The input batch of (X, y) pairs

        Returns
        ----------
        ret : tuple(np.array, np.array)
            : tuple(np.array, np.array, dict)
            The output batch of transformed (X, y)
            pairs. y --> one-hot-encoded, weighted
            by the amount of text drawn from each
            of the inputs. May return metadata dict
            if requested.
        """
        data, targets = batch
        if type(data) == list:
            data = np.array(data, dtype=np.string_)
            targets = np.array(targets)

        # shuffle data, targets
        indices = np.random.permutation(len(data))
        shuffled_data = data[indices]
        shuffled_targets = targets[indices]

        # concatenate data
        textmix = concat_text(data, shuffled_data)

        # mix words
        word_shuffle_ = np.vectorize(word_shuffle)
        wordmix = np.apply_along_axis(word_shuffle_, 0, textmix)

        # create soft target labels by first
        # one-hot-encoding targets if necessary
        if len(targets.shape) == 1:
            ohe_target1 = pd.get_dummies(targets).to_numpy(dtype=np.float)
            ohe_target2 = pd.get_dummies(shuffled_targets).to_numpy(dtype=np.float)
            classes1 = targets
            classes2 = shuffled_targets
        else:
            ohe_target1 = targets
            ohe_target2 = shuffled_targets
            classes1 = np.argmax(ohe_target1, axis=1)
            classes2 = np.argmax(ohe_target2, axis=1)
        idx = np.arange(len(ohe_target1))

        # calculate length of each data and use that
        # to determine the lambda weight assigned to
        # the index for the target
        len_data = np.char.str_len(data)
        len_shuffled_data = np.char.str_len(shuffled_data)
        lam = len_data / (len_data + len_shuffled_data)

        ohe_target1[idx, classes1] *= lam
        ohe_target2[idx, classes2] *= 1-lam

        ohe_targets = ohe_target1 + ohe_target2

        ret = (wordmix, ohe_targets)

        # metadata
        if self.metadata: 
            meta = {'change': True}
            return ret, meta
        return ret

    def get_tran_types(self, task_name=None, tran_type=None, label_type=None):
        self.tran_types = {
            'task_name': ['sentiment', 'topic'],
            'tran_type': ['SIB', 'SIB'],
            'label_type': ['soft', 'soft']
        }
        df = self._get_tran_types(self.tran_types, task_name, tran_type, label_type)
        return df