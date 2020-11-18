from .abstract_transformation import AbstractTransformation
import collections
import pattern
import spacy
import en_core_web_sm

class AddNegation(AbstractTransformation):
    """
    An abstract class for transformations to be applied 
    to input data. 
    """

    def __init__(self):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed
        """
        self.nlp = en_core_web_sm.load()
    
    def __call__(self, string):
        """Removes negation from a string by first 
        converting it to spacy.token.Doc. 

        Parameters
        ----------
        string : str
            Input to have negation removed

        Returns
        -------
        ret : str
            Output with *all* negations removed

        """
        # This removes all negations in the doc. I should maybe add an option to remove just some.
        doc = self.nlp(string)
        for sentence in doc.sents:
            if len(sentence) < 3:
                continue
            root_id = [x.i for x in sentence if x.dep_ == 'ROOT'][0]
            root = doc[root_id]
            if '?' in sentence.text and sentence[0].text.lower() == 'how':
                continue
            if root.lemma_.lower() in ['thank', 'use']:
                continue
            if root.pos_ not in ['VERB', 'AUX']:
                continue
            neg = [True for x in sentence if x.dep_ == 'neg' and x.head.i == root_id]
            if neg:
                continue
            if root.lemma_ == 'be':
                if '?' in sentence.text:
                    continue
                if root.text.lower() in ['is', 'was', 'were', 'am', 'are', '\'s', '\'re', '\'m']:
                    return doc[:root_id + 1].text + ' not ' + doc[root_id + 1:].text
                else:
                    return doc[:root_id].text + ' not ' + doc[root_id:].text
            else:
                aux = [x for x in sentence if x.dep_ in ['aux', 'auxpass'] and x.head.i == root_id]
                if aux:
                    aux = aux[0]
                    if aux.lemma_.lower() in ['can', 'do', 'could', 'would', 'will', 'have', 'should']:
                        lemma = doc[aux.i].lemma_.lower()
                        if lemma == 'will':
                            fixed = 'won\'t'
                        elif lemma == 'have' and doc[aux.i].text in ['\'ve', '\'d']:
                            fixed = 'haven\'t' if doc[aux.i].text == '\'ve' else 'hadn\'t'
                        elif lemma == 'would' and doc[aux.i].text in ['\'d']:
                            fixed = 'wouldn\'t'
                        else:
                            fixed = doc[aux.i].text.rstrip('n') + 'n\'t' if lemma != 'will' else 'won\'t'
                        fixed = ' %s ' % fixed
                        return doc[:aux.i].text + fixed + doc[aux.i + 1:].text
                    return doc[:root_id].text + ' not ' + doc[root_id:].text
                else:
                    # TODO: does, do, etc. Remover return None de cima
                    subj = [x for x in sentence if x.dep_ in ['csubj', 'nsubj']]
                    p = pattern.en.tenses(root.text)
                    tenses = collections.Counter([x[0] for x in pattern.en.tenses(root.text)]).most_common(1)
                    tense = tenses[0][0] if len(tenses) else 'present'
                    params = [tense, 3]
                    if p:
                        tmp = [x for x in p if x[0] == tense]
                        if tmp:
                            params = list(tmp[0])
                        else:
                            params = list(p[0])
                    if root.tag_ not in ['VBG']:
                        do = pattern.en.conjugate('do', *params) + 'n\'t'
                        new_root = pattern.en.conjugate(root.text, tense='infinitive')
                    else:
                        do = 'not'
                        new_root = root.text
                    return '%s %s %s %s' % (doc[:root_id].text, do, new_root,  doc[root_id + 1:].text)