from ..abstract_transformation import AbstractTransformation
from urllib.request import urlopen
from bs4 import BeautifulSoup 
from bs4.element import Comment
import re

class ImportLinkText(AbstractTransformation):
    """
    Appends a given / constructed URL to a string input.
    Current implementation constructs a default URL that
    makes use of dictionary.com and is sensitive to changes
    in routing structure. 
    """

    def __init__(self):
        """
        Initializes the transformation and provides an
        opporunity to supply a configuration if needed

        Parameters
        ----------
        NA
        
        """
        # https://gist.github.com/uogbuji/705383#gistcomment-2250605
        self.URL_REGEX = re.compile(r'(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:\'".,<>?\xab\xbb\u201c\u201d\u2018\u2019]))')
    
    def __call__(self, string):
        """
        Add extracted (visible)

        Parameters
        ----------
        string : str
            Input string

        Returns
        -------
        ret
            String with visible text from the URL appended
        """
        def replace(match):
            url = match.group(0)
            return get_url_text(url)
        ret = self.URL_REGEX.sub(replace, string)
        return ret

def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_url_text(url):
    try:
        html = urlopen(url).read()
    except:
        return url
    soup = BeautifulSoup(html, 'html.parser')
    texts = soup.findAll(text=True)
    visible_texts = filter(tag_visible, texts)  
    return u" ".join(t.strip() for t in visible_texts)