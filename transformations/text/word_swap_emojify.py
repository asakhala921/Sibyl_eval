from .abstract_transformation import AbstractTransformation
from emoji_translate import Translator

class Emojify(AbstractTransformation):
	def __init__(self, exact_match_only=False, randomize=True):
		self.exact_match_only = exact_match_only
		self.randomize = randomize
		self.emo = Translator(self.exact_match_only, self.randomize)

	def __call__(self, string):
		return self.emo.emojify(string)