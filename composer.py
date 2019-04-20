
import re

# Load the data set:
RAW_TEXT = open('lyrics.txt', encoding = 'UTF-8').read()

# create mapping of unique chars to integers
CHARS = sorted(list(set(RAW_TEXT)))
CHAR_TO_INT = dict((c, i) for i, c in enumerate(CHARS))
INT_TO_CHAR = dict((i, c) for i, c in enumerate(CHARS))

TOTAL_CHARS = len(RAW_TEXT)
VOCAB_SIZE = len(CHARS)

# print('Total Characters: ', TOTAL_CHARS) # 1.1M
# print('Total Vocab: ', VOCAB_SIZE) # 46 distinct characters (much more than 26 in the alphabet)
# print(CHARS)

# TODO: try using tokens instead of chars
# tokens = set(re.sub('[^a-z0-9- \n]', '', RAW_TEXT).split())
# print(len(tokens))
# print(list(tokens)[:1000])



# prepare the data set of input to output pairs encoded as integers
SEQ_LENGTH = 100



