import re
import numpy
from tensorflow.keras import utils
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import LSTM
from tensorflow. keras.callbacks import ModelCheckpoint
import sys
# tf.enable_eager_execution()

# Load the data set:
RAW_TEXT = open('lyrics.txt', encoding = 'UTF-8').read()

# using tokens instead of chars
# regex --> we don' want anything but !?\'&\:\n., or word-word, or word
TOKENS = list(filter(lambda s: len(s) > 0 or s == ' ', re.findall(r'[ !?\'&\:\n.,]|\w+-\w+|\w+', RAW_TEXT)))
UNIQUE_TOKENS = sorted(set(TOKENS))

# create mapping of unique tokens to integers
TOKEN_TO_INT = dict((c, i) for i, c in enumerate(UNIQUE_TOKENS))
INT_TO_TOKEN = dict((i, c) for i, c in enumerate(UNIQUE_TOKENS))

TOTAL_TOKENS = len(TOKENS)
VOCAB_SIZE = len(UNIQUE_TOKENS)

EPOCHS = 50
BATCH_SIZE = 128
SEQ_LENGTH = 100

print('Total Tokens: ', TOTAL_TOKENS) # ~400M
print('Total Vocab: ', VOCAB_SIZE) # ~12K
print('Some of the Tokens: \n', UNIQUE_TOKENS[:100])


def load_data():
    # prepare the data set of input to output pairs encoded as integers
    dataX = []
    dataY = []
    for i in range(0, TOTAL_TOKENS - SEQ_LENGTH, 1):
        seq_in = TOKENS[i:i + SEQ_LENGTH]
        seq_out = TOKENS[i + SEQ_LENGTH]
        dataX.append([TOKEN_TO_INT[token] for token in seq_in])
        dataY.append(TOKEN_TO_INT[seq_out])
    n_patterns = len(dataX)

    print ("Total Patterns: ", n_patterns) # a bit under 12K (TOTAL_TOKENS = SEQ_LENGTH)
    # reshape X to be [samples, time steps, features]
    X = numpy.reshape(dataX, (n_patterns, SEQ_LENGTH, 1))

    # normalize
    X = X / float(VOCAB_SIZE)

    # one hot encode the output variable
    # TODO: fix warning. Use integer labels, instead of one hot encode labels, with the sparse_categorical_crossentropy loss
    y = utils.to_categorical(dataY, dtype='uint8')

    return X,y

def create_model(X, y):
    # define the LSTM model
    # Problem the model solves: single character classification problem with 46 classes (VOCAB_SIZE)
    # There is no test dataset. Model the entire training dataset to learn the probability of each character in a sequence
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2]), return_sequences=True)) #hidden LSTM layer with 256 memory units
    model.add(Dropout(0.2))
    model.add(LSTM(256))
    model.add(Dropout(0.2))
    model.add(Dense(y.shape[1], activation='softmax')) # outputs a probability prediction for each of the 46 characters between 0 and 1
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    return model

def train(X, y, checkpoint_file=None):
    model = create_model(X, y)

    # use model checkpointing for optimization (too slow to train)
    # record all of the network weights to file each time
    # an improvement in loss is observed at the end of the epoch
    # then the best set of weights (lowest loss) to instantiate the generative model in the next section

    # define the checkpoint
    filepath = 'token-weights-improvement-{epoch:02d}-{loss:.4f}.hdf5'
    checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min', period=5)
    callbacks_list = [checkpoint]

    if checkpoint_file:
        print("Loading checkpoint: " + checkpoint_file)
        model.load_weights(checkpoint_file)

    # Fit model to the data (for now use 20 epochs and a large batch size of 128 patterns)
    model.fit(X, y, epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=callbacks_list, shuffle=True)

def generate_lyric(X, y, weights_file):
    model = create_model(X, y)

    # load the network weights
    model.load_weights(weights_file)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    n_patterns = TOTAL_TOKENS - SEQ_LENGTH
    start = numpy.random.randint(0, n_patterns-1)
    pattern = [TOKEN_TO_INT[token] for token in TOKENS[start:start+SEQ_LENGTH]]
    output = [INT_TO_TOKEN[value] for value in pattern]

    print('Seed:')
    print(' '.join(output))

    # generate characters
    for i in range(1000):
        X = numpy.reshape(pattern, (1, len(pattern), 1))
        X = X / float(VOCAB_SIZE)
        prediction = model.predict(X, verbose=0)
        index = numpy.argmax(prediction)
        result = INT_TO_TOKEN[index]
        output.append(result)
        pattern.append(index)
        pattern = pattern[1:len(pattern)]

    return ' '.join(output)

if __name__ == '__main__':
    x, y = load_data()

    if sys.argv[1] == "train":
        train(x, y, checkpoint_file=None if len(sys.argv) < 3 else sys.argv[2])
    else:
        print('\nOUTPUT:\n', generate_lyric(x, y, sys.argv[1]))

    print('END')