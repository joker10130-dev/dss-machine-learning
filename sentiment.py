import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

from numpy import array
from numpy import asarray
from numpy import zeros
from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt

class sentiment:


    def __init__(self):
        self.X = []

        self.movie_reviews = pd.read_csv("./IMDB Dataset.csv")

        self.movie_reviews.isnull().values.any()

        self.movie_reviews.shape

        print(self.movie_reviews.head())
        print(self.movie_reviews["review"][3])

        self.sentences = list(self.movie_reviews['review'])
        for sen in self.sentences:
            self.X.append(self.preprocess_text(sen))

        print(self.X[3])

        self.y = self.movie_reviews['sentiment']

        self.y = np.array(list(map(lambda x: 1 if x=="positive" else 0, self.y)))
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=42)

        self.tokenizer = Tokenizer(num_words=5000)
        self.tokenizer.fit_on_texts(self.X_train)

        self.X_train = self.tokenizer.texts_to_sequences(self.X_train)
        self.X_test = self.tokenizer.texts_to_sequences(self.X_test)

        self.vocab_size = len(self.tokenizer.word_index) + 1

        self.maxlen = 100

        self.X_train = pad_sequences(self.X_train, padding='post', maxlen=self.maxlen)
        self.X_test = pad_sequences(self.X_test, padding='post', maxlen=self.maxlen)



        self.embeddings_dictionary = dict()
        self.glove_file = open('./glove.6B.100d.txt', encoding="utf8")

        for line in self.glove_file:
            self.records = line.split()
            self.word = self.records[0]
            self.vector_dimensions = asarray(self.records[1:], dtype='float32')
            self.embeddings_dictionary [self.word] = self.vector_dimensions
        self.glove_file.close()

        self.embedding_matrix = zeros((self.vocab_size, 100))
        for word, index in self.tokenizer.word_index.items():
            self.embedding_vector = self.embeddings_dictionary.get(word)
            if self.embedding_vector is not None:
                self.embedding_matrix[index] = self.embedding_vector
                #print(embedding_vector)

        self.model = Sequential()
        self.recurrnt_neural_lstm()

    def preprocess_text(self,sen):
        # Removing html tags
        self.sentence = re.compile(r'<[^>]+>').sub('', sen)

        # Remove punctuations and numbers
        self.sentence = re.sub('[^a-zA-Z]', ' ', self.sentence)

        # Single character removal
        self.sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', self.sentence)

        # Removing multiple spaces
        self.sentence = re.sub(r'\s+', ' ', self.sentence)

        return self.sentence



    

    def simple_neural():
        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen , trainable=False)
        model.add(embedding_layer)

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        print(model.summary())

        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

        score = model.evaluate(X_test, y_test, verbose=1)
        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc='upper left')
        plt.show()

    def convolution_neural():

        embedding_layer = Embedding(vocab_size, 100, weights=[embedding_matrix], input_length=maxlen, trainable=False)
        model.add(embedding_layer)

        model.add(Conv1D(128, 5, activation='relu'))
        model.add(MaxPooling1D(3))
        model.add(Dropout(0.5))
        model.add(GlobalMaxPooling1D())

        model.add(Dense(1, activation='sigmoid'))
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])

        print(model.summary())

        history = model.fit(X_train, y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)

        score = model.evaluate(X_test, y_test, verbose=1)

        print("Test Score:", score[0])
        print("Test Accuracy:", score[1])

        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc = 'upper left')
        plt.show()

        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train','test'], loc = 'upper left')
        plt.show()

    def recurrnt_neural_lstm(self):
        self.embedding_layer = Embedding(self.vocab_size, 100, weights=[self.embedding_matrix], input_length=self.maxlen, trainable=False)
        self.model.add(self.embedding_layer)
        self.model.add(LSTM(128))

        self.model.add(Dense(1, activation='sigmoid'))
        self.model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
        print(self.model.summary())

        self.history = self.model.fit(self.X_train, self.y_train, batch_size=128, epochs=6, verbose=1, validation_split=0.2)
        self.score = self.model.evaluate(self.X_test, self.y_test, verbose=1)

        print("Test Score:", self.score[0])
        print("Test Accuracy:", self.score[1])

        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])

        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()



    
    #instance = "bad rice's taste"
    #print(instance)

    def text_process(self, word):
        self.instance = self.tokenizer.texts_to_sequences(word)

        self.flat_list = []
        for sublist in self.instance:
            for item in sublist:
                self.flat_list.append(item)

        self.flat_list = [self.flat_list]

        self.instance = pad_sequences(self.flat_list, padding='post', maxlen=self.maxlen)
        #return model.predict(instance)

        print(self.model.predict(self.instance))
        return self.model.predict(self.instance).tolist()
        