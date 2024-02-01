#Tutorial: https://www.kaggle.com/code/kredy10/simple-lstm-for-text-classification/notebook

import os
import nltk
import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from keras.models import load_model
from keras.optimizers import RMSprop
from keras.utils import to_categorical
from nltk.stem import WordNetLemmatizer
from keras.preprocessing import sequence
from keras.callbacks import EarlyStopping
from keras.models import Model, Sequential
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split
from keras.layers import LSTM, Activation, Dense, Dropout, Input, Embedding

nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()

class LSTMClassifier:
    def __init__(self):
        self.max_words = 2000
        self.max_len = 300
        self.tokeniser = Tokenizer(num_words=self.max_words)

        #Model adapted from tutorial
        self.model = Sequential([Input(name='inputs',shape=[self.max_len]),
        Embedding(self.max_words,50,input_length=self.max_len),
        LSTM(64),
        Dense(256,name='FC1'),
        Activation('relu'),
        Dropout(0.5),
        Dense(1,name='out_layer'),
        Activation('sigmoid')]);

    #My code
    def load_dataset(self, file):
        try:
            self.data = pd.read_csv(file,delimiter=',', encoding='latin-1')
        except IOError:
            print(f"Unexpected error with reading file {file}")
    
    #My code
    def load_model(self, model, tokeniser):
        try:
            self.model = load_model(model)

            # Load the Tokenizer
            with open(tokeniser, 'rb') as tokeniser_file:
                self.tokeniser = pickle.load(tokeniser_file)

        except IOError:
            print(f"Error with loading model from {file}")
        

    """
    @param test_size: default size is 20% (takes a float value between 0.0 - 1.0)
    """
    def test_train_split(self, test_size=0.20):
        self.X_train,self.X_test,self.Y_train,self.Y_test = train_test_split(self.X,self.Y,test_size=test_size)

    """
    This function should be used to tokenise the training data only.
    However, as the @param self is passed, the tokeniser variable may
    be used to predict texts when the model is trained.
    """

    #My code
    def standardise_text(self, data):
        punctuation = """!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""
        data = str(data).lower()

        #Remove puncutation
        for i in punctuation:
            if i in data:
                data = data.replace(i, "")
        
        #Remove stop words
        for word in stopwords.words("english"):
            if word in data:
                data.replace(word, "")
        
        filtered = []
        #Lemmonise words
        for word in data.split(" "):
            filtered.append(lemmatizer.lemmatize(word))

        #Return filtered array as string
        return ' '.join(filtered).strip()

        # for line in data:
        #     print(line)


    #CODE BORROWED FROM TUTORIAL
    def tokenise_training_data(self):
        self.standardised_x_train = self.X_train.apply(self.standardise_text)
        # self.standardised_x_train = self.X_train
        print(self.standardised_x_train)
        self.tokeniser.fit_on_texts(self.standardised_x_train)
        self.sequences = self.tokeniser.texts_to_sequences(self.standardised_x_train)
        self.sequences_matrix = sequence.pad_sequences(self.sequences,maxlen=self.max_len)

    
    """
    Any dataset with 2 columns labelled Label, Text must be used
    """
    #CODE BORROWED FROM TUTORIAL
    def create_vectors(self):
        self.X = self.data.Text
        self.Y = self.data.Label
        self.le = LabelEncoder()
        self.Y = self.le.fit_transform(self.Y)
        self.Y = self.Y.reshape(-1,1)
    
    #My code
    def save_model(self):
        c = input("Save model? Y/N")

        if c == "Y":
            # Save the Tokeniser and model
            self.model.save("model.keras")

            with open('tokeniser.pkl', 'wb') as tokeniser_file:
                pickle.dump(self.tokeniser, tokeniser_file)
            print("Model saved")

    #Code borrowed from https://www.tensorflow.org/tutorials/keras/text_classification#load_the_dataset
    def plot_training(self):
        print(self.history_dict.keys())
        acc = self.history_dict['accuracy']
        val_acc = self.history_dict['val_accuracy']
        loss = self.history_dict['loss']
        val_loss = self.history_dict['val_loss']

        epochs = range(1, len(acc) + 1)

        # "bo" is for "blue dot"
        plt.plot(epochs, loss, 'bo', label='Training loss')
        # b is for "solid blue line"
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()

        plt.show()

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend(loc='lower right')

        plt.show()


    
    #Code borrowed from tutorial
    def train(self, batch_size=128, epoch=10, validation_split=0.2):
        self.create_vectors()
        self.test_train_split(validation_split)
        self.tokenise_training_data()
        self.model.compile(loss='binary_crossentropy',optimizer=RMSprop(),metrics=['accuracy'])
        self.history = self.model.fit(self.sequences_matrix,self.Y_train,batch_size=batch_size,epochs=epoch,
          validation_split=validation_split,callbacks=[EarlyStopping(monitor='val_loss',min_delta=0.0001)])
        
        self.history_dict = self.history.history
        self.plot_training()
        self.save_model()
        
        print("Model training complete")
    
    """
    Used to evaluate the accuracy of the model using the split test data
    """
    #Code borrowed from tutorial
    def evaluate(self):
        self.test_sequences = self.tokeniser.texts_to_sequences(self.X_test.apply(self.standardise_text))
        self.test_sequences_matrix = sequence.pad_sequences(self.test_sequences,maxlen=self.max_len)
        self.accuracy = self.model.evaluate(self.test_sequences_matrix,self.Y_test)
        print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(self.accuracy[0],self.accuracy[1]))
        return self.accuracy

    #My code
    def predict(self, data: list):
        process_sequences = self.tokeniser.texts_to_sequences(data)
        process_sequences_matrix = sequence.pad_sequences(process_sequences, maxlen=self.max_len)
        prediction = self.model.predict(process_sequences_matrix)
        class_predictions = [("smish", p[0]) if p > 0.5 else ("ham", p[0]) for p in prediction]
        return class_predictions

 



