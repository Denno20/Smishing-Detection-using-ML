import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import seaborn as sns

class SMSModel:
    def __init__(self, datafile, epoch) -> None:
        self.data = pd.read_csv(datafile)

        self.vocab_size=500
        self.oov_tok='<OOV>'
        self.max_len=50
        self.epoch = epoch

        #model
        self.model=tf.keras.models.Sequential([tf.keras.layers.Embedding(self.vocab_size,16,input_length=50),
                                    tf.keras.layers.GlobalAveragePooling1D(),
                                    tf.keras.layers.Dense(32,activation='relu'),
                                    tf.keras.layers.Dropout(0.3),
                                    tf.keras.layers.Dense(1,activation='sigmoid')])
    
    def getBalancedData(self):
        balanced_data=self.ham_msg._append(self.smish_msg).reset_index(drop=True)
        balanced_data['label']=balanced_data['Category'].map({'ham':0,'smish':1})
        return balanced_data
    
    def train(self):
        self.ham_msg = self.data[self.data.Category =='ham']
        self.smish_msg = self.data[self.data.Category=='smish']
        self.ham_msg= self.ham_msg.sample(n=len(self.smish_msg),random_state=42)

        self.train_msg, self.test_msg, self.train_labels, self.test_labels =train_test_split(self.getBalancedData()['Message'],self.getBalancedData()['label'],test_size=0.2,random_state=434)

        self.token=Tokenizer(num_words=self.vocab_size,oov_token=self.oov_tok)
        self.token.fit_on_texts(self.train_msg)

        self.padding_type='post'
        self.truncate_type='post'
        self.Trainning_seq=self.token.texts_to_sequences(self.train_msg)
        self.Trainning_pad=pad_sequences(self.Trainning_seq,maxlen=50,padding=self.padding_type,truncating=self.truncate_type)

        self.Testing_seq=self.token.texts_to_sequences(self.test_msg)
        self.Testing_pad=pad_sequences(self.Testing_seq,maxlen=50,padding=self.padding_type,truncating=self.truncate_type)
        
        self.model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'],optimizer='adam')

        self.early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
        history=self.model.fit(self.Trainning_pad, self.train_labels ,validation_data=(self.Testing_pad, self.test_labels),epochs=self.epoch,callbacks=[self.early_stop],verbose=2)

        print("Training complete")

    def evaluate(self):
        return self.model.evaluate(self.Testing_pad, self.test_labels)


    def predict(self, predict_msg):
        new_seq = self.token.texts_to_sequences(predict_msg)
        padded = pad_sequences(new_seq, maxlen =50,
                        padding = 'post',
                        truncating='post')
        return (self.model.predict(padded))

    def predict_sms(self, predict_msg):
        new_seq = self.token.texts_to_sequences(predict_msg)
        padded = pad_sequences(new_seq, maxlen =50,
                      padding = 'post',
                      truncating='post')
        return (self.model.predict(padded))