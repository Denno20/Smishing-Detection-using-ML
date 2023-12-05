# import tensorflow as tf
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences
# import seaborn as sns


# def main():
#     data = pd.read_csv("sms.csv")
#     print(data.head())


#     print(data.groupby('Category').describe().T)
#     plt.figure(figsize=(12,14))
#     sns.countplot(data['Category'])

#     ham_msg = data[data.Category =='ham']
#     smish_msg = data[data.Category=='smish']
#     ham_msg=ham_msg.sample(n=len(smish_msg),random_state=42)
    

#     print(ham_msg.shape,smish_msg.shape)

#     balanced_data=ham_msg._append(smish_msg).reset_index(drop=True)
#     plt.figure(figsize=(8,6))
#     sns.countplot(balanced_data.Category)
#     plt.title('Distribution of ham and phish email messages (after downsampling)')
#     plt.xlabel('Message types')
#     print(balanced_data.head())

#     balanced_data['label']=balanced_data['Category'].map({'ham':0,'smish':1})

#     train_msg, test_msg, train_labels, test_labels =train_test_split(balanced_data['Message'],balanced_data['label'],test_size=0.2,random_state=434)
#     vocab_size=500
#     oov_tok='<OOV>'
#     max_len=50

#     token=Tokenizer(num_words=vocab_size,oov_token=oov_tok)
#     token.fit_on_texts(train_msg)


#     word_index=token.word_index
#     print(word_index)

#     padding_type='post'
#     truncate_type='post'
#     Trainning_seq=token.texts_to_sequences(train_msg)
#     Trainning_pad=pad_sequences(Trainning_seq,maxlen=50,padding=padding_type,truncating=truncate_type)

#     Testing_seq=token.texts_to_sequences(test_msg)
#     Testing_pad=pad_sequences(Testing_seq,maxlen=50,padding=padding_type,truncating=truncate_type)


#     #model
#     model=tf.keras.models.Sequential([tf.keras.layers.Embedding(vocab_size,16,input_length=50),
#                                     tf.keras.layers.GlobalAveragePooling1D(),
#                                     tf.keras.layers.Dense(32,activation='relu'),
#                                     tf.keras.layers.Dropout(0.3),
#                                     tf.keras.layers.Dense(1,activation='sigmoid')])


#     model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),metrics=['accuracy'],optimizer='adam')



#     epoch=30
#     early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
#     history=model.fit(Trainning_pad, train_labels ,validation_data=(Testing_pad, test_labels),epochs=epoch,callbacks=[early_stop],verbose=2)


#     print(model.evaluate(Testing_pad, test_labels))

#     plt.plot(history.history['accuracy'])
#     plt.plot(history.history['val_accuracy'])
#     plt.title('model accuracy')
#     plt.ylabel('accuracy')
#     plt.xlabel('epoch')

    

#     predict_msg = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
#           "Ok lar... Joking wif u oni...",
#           "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]



#     print(predict_spam(predict_msg, token, model))

# def predict_spam(predict_msg, token, model):
#     new_seq = token.texts_to_sequences(predict_msg)
#     padded = pad_sequences(new_seq, maxlen =50,
#                       padding = 'post',
#                       truncating='post')
#     return (model.predict(padded))



from model import SMSModel

def main():
    model = SMSModel("Datasets/sms.csv", 30)
    print(model.train())
    print(model.evaluate())
    
    predict_msg = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
          "Ok lar... Joking wif u oni...",
          "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]


    outcome = model.predict_sms(predict_msg=predict_msg)
    print(outcome)


if __name__=="__main__":
    main()


