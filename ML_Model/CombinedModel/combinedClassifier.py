
#The tutorial on which I built my code
# https://www.kaggle.com/code/llabhishekll/text-preprocessing-and-sms-spam-detection/notebook

#Import required libraries
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score, classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import string
import nltk
import re

#Download stop words if needed
nltk.download('stopwords')
nltk.download('wordnet')

#Define the lemmatizer
lemmatizer = WordNetLemmatizer()
stopset = list(set(stopwords.words("english")))
random_state = 42

class CombinedClassifier:
    def __init__(self):
        #Define the vectorizer
        self.vectorizer = TfidfVectorizer(stop_words=stopset)
        #Define the classifiers that will be used in the voting classifier
        self.A = MultinomialNB(alpha=1.0,fit_prior=False)
        self.B = AdaBoostClassifier(n_estimators=100,random_state=random_state)
        self.C = RandomForestClassifier(n_estimators=100,random_state=random_state)
        self.D = MLPClassifier(early_stopping=True, batch_size=128, verbose=False,random_state=random_state)
        self.E = BernoulliNB(alpha=1.0, fit_prior=False)
        self.F = DecisionTreeClassifier(random_state=random_state)
        self.classifiers = [self.A,self.B,self.C,self.D,self.E, self.F]
        self.objects = ('MultiNB', 'ADB', 'RF', 'MLP', 'BNB', 'DT')

    # Function to load the dataset
    def load_dataset(self, file):
        self.data = pd.read_csv(file, encoding="latin-1")

    # function to train classifier
    def train_classifier(self, clf, X_train, y_train):    
        clf.fit(X_train, y_train)

    # function to predict features 
    def predict_labels(self, clf, features):
        return(clf.predict(features))

        # Preprocess the new data
    def preprocess_new_data(self, new_data):
        preprocessed_data = new_data.apply(self.standardise_text)
        return preprocessed_data

    # Predict labels for the new data
    def predict_new_data(self, clf, X_new):
        y_pred_new = clf.predict(X_new)
        return y_pred_new
        
    #Function to standardise text before TF-IDF features are extracted
    def standardise_text(self,data):
        #Convert to lowercase
        data = data.lower()
        #Remove punctuation
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        data = regex.sub('', data)
        
        filtered = []
        #Lemmonise words
        for word in data.split(" "):
            filtered.append(lemmatizer.lemmatize(word))
        #Return filtered array as string
        return ' '.join(filtered).strip()

    #Function to load the model
    def load_model(self, model, vectorizer):
        #Using joblib, the joblib file gets loaded as a new model
        self.votingClassifier = joblib.load(model)
        #The vectorizer also needs loading due to configuration during training
        self.vectorizer = joblib.load(vectorizer)
    


    #Function to save the model which is called after training
    def save_model(self):
        #Ask if the user wants to save the model
        c = input("Save model? Y/N").upper()
        if c == "Y":
            joblib.dump(self.votingClassifier, 'voting_classifier.joblib')
            joblib.dump(self.vectorizer, 'vectorizer.joblib')
            print("Model saved")

    def get_balanced_data(self):
        #Drop all unused data
        balanced_data=self.legitimate._append(self.smishing).reset_index(drop=True)
        balanced_data['Class']=balanced_data['Label'].map({'ham':0,'smish':1})
        print(f"The length of balanced data {len(balanced_data)}")
        return balanced_data

    def train(self):
        data = self.data

        #Convert label to numerical variable
        data["Class"] = data["Label"].map({'ham':0, 'smish': 1})

        #Get 2 class lists
        self.legitimate = data[data["Class"] == 0]
        self.smishing = data[data["Class"] == 1]
        #Sample the legitimate list to the same size as the smishing list
        self.legitimate = self.legitimate.sample(n=len(self.smishing), random_state=random_state)
        data = self.get_balanced_data()


        print(data.info())
        sns.countplot(data.Label)
        plt.xlabel('Balanced Data')
        plt.title('Number of ham and smish texts')
        plt.show()

        # Define an empty list
        corpus = []

        # For each text in the dataset, apply pre-processing and add to corpus list
        for i in range(0, len(data.Text)):
            message = data.Text[i]
            message = self.standardise_text(message)
            corpus.append(message)

        # Extract feature column 'Text'
        X = self.vectorizer.fit_transform(corpus).toarray()


        # Get the feature names
        feature_names = self.vectorizer.get_feature_names_out() 

        # Get a dictionary using the feature names
        word_scores = dict(zip(feature_names, X.sum(axis=0).tolist()))

        # Create the word cloud
        word_cloud = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(word_scores)

        # Display word cloud
        plt.figure(figsize=(10, 5))
        plt.imshow(word_cloud, interpolation="bilinear")
        plt.axis("off")
        plt.show()


        # Extract target column 'Class'
        y = data.Class

        #Split the training data into a 80:20 split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, train_size=0.80,random_state=random_state)
        cv_score = []

        #Weight calculation borrowed from tutorial
        #Calculate the cross value scores for each classifier, used to assign a weight to each classifier
        for c in self.classifiers:
            scores = cross_val_score(c, X_train, y_train, cv=5, scoring='accuracy')
            cv_score.append(scores.mean())

        #Calculate weights which give a "priority" to stronger classifiers
        total_score = sum(cv_score)
        weights = [score / total_score for score in cv_score]
        
        #Create the voting classsifier
        self.votingClassifier = VotingClassifier(estimators=[
        ('NB', self.A),
        ('ABC', self.B),
        ('RF', self.C),
        ('NN', self.D),
        ('BNB', self.E),
        ('DTC', self.F)
        ], 
        voting='soft', 
        weights=weights,
        verbose=True,
        n_jobs=1,
        )

        #F1 calculation code borrowed from tutorial
        #Default values for all classifiers are 0
        pred_val = [0,0,0,0,0,0]

        #It isn't necessary to train each classifier individually
        #to fit the voting classifier, this is just to show the F1 score
        #for each classifier

        #Loop through each classifier
        for a in range(0,(len(self.classifiers))):
            #Train the classifier using the extracted data
            self.train_classifier(self.classifiers[a], X_train, y_train)
            y_pred = self.predict_labels(self.classifiers[a],X_test)
            #Calcuate the F1 score
            pred_val[a] = f1_score(y_test, y_pred) 
            print(pred_val[a])

        #Plot data for F1 Score
        y_pos = np.arange(len(self.objects))
        y_val = [ x for x in pred_val]
        plt.bar(y_pos,y_val, align='center', alpha=0.7)
        plt.xticks(y_pos, self.objects)
        plt.ylabel('Accuracy Score')
        plt.title('Accuracy of Models')
        plt.show()

        print(classification_report(y_pred,y_test))


        # plt.title('Voting Classifier \nAccuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))


        # Creates a confusion matrix
        cm = confusion_matrix(y_pred,y_test)
        print(accuracy_score(y_test, y_pred))

        # Transform to df for easier plotting
        cm_df = pd.DataFrame(cm,
                            index=['TP', 'FP'],
                            columns=['FN', 'TN'])

        plt.figure(figsize=(5.5,4))
        sns.heatmap(cm_df, annot=True,fmt='g')
        plt.title('Voting Classifier - public dataset only')
        plt.ylabel('Predicted class')
        plt.xlabel('Actual class')
        plt.show()

        
        #Train the voting classifier
        self.votingClassifier.fit(X_train, y_train)

        #Save the model
        self.save_model()

    def predict(self, sample_messages):
        # Transform the sample dataset
        corpus = []

        #Pre-process the input data
        for i in range(0, len(sample_messages)):
            message = re.sub('[^a-zA-Z]', ' ', sample_messages[i])
            message = self.standardise_text(message)
            corpus.append(message)

        # Extract feature column 'Text'
        sample_features = self.vectorizer.transform(corpus).toarray()

        # Perform classification on the sample dataset
        predictions = self.votingClassifier.predict_proba(sample_features)
        print(self.votingClassifier.predict(sample_features))

        #Assign a label to the prediction based on the threshold of 0.5
        class_predictions = [("smish", p[1]) if p[1] > 0.5 else ("ham", p[1]) for p in predictions]
        return class_predictions

    def predict_classes_only(self, sample_messages):
        # Transform the sample dataset
        corpus = []

        #Pre-process the input data
        for i in range(0, len(sample_messages)):
            message = re.sub('[^a-zA-Z]', ' ', sample_messages[i])
            message = self.standardise_text(message)
            corpus.append(message)

        # Extract feature column 'Text'
        sample_features = self.vectorizer.transform(corpus).toarray()

        # Perform classification on the sample dataset

        return self.votingClassifier.predict(sample_features)