import pandas as pd
from combinedClassifier import CombinedClassifier
from sklearn.metrics import f1_score, confusion_matrix,accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats

def get_confusion_matrix(predicted, data, title):
    # Creates a confusion matrix
    cm = confusion_matrix(predicted,data.Class)
    print(accuracy_score(predicted,data.Class))
    print(classification_report(predicted,data.Class))

    # Transform to df for easier plotting
    cm_df = pd.DataFrame(cm,
                        index=['TP', 'FP'],
                        columns=['FN', 'TN'])

    plt.figure(figsize=(5.5,4))
    sns.heatmap(cm_df, annot=True,fmt='g')
    plt.title(title)
    plt.ylabel('Predicted class')
    plt.xlabel('Actual class')
    plt.show()

def main():
    data = pd.read_csv("validation_data.csv", encoding="latin-1")
    model = CombinedClassifier()
    # model.load_model("voting_classifier.joblib", "vectorizer.joblib")
    model.load_dataset("../Datasets/public.csv")
    model.train()
    public_dataset_only_predictions = model.predict_classes_only(pd.Series(data.Text))
    get_confusion_matrix(public_dataset_only_predictions, data, 'Voting Classifier - public dataset only')

    with open("before_output.txt", "w") as f:
        s = ""
        for val in public_dataset_only_predictions:
            s += str(val) + "\n"
    
        f.write(s.strip("\n"))

    model = CombinedClassifier()
    model.load_dataset("../Datasets/sms.csv")
    model.train()

    combined_dataset_predictions = model.predict_classes_only(pd.Series(data.Text))
    get_confusion_matrix(combined_dataset_predictions, data, 'Voting Classifier with combined dataset')

    with open("after_output.txt", "w") as f:
        s = ""
        for val in combined_dataset_predictions:
            s += str(val) + "\n"
    
        f.write(s.strip("\n"))


if __name__ == "__main__":
    main()