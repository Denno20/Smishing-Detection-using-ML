from model import LSTMClassifier

def main():

    data = ["Hello there", "Dear Customer, click the link below to update your account information https://1fgjfjd..com"]


    
    model = LSTMClassifier()
    model.load_model("model.keras", "tokeniser.pkl")
    predictions = model.predict(data)

    print(predictions)

    # model.load_dataset("../Datasets/sms.csv")
    # model.train(batch_size=256, epoch=30, validation_split=0.20)
    # model.evaluate()
    # model.predict(data)
    # predictions = model.predict(data)

    # print(predictions)

if __name__ == "__main__":
    main()



