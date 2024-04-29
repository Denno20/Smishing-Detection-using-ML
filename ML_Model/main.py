from model import SMSModel

def main():
    model = SMSModel("Datasets/sms.csv", 30)
    print(model.train())
    print(model.evaluate())
    
    predict_msg = ["Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
          "Ok lar... Joking wif u oni...",
          "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]


    outcome = model.predict_sms(predict_msg=predict_msg)
    print("Hello world")
    print(outcome)


if __name__=="__main__":
    main()


