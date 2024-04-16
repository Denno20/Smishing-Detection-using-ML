from combinedClassifier import CombinedClassifier
import pandas as pd

def main():
    model = CombinedClassifier()
    model.load_dataset("../Datasets/sms.csv")

    model.train()
    # model.load_model("voting_classifier.joblib", "vectorizer.joblib")
    # unseen = ["""Dear Customer,
    # At admiral, ensuring the accuracy of our records is integral to providing you with seamless service. As part of our commitment to maintaining up-to-date information, we kindly request your assistance in verifying your current location.
    # Your cooperation in confirming your location annually helps us ensure that our services remain tailored to your needs and that we continue to serve you effectively.
    # To confirm your current location, please click on the following link: http://my.account.admiral.com. This simple step will help us update our records accordingly.
    # Should you have any concerns or questions regarding this request, please don't hesitate to reach out to our support team at Admiral.
    # We sincerely appreciate your prompt attention to this matter and thank you for being a valued customer of Admiral.
    # Best Regards,
    # Admiral Support Team""", """URGENT: Your bank account has been compromised. Please click the following link to verify your account and prevent unauthorized access: https://secure.bank-login.com/account=123456789&sessionid=abcdef123456&redirect=http://phishingsite.com/login. Do not ignore this message, act now to secure your funds.""",
    # "Go until jurong point, crazy.. Available only in bugis n great world la e buffet... Cine there got amore wat...",
    #     "Ok lar... Joking wif u oni...",
    #     "Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's",
    #     "Your IRS tax refund is pending acceptance. Must accept within 24 hours: http://bit.ly/sdfsdf", "Check out this amazing offer! You have won a prize! Click the Link below or CALL 043505",
    #     "For security reasons, we've restricted your account. Please visit https://bankverify to verify now.",
    #     "Paypal: we have detected unusual activity from this account please sign in https://login-id9374958.com?p=1 to confirm the transactions",
    #     "Your Apple ID Account has been locked due to unauthorised login attempts. Please log in here and verify your information:http://update-apple.uk",
    #     "You have won $5000. The prize needs to be claimed ASAP. Please reply with your bank information so we can deposit the money into your account.",
    #     "Hello mate, your FEDEX Package with tracking code GB-6412-GH83 is waiting for you to set delivery preferences: c7dvf.info/GFdFtk12viiM",
    #     "Your credit card has been deactivated. Please visit bit.ly/sZ68GJ to reactivate it",
    #     "Deae valued netflix user. The monthly automatic payment failed. To avoid services suspension and restore access join us at: netflixhelpsupport.lpages.co/support",
    #     "Royal Mail: your package has a £2.99 shipping fee, to pay this now visit http://royalmail-redelivery.support. Actions will be taken if you do not pay this fee."]

    unseen = open("testdata.txt", "r").readlines()

    predicted = model.predict(pd.Series(unseen))

    YES = 0
    NO = 0
    TOTAL = len(predicted)

    with open("results.csv", "w") as f:
        for p in predicted:
            if (p[0] == "smish"):
                f.write(str(p[0]) + "," + str(p[1]) + "," + "YES" + "\n")
                YES += 1
            else:
                f.write(str(p[0]) + "," + str(p[1]) + "," + "NO" + "\n")
                NO += 1

    print(f"Number of YES: {YES}")
    print(f"Number of NO: {NO}")
    print(f"Total: {TOTAL}")

if __name__ == "__main__":
    main()