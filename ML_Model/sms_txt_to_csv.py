import csv

def main():
    rowList = [["Category", "Message"]]

    with open("Datasets/sms.txt", "r") as f:
        messages = f.readlines()

    for message in messages:
        current = message.strip().split("\t")
        rowList.append(current)

    with open("Datasets/sms.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rowList)

if __name__ == "__main__":
    main()