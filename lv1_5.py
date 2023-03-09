file= open("SMSSpamCollection.txt")

spam_messages=0
ham_messages=0
spam_words=0
ham_words=0
spam_exclamation=0

for line in file:
    words=line.split()
    if words[0] == "spam":
        spam_messages = spam_messages + 1
        spam_words = spam_words + len(spam_words)
        if line.endswith("!"):
            spam_exclamation += 1
    elif words[0] == "ham":
        ham_messages= ham_messages + 1
        ham_words = ham_words + len(ham_words)
    
file.close()
print("Average for spam is: ", spam_words/spam_messages)
print("Average for ham is: ", ham_words/ham_messages)
print("Message that is spam and ends with exclamation count is: ", spam_exclamation)