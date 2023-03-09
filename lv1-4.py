file=open("song.txt")
dictionary={}
count=0

for line in file:
    for word in line.split():
        if word in dictionary:
            dictionary[word]=dictionary[word]+1
        else:
            dictionary[word]=1

for key in list(dictionary.keys()):
    print(key, ":", dictionary[key])
    if dictionary[key]==1:
        count +=1

print("Broj riječi koje se samo jednom ponavljaju: "+ str(count))


file.close()
