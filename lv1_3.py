    
list = []

while True:
    c=input("unesi broj")
    if c == "Done":
        break
    elif c.isnumeric()== False:
        print("broj nije u intervalu")
    else:
        list.append(int(c))


print("Ukupni broj unesenih brojeva: ", len(list))
print("Minimalna vrijednost: ", min(list))
print("Maksimalna vrijednost: ", max(list))
