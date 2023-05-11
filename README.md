# Osnove-strojnog-u-enja---lv
Zadatak 8.4.1 MNIST podatkovni skup za izgradnju klasifikatora rukom pisanih znamenki
dostupan je u okviru Keras-a. Skripta zadatak_1.py uˇcitava MNIST podatkovni skup te podatke
priprema za uˇcenje potpuno povezane mreže.
1. Upoznajte se s uˇcitanim podacima. Koliko primjera sadrži skup za uˇcenje, a koliko skup za
testiranje? Kako su skalirani ulazni podaci tj. slike? Kako je kodirana izlazne veliˇcina?
2. Pomo´cu matplotlib biblioteke prikažite jednu sliku iz skupa podataka za uˇcenje te ispišite
njezinu oznaku u terminal.
3. Pomo´cu klase Sequential izgradite mrežu prikazanu na slici 8.5. Pomo´cu metode
.summary ispišite informacije o mreži u terminal.
4. Pomo´cu metode .compile podesite proces treniranja mreže.
8.5 Izvještaj s vježbe 75
5. Pokrenite uˇcenje mreže (samostalno definirajte broj epoha i veliˇcinu serije). Pratite tijek
uˇcenja u terminalu.
6. Izvršite evaluaciju mreže na testnom skupu podataka pomo´cu metode .evaluate.
7. Izraˇcunajte predikciju mreže za skup podataka za testiranje. Pomo´cu scikit-learn biblioteke
prikažite matricu zabune za skup podataka za testiranje.
8. Pohranite model na tvrdi disk.
Zadatak 8.4.2 Napišite skriptu koja ´ce uˇcitati izgra ¯ denu mrežu iz zadatka 1 i MNIST skup
podataka. Pomo´cu matplotlib biblioteke potrebno je prikazati nekoliko loše klasificiranih slika iz
skupa podataka za testiranje. Pri tome u naslov slike napišite stvarnu oznaku i oznaku predvid¯enu
mrežom.
Zadatak 8.4.3 Napišite skriptu koja ´ce uˇcitati izgra ¯ denu mrežu iz zadatka 1. Nadalje, skripta
treba uˇcitati sliku test.png sa diska. Dodajte u skriptu kod koji ´ce prilagoditi sliku za mrežu,
klasificirati sliku pomo´cu izgra ¯ dene mreže te ispisati rezultat u terminal. Promijenite sliku
pomo´cu nekog grafiˇckog alata (npr. pomo´cuWindows Paint-a nacrtajte broj 2) i ponovo pokrenite
skriptu. Komentirajte dobivene rezultate za razliˇcite napisane znamenke.
