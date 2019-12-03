import csv

path = '../rmac/Train_GCC-training.tsv'

with open(path, 'r') as f:
    d = csv.reader(f, delimiter="\t")
    data = []
    count = 0
    for i in d:
        count += 1
        data.append(i[0])
        if count == 1000000:
            break

with open('corpus.txt', 'w') as f:
    for i in data:
        f.write(i + ' ')
print ('done')