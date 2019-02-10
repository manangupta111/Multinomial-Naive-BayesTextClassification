from pprint import pprint
from sklearn.model_selection import train_test_split
from textblob.classifiers import NaiveBayesClassifier

#filename = 'Data/Yelp_Labelled.txt'
filename = 'Data/IMDB_Labelled.txt'

lines = [line.rstrip('\n') for line in open(filename)]

rows = []
for line in lines:
	sentence = line.split("#")
	rows.append(tuple(sentence))


train, test = train_test_split(rows, test_size = 0.25)


model = NaiveBayesClassifier(train)
print(model)

print(model.accuracy(test))
