'''
Compare Machine Learning Algorithms
'''

from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn import tree
from functions import *
from classes import *
from math import floor
import pickle
import numpy
import time

# flag that indicates if the semantic features will be used. 0: no, 1: yes
semantics = 1

start = time.time()

filenames = pickle.load(open("filenames.p", 'r'))
files = pickle.load(open("sem_files.p", 'r'))
kf = KFold(n_splits=5)


# mla01 = neighbors.KNeighborsClassifier(15, weights='uniform')
# mla02 = neighbors.KNeighborsClassifier(15, weights='distance')
mla03 = GaussianNB()
mla04 = tree.DecisionTreeClassifier()
mla05 = RandomForestClassifier()

# mla01 and mla02 have been removed since the calculation took too long
mlas = [mla03, mla04, mla05]

# Calculate scores for each of the machine learning algorithms
for mla in mlas:
	start1 = time.time()
	print "MLA:", mla

	# use 5-fold cross validation
	for train_index, test_index in kf.split(filenames):
		X, y, ids = files[filenames[train_index[0]]].get_features()
		if semantics:
			X = files[filenames[train_index[0]]].get_semantic_features()
		for i in train_index[1:]:
			X1, y1, ids1 = files[filenames[i]].get_features()
			if semantics:
				X1 = files[filenames[i]].get_semantic_features()
			X = numpy.concatenate((X,X1))
			y = numpy.concatenate((y,y1))
			ids = numpy.concatenate((ids,ids1))

		X_test, y_test, ids_test = files[filenames[test_index[0]]].get_features()
		if semantics:
			X_test = files[filenames[test_index[0]]].get_semantic_features()
		for j in test_index[1:]:
			X1, y1, ids1 = files[filenames[i]].get_features()
			if semantics:
				X1 = files[filenames[i]].get_semantic_features()
			X_test = numpy.concatenate((X_test, X1))
			y_test = numpy.concatenate((y_test, y1))
			ids_test = numpy.concatenate((ids_test, ids1))

		# Start prediction and calculate performance
		y_pred = mla.fit(X, y).predict(X_test)
		print "Accuracy:", numpy.mean(y_pred == y_test)

		positives = ids_test[y_pred]
		test_filenames = np.array(filenames)[test_index]
		test_chains = get_chains(test_filenames, files)

		partials = get_partials(test_chains, positives)
		print "Recall of partial chain matches:", len(partials)/float(len(test_chains))*100, len(partials), len(test_chains)

		completes = get_complete(test_chains, positives)
		print "Recall of complete chain matches:", (len(completes)/float(len(test_chains)))*100, len(completes), len(test_chains)

		end1 = time.time()
		print "Time elapsed:", str(int(floor((end1-start1)/60))) + ":" + str(round(((end1-start1)/60 - floor((end1-start1)/60))*60))

		print "--------------------------------------------------------------"
		print

# print time it took to calculate
end = time.time()
print "Time elapsed:", str(int(floor((end-start)/60))) + ":" + str(round(((end-start)/60 - floor((end-start)/60))*60))
