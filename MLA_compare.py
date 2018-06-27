'''
Compare Machine Learning Algorithms
'''

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn import neighbors
from sklearn import tree
from sklearn import svm
from functions import *
from classes import *
from math import floor
import pickle
import numpy
import time

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

# mla01, mla02,
mlas = [mla03, mla04, mla05]
# mlas = [mla02]

for mla in mlas:
	start1 = time.time()
	print "MLA:", mla

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

		print "starting prediction", X.shape
		y_pred = mla.fit(X, y).predict(X_test)
		print "Accuracy:", numpy.mean(y_pred == y_test), sum(y_pred), sum(y_test)

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


end = time.time()
print "Time elapsed:", str(int(floor((end-start)/60))) + ":" + str(round(((end-start)/60 - floor((end-start)/60))*60))
