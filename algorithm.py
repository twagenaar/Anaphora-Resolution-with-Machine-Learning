'''
Creating the features
'''

from collections import defaultdict, Counter
from classes import *
from math import floor
import random
import pickle
import numpy
import glob
import time
import re

start = time.time()

def cleantaglist(taglist):
	'''
	clean the taglist of double tags
	'''
	done = []
	tags = []
	for tag in taglist:
		temp = tag.get_tuple()
		if temp not in done:
			done.append(temp)
			tags.append(tag)
	return tags

def getfeatures(fileinfo):
	'''
	collect the features for the given file
	'''
	nps = fileinfo.get_nps()
	nps = cleantaglist(nps)
	chains = fileinfo.get_chains()
	start = TagInfo("", 0, [0])
	sortedlist = sorted(nps, key=lambda obj: start.distance(obj, fileinfo))
	features = []
	ids = []
	y = []
	for i, np1 in enumerate(sortedlist[:]):
		for np2 in sortedlist[:i]:
			if np1 != np2:
				dist = np1.distance(np2, fileinfo)
				match = np1.str_match(np2)
				pmatch = np1.partial_str_match(np2)
				y.append(np1.find_match(np2, chains))
				features.append([dist, match, pmatch])
				ids.append([np1, np2])
	return numpy.matrix(features), numpy.array(y), numpy.matrix(ids)

# collect the filenames
files = pickle.load(open('new_files.p', 'r'))
filenames = glob.glob("ontonotes-release-5.0/data/files/data/english/annotations/bn/*/*/*.coref")
random.shuffle(filenames)
pickle.dump(filenames, open("filenames.p", 'w'))

counter = 0
print "Number of files:", len(filenames)
for filename in filenames:
	X1, y1, ids1 = getfeatures(files[filename])
	fileinfo = files[filename]
	fileinfo.set_features(X1, y1, ids1)
	files[filename] = fileinfo
	counter += 1
	if counter % 100 == 0:
		print "Current file:", counter

pickle.dump(files, open("new_new_files.p", 'w'))
print "Pickle Created!"

# print time it took to calculate
end = time.time()
print "Time elapsed:", str(int(floor((end-start)/60))) + ":" + str(((end-start)/60 - floor((end-start)/60))*60)
