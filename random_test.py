'''
Calculate random accuracy.
'''

from functions import *
from classes import *
from math import floor
import pickle
import numpy
import time

start = time.time()

for i in range(10):
    print i

filenames = pickle.load(open("filenames.p", 'r'))
files = pickle.load(open("sem_files.p", 'r'))

_, y, _ = files[filenames[0]].get_features()

for i in range(1,len(filenames)):
    _, y1, _ = files[filenames[i]].get_features()
    y = numpy.concatenate((y,y1))


tmp = numpy.ones(y.shape)
print "CHANCE: ", numpy.mean(y == tmp)

tmp = numpy.zeros(y.shape)
print "CHANCE: ", numpy.mean(y == tmp)
