'''
Use GloVe to get semantic information
'''

file = "glove.6B.50d.txt"
from math import floor
import numpy as np
import pickle
import pandas as pd
import time
import csv

start = time.time()
def loadGloveModel(gloveFile):
  print ("Loading Glove Model")


  with open(gloveFile, 'r') as f:
    content = [line.decode('utf-8') for line in f.readlines()]
  model = {}
  for line in content:
    splitLine = line.split()
    word = splitLine[0]
    embedding = np.array([float(val) for val in splitLine[1:]])
    model[word] = embedding
  print ("Done.",len(model)," words loaded!")
  return model


model = loadGloveModel(file)

print (model['hello'])



words = pd.read_table(file, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)


def vec(w):
  return words.loc[w].as_matrix()


print (vec('hello'))

words = words.drop("table", axis=0)
words = words.drop("tables", axis=0)

words_matrix = words.as_matrix()

def find_closest_word(v):
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  i = np.argmin(delta)
  return words.iloc[i].name


print (find_closest_word(model['table']))
#output:  place

#If we want retrieve more than one closest words here is the function:

def find_N_closest_word(v, N, words):
  Nwords=[]
  for w in range(N):
     diff = words.as_matrix() - v
     delta = np.sum(diff * diff, axis=1)
     i = np.argmin(delta)
     Nwords.append(words.iloc[i].name)
     words = words.drop(words.iloc[i].name, axis=0)

  return Nwords

print (find_N_closest_word(model['table'], 10, words))

pickle.dump(words, open("words.p", 'w'))
pickle.dump(model, open("glove.p", 'w'))
print "Pickle created!"


end = time.time()
print "Time elapsed:", str(int(floor((end-start)/60))) + ":" + str(round(((end-start)/60 - floor((end-start)/60))*60))
