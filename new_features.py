'''
Add the semantic features.
'''

from collections import defaultdict
from nltk.corpus import wordnet
from itertools import chain
from functions import *
from classes import *
from math import floor
import numpy as np
import pickle
import string
import time
import re

start = time.time()

files = pickle.load(open('new_new_files.p', 'r'))
filenames = pickle.load(open("filenames.p", 'r'))
model = pickle.load(open("glove.p", 'r'))
words = pickle.load(open("words.p", 'r'))
words_matrix = words.as_matrix()

def get_glove_score(context1, context2):
  '''
  get the glove score of the context of the markables
  '''
  score = 0
  for word in context1:
    if word in model:
      closest = find_N_closest_word(model[word], 4, words)
      for close in closest:
        if close in context2:
          score += 1
  if context1:
    return score/len(context1)*100
  return 0

def get_normal_score(context1, context2):
  '''
  get the percentage of the context of the markables that matches
  '''
  score = 0
  for word in context1:
    if word in context2:
      score += 1
  if context1:
    return score/len(context1)*100
  return 0

def get_synonym_score(context1, context2):
  '''
  get the percentage of the context that matches synonyms of the context of the other markables
  '''
  score = 0
  for word in context1:
    b = False
    synonyms = wordnet.synsets(word)
    syns = set(chain.from_iterable([word.lemma_names() for word in synonyms]))
    for word2 in syns:
      if word2 in context2:
        score += 1
        b = True
        break
      if b: break
  if context1:
    return score/len(context1)*100
  return 0

def clean_list(l):
  '''
  remove nonwords and empty strings from the context
  '''
  table = string.maketrans("","")
  cleaned = [word.translate(table, string.punctuation) for word in l]
  cleaned = filter(None, cleaned)
  return cleaned

def get_context(taginfo, window, fileinfo):
  '''
  get the context of a markable with a windowsize of "window"
  '''
  context = []
  tag = taginfo.get_tuple()
  sen1 = tag[1]
  sen2 = tag[1]
  begin = tag[2][0]-window
  end = tag[2][-1]+window+1
  if begin < 0:
    line = fileinfo.get_line(sen1).split(' ')
    context += line[:begin+window]
    while begin < 0 and sen1 > 1:
      sen1 -= 1
      line = fileinfo.get_line(sen1).split(' ')
      len1 = len(line)
      if abs(begin) <= len1:
        context = line[begin:] + context
        break
      else:
        begin += len1
        context = line + context
  else:
    context += fileinfo.get_line(sen1).split(' ')[begin:tag[2][0]]
  line = fileinfo.get_line(sen2).split(' ')
  len2 = len(line)
  while end >= len2:
    sen2 += 1
    new_line = fileinfo.get_line(sen2)
    if new_line:
      new_line = new_line.split(' ')
    if new_line != None:
      line += new_line
      len2 = len(line)
    else: break
  context += line[tag[2][-1]+1:min(end, len2)]
  return clean_list(context)

def add_features(X, y, ids, fileinfo, model, window):
  '''
  create the features with the corresponding classes and markable pairs
  '''
  X_sem = np.zeros((X.shape[0], X.shape[1]+2))
  for i in range(X.shape[0]):
    taginfo1 = ids[i,0]
    taginfo2 = ids[i,1]
    context1 = get_context(taginfo1, window, fileinfo)
    context2 = get_context(taginfo2, window, fileinfo)
    if not context1:
        print "c1:", context1, taginfo1
        print "c2:", context2, taginfo2
        print fileinfo.get_processed()
    # glove_score = get_glove_score(context1, context2)
    normal_score = get_normal_score(context1, context2)
    syn_score = get_synonym_score(context1, context2)
    X_sem[i,:-2] = X[i,:]
    # X_sem[i,-1] = glove_score
    X_sem[i,-2] = normal_score
    X_sem[i,-1] = syn_score

  return X_sem

counter = 0
new_files = defaultdict(None)
for filename in filenames[:]:
  fileinfo = files[filename]
  X, y, ids = fileinfo.get_features()
  X_sem = add_features(X, y, ids, fileinfo, model, 5)
  fileinfo.add_semantic_features(X_sem)
  new_files[filename] = fileinfo
  if counter % 10 == 0:
    print "Currect file:", counter
  counter += 1

pickle.dump(new_files, open("sem_files.p", 'w'))
print "Pickle Created!"

# print the time it took to calculate
end = time.time()
print "Time elapsed:", str(int(floor((end-start)/60))) + ":" + str(round(((end-start)/60 - floor((end-start)/60))*60))
