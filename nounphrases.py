'''
Noun Phrase Extractor
'''
from collections import defaultdict
from nltk.corpus import stopwords
from textblob import TextBlob
from classes import *
from math import floor
import numpy
# import spacy
import phrasemachine
import pickle
import nltk
import glob
import time
import re

# nltk.download()

def get_pronouns(text):
  blob = TextBlob(text)
  return [word for (word,tag) in blob.tags if (tag == "PRP" or tag == "PRP$" or tag == "NN")]

# def get_nps(text):
#   nlp = spacy.load('en')
#   nps = []
#   doc = nlp(unicode(text))
#   for np in doc.noun_chunks:
#     nps.append(np.text)
#   return nps

# Best functioning NP Extractor
# def get_nps(text):
#   nps = phrasemachine.get_phrases(text)
#   return list(nps['counts']) + get_pronouns(text)

def get_nps(text):
  blob = TextBlob(text)
  return blob.noun_phrases + get_pronouns(text)

start = time.time()

files = pickle.load(open("files.p", 'r'))
new_files = defaultdict(None)
filenames = glob.glob("ontonotes-release-5.0/data/files/data/english/annotations/bn/*/*/*.coref")
print "Number of files:", len(filenames)
counter = 0
correct = []
correct2 = []
# Extract the markables from each file
for filename in filenames:
  fileinfo = files[filename]
  text = files[filename].get_processed()
  text = re.sub('<DOC.*\>', '', text).strip()

  # Get the NPs
  nps = list(set(get_nps(text)))

  # Statistics
  chains = files[filename].get_chains()
  np_tags1 = [chains[x] for x in chains]
  np_tags = []
  for l in np_tags1:
    np_tags += l
  np_tags = [x.get_text().lower() for x in np_tags if x]
  # correct += [x.lower() in nps for x in np_tags]

  # Processing, get line and position in line
  np_list = []
  for np in nps:
    orig = fileinfo.get_original().split('\n')
    sen_lens = []
    indices = []
    sen_indices = []
    for senti, sentence in enumerate(orig):
      sentence = re.sub('</?[\w\s ="@\-\_\.]*\>', '', sentence).strip()
      check = True
      copy = list(sentence.lower().split(' '))
      occurances = [i for i, x in enumerate(copy) if x == np.split(' ')[0]]
      np_test = np.split(' ')[1:]
      for occurance in occurances:
        i = [occurance]
        check = True
        while np_test:
          item = np_test.pop(0)
          if occurance < len(copy)-1:
            occurance += 1
          if copy[occurance] != item:
            check = False
            i = []
            break
          else:
            i.append(occurance)
        if check:
          np_tag = TagInfo(np, senti, i)
          np_list.append(np_tag)
          break
  fileinfo.set_nps(np_list)

  # Calculate how many of the annotated strings are found.
  chain_items = []
  for key in chains:
    chain_items += chains[key]
  for tag in chain_items:
    correct += [sum([np.within(tag) for np in np_list]) > 0]
    correct2 += [sum(np.get_tuple() == tag.get_tuple() for np in np_list if (np and tag)) > 0]

  new_files[filename] = fileinfo
  if counter % 100 == 0:
    print "Currect file:", counter
  counter += 1



print "AVERAGE NUMBER OF NPS FOUND:", numpy.mean(correct)
print "AVERAGE NUMBER OF NPS FOUND (COMPLETE MATCH):", numpy.mean(correct2)

pickle.dump(new_files, open("new_files.p", "w"))
print "Pickle created"
end = time.time()
print "Time elapsed:", str(int(floor((end-start)/60))) + ":" + str(((end-start)/60 - floor((end-start)/60))*60)
