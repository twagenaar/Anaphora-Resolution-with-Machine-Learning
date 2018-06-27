'''
Compare Noun Phrase Extractors
'''
from collections import defaultdict
from nltk.corpus import stopwords
from textblob import TextBlob
# from spacy.lang.en import English
from classes import *
import numpy
import spacy
import phrasemachine
import pickle
import nltk
import glob
import time
import re
start = time.time()

# def get_nps(text):
#   blob = TextBlob(text)
#   nps = blob.noun_phrases 
	# return nps

def get_nps(text):
	# nlp = English()
	nlp = spacy.load('en')
	nps = []
	# print text
	doc = nlp(unicode(text))
	for np in doc.noun_chunks:
	  # print np.text
	  nps.append(np.text)
	return nps



files = pickle.load(open("files.p", 'r'))
new_files = defaultdict(None)
filenames = glob.glob("ontonotes-release-5.0/data/files/data/english/annotations/bn/*/*/*.coref")
print "Number of files:", len(filenames)
counter = 0
correct = []
for filename in filenames[:]:
  fileinfo = files[filename]
  text = files[filename].get_processed()
  text = re.sub('<DOC.*\>', '', text).strip()

  # Get the NPs
  nps = get_nps(text)

  # Statistics
  chains = files[filename].get_chains()
  np_tags1 = [chains[x] for x in chains]
  np_tags = []
  for l in np_tags1:
    np_tags += l
  # print filename, np_tags
  np_tags = [x.get_text().lower() for x in np_tags if x]
  # correct += [x.lower() in nps for x in np_tags]
  
  # Processing
  np_list = []
  for np in nps: 
    # print len(nps), np
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

      # print sentence
      for occurance in occurances:
        i = [occurance]
        check = True
        while np_test:
          item = np_test.pop(0)
          if occurance < len(copy)-1:
            occurance += 1
          # print occurance, len(copy), ' '.join(copy), ';', np
          if copy[occurance] != item:
            check = False
            # print "BREAK1", np, ';', sentence, ';', item, occurance, copy.index(item)
            i = []
            break
          else:
            i.append(occurance)
        if check:
          # print "BREAK2", np, ';', sentence, i
          np_tag = TagInfo(np, senti, i)
          np_list.append(np_tag)
          break
  fileinfo.set_nps(np_list)
  # TODO: Check correct!
  # for tag in chains:
  #   correct += [np.within(tag) for np in np_list]
  chain_items = []
  for key in chains:
    chain_items += chains[key]
  for tag in chain_items:
    correct += [sum([np.within(tag) for np in np_list]) > 0]

  new_files[filename] = fileinfo
  if counter % 100 == 0:
    print "Currect file:", counter
  counter += 1



print "AVERAGE NUMBER OF NPS FOUND:", numpy.mean(correct)

# pickle.dump(new_files, open("new_files.p", "w"))
# print "Pickle created"
# end = time.time()
# print "Time elapsed:", end-start, "sec"
