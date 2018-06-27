'''
Load in ontonotes files
'''

from collections import defaultdict
from classes import *
import xml.etree.ElementTree as ET
import glob
import time
import pprint
import re
import pickle

start = time.time()

filenames = glob.glob("ontonotes-release-5.0/data/files/data/english/annotations/bn/*/*/*.coref")

def open_file(filename):
  f = open(filename, 'r')
  new_text = []
  old_text = ""
  for line in f.readlines():
    old_text += line
    new_line = re.sub('</?[\w\s ="@\-\_\.]*\>', '', line).strip()
    new_text.append(new_line)
  fileinfo = FileInfo(old_text, new_text)
  return fileinfo

def get_chains(filename, fileinfo):
  chains = defaultdict(list)
  tree = ET.parse(filename)
  root = tree.getroot()
  nodes = [root]
  orig = fileinfo.get_original().split('\n')
  correct = 0
  total = 0
  unprocessed = []
  while nodes:
    for child in nodes.pop(0):
      nodes.append(child)
      if child.tag == "COREF":
        i = []
        text = ""
        if child.text:
          text = child.text
        children = [child]
        while children:
          for baby in children.pop(0):
            if baby.text:
              text += baby.text
            if baby.tail:
               text += baby.tail
            children.append(baby)
        text = text.split(' ')

        sen_lens = []
        indices = []
        sen_indices = []
        for senti, sentence in enumerate(orig):
          sentence = re.sub('</?[\w\s ="@\-\_\.]*\>', '', sentence).strip().split(' ')
          copy = list(sentence)
          check = True
          occurances = [i for i, x in enumerate(copy) if x == text[0]]
          text_test = text[1:]
          for occurance in occurances:
            i = [occurance]
            check = True
            while text_test:
              item = text_test.pop(0)
              if occurance < len(copy)-1:
                occurance += 1
              if copy[occurance] != item:
                check = False
                i = []
                break
              else:
                i.append(occurance)

            if check:
              # sen_lens.append(len(sentence))
              indices.append(i)
              sen_indices.append(senti)

        tag = None
        for i, index in enumerate(sen_indices):
          if text[-1] + "</COREF>" in re.sub('<[\w\s ="@\-\_\.]*\>', '', orig[index]).split(' '):
            tag = TagInfo(" ".join(text), index, indices[i])
        chains[child.attrib["ID"]].append(tag)

  return chains


def get_sentence_lengths(fileinfo):
  # print fileinfo.get_processed()
  text = fileinfo.get_processed()
  lens = [len(s.split(' ')) if s else 0 for s in text.split('\n')]
  # print lens
  return lens

files = defaultdict(None)
counter = 0
print "Number of files:", len(filenames)
for filename in filenames[:]:
  fileinfo = open_file(filename)

  chains = get_chains(filename, fileinfo)
  fileinfo.set_chains(chains)
  sentence_lengths = get_sentence_lengths(fileinfo)
  fileinfo.set_sentence_lenghts(sentence_lengths)
  files[filename] = fileinfo
  if counter % 100 == 0:
    print "Current file:", counter
  counter += 1

pickle.dump(files, open("files.p", "w"))
print "Pickle created"
end = time.time()
print "Time elapsed:", end-start, "sec"
# print files['ontonotes-release-5.0/data/files/data/english/annotations/bn/cnn/01/cnn_0178.coref'].get_chains()
