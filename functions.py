'''
A collection of the functions used
'''

import numpy as np

def get_chains(filenames, files):
  '''
  extract the coreference chains from the annotated text
  '''
  chains = []
  for filename in filenames:
    fileinfo = files[filename]
    temp_chains = fileinfo.get_chains()
    for _, li in temp_chains.items():
      chains.append(li)
  return chains

def get_partials(chains, positives):
  '''
  get the recall of the partial chain matches
  '''
  results = []
  for chain in chains:
    b = False
    for item in chain:
      for pos in positives:
        if pos[0,0] and pos[0,1] and item:
          if pos[0,0].within(item):
            for item2 in chain:
              if pos[0,1].within(item2):
                b = True
                results.append(chain)
                break
              if b: break
          if pos[0,1].within(item):
            for item2 in chain:
              if pos[0,0].within(item2):
                b = True
                results.append(chain)
                break
              if b: break
        if b: break
      if b: break
  return results

def get_complete(chains, positives):
  '''
  get the recall of the complete chain matches
  '''
  results = []
  for chain in chains:
    b = False
    for item in chain:
      completed_chain = set()
      for pos in positives:
        if pos[0,0] and pos[0,1] and item:
          if pos[0,0].within(item):
            for item2 in chain:
              if pos[0,1].within(item2):
                completed_chain.add(item)
                completed_chain.add(item2)
                if list(completed_chain) == chain:
                  results.append(chain)
                  b = True
              if b: break
        if b: break
      if b: break

  return results

# def to_matches(chains):
#   '''
#
#   '''
#   results = []
#   for chain in chains:
#     for i, item in enumerate(chain):
#       for item2 in chain[:i]:
#         result.append([item, item2])
#   return results

# def run_tests(test_chains, positives):
#     '''
#
#     '''
#     partials = get_partials(test_chains, positives)
#     print "Recall of partial chain matches:", len(partials)/float(len(test_chains))*100
#     print "Precision of partial chain matches:", len(partials)/float(len(positives))*100
#
#     completes = get_complete(test_chains, positives)
#     print "Recall of complete chain matches:", (len(completes)/float(len(test_chains)))*100
#     print "Precision of complete chain matches:", (len(completes)/float(len(positives)))*100

def find_closest_word(v):
  '''
  find the closest word in the glove word vector space
  '''
  diff = words_matrix - v
  delta = np.sum(diff * diff, axis=1)
  i = np.argmin(delta)
  return words.iloc[i].name

def find_N_closest_word(v, N, words):
  '''
  find the n closest words in the glove word vector space
  '''
  Nwords=[]
  for w in range(N):
     diff = words.as_matrix() - v
     delta = np.sum(diff * diff, axis=1)
     i = np.argmin(delta)
     Nwords.append(words.iloc[i].name)
     words = words.drop(words.iloc[i].name, axis=0)

  return Nwords

def vec(w):
  return words.loc[w].as_matrix()
