'''
The python classes used
'''

class FileInfo:
  '''
  class containing information about a file
  - features
  - semantic features
  - processed and unprocessed text
  - coreference chains
  - markables extracted from the text
  '''
  def __init__(self, old_text, new_text):
    self.original = old_text
    self.processed = new_text

  def set_features(self, features, classes, ids):
    self.X = features
    self.y = classes
    self.ids = ids

  def add_semantic_features(self, semantic_features):
    self.X_sem = semantic_features

  def get_semantic_features(self):
    return self.X_sem

  def get_features(self):
    return self.X, self.y, self.ids

  def get_word(self, sentence, word):
    return self.get_line(sentence).split(" ")[word]

  def get_processed(self):
    return '\n'.join(self.processed)

  def get_original(self):
    return self.original

  def get_line(self, index):
    if index >= len(self.processed):
      return None
    return self.processed[index]

  def get_chains(self):
    return self.chains

  def set_chains(self, chains):
    self.chains = chains

  def get_nps(self):
    return self.nps

  def set_nps(self, nps):
    self.nps = nps

  def get_sentence_lengths(self):
    return self.lens

  def set_sentence_lenghts(self, lengths):
    self.lens = lengths

  def __repr__(self):
    return "FILEINFO(%s)"%(self.processed[0])

class TagInfo:
  '''
  Contains information about a tag and gives you the possibility to calculate the get_features
  - text of the markable
  - sentence and word indices of the markable
  - calculate the features between this markable and another
  '''
  def __init__(self, text, sentence_index, word_indices):
    self.text = text
    self.sentence = sentence_index
    self.words = word_indices

  def find_match(self, taginfo, annotations):
    check = False
    for tagid in annotations:
      tags = annotations[tagid]
      for tag in tags:
        w1 = self.within(tag)
        w2 = taginfo.within(tag)
        t1 = self
        t2 = taginfo
        if w2:
          w1,w2 = w2,w1
          t1,t2 = t2,t1
        if w1:
          for tag2 in tags:
            w3 = t2.within(tag2)
            if w3:
              return True
    return False


  def get_tuple(self):
    return (self.text, self.sentence, self.words)

  def get_text(self):
    return self.text

  def get_indices(self):
    return self.words

  def within(self, tagInfo):
    if tagInfo:
      i = tagInfo.get_indices()
      j = self.words
      s1 = tagInfo.get_tuple()[1]
      s2 = self.sentence
      return (i[0] <= j[0] and i[-1] >= j[-1]) and (s1 == s2)
    return False

  def distance(self, tagInfo, fileinfo):
    '''
    Calculate how many words are between two tags.
    '''
    s1 = self.sentence
    s2 = tagInfo.sentence

    i1 = self.words
    i2 = tagInfo.words

    # switch item 1 and 2 so item 1 is the first np encountered in the text
    if s1 > s2 or (s1 == s2 and i1 > i2):
      s1,s2 = s2,s1
      i1,i2 = i2,i1

    sen_lens = fileinfo.get_sentence_lengths()

    used = [sen_lens[i] for i in range(s1, s2+1)]
    used[0] = used[0] - i1[-1]
    used[-1] = i2[0]

    return sum(used)


  def str_match(self, tagInfo):
    return self.text == tagInfo.text

  def partial_str_match(self, tagInfo):
    return (self.text in tagInfo.text) or (tagInfo.text in self.text)

  def equal(self, tagInfo):
    return self.get_tuple() == tagInfo.get_tuple()


  def __str__(self):
    return "TAG(%s,%d,%d-%d)"%(self.text,self.sentence,self.words[0],self.words[-1])

  def __repr__(self):
    return "TAG(%s,%d,%s)"%(self.text,self.sentence,str(self.words))
