import requests
from phonemizer import phonemize
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

from phonemizer.backend import festival
from phonemizer.backend import EspeakBackend
from phonemizer.punctuation import Punctuation
from phonemizer.separator import Separator

import bisect

def clean_strs(strings):
  fixed = [i.lower().replace(r'\n',' ').replace(r'\[[^\[\]]*]','').replace(r"\'\w*",'').replace(r'[^\w\d\s]+','').strip() for i in strings]
  return fixed

def process_file(input_file):
  lines = []
  lines_wo = []
  with open(input_file , 'r') as f:
    for line in f.readlines():
      if line == '\n':
        pass
      else:
        lines.append(line)
        lines_wo.append(line + ' [sep]')
    
  s = [' '.join(lines).replace('\n','')]
  s = s[0].lower()
  text = Punctuation(';:,.!"?()-').remove(s)

  s_wo = [' '.join(lines_wo).replace('\n','')]
  s_wo = s_wo[0].lower()
  text_wo = Punctuation(';:,.!"?()-').remove(s_wo)

  phn = phonemize(
    text,
    language='en-us',
    backend='festival',
    separator=Separator(phone='_', word=' ', syllable=' @@'),
    strip=True,
    preserve_punctuation=True,
    njobs=4)
  
  phones = phn.split(' ')

  new = []
  for i,v in enumerate(phones):
    if v[0] != '@':
      new.append(v)
    else:
      if '_' in v:
        splits = v.split('_',1)
        new[-1] = new[-1] + '_' + splits[0][2:]
        new.append('@@' + splits[1])
      else:
        new.append(v)
  
  phones = ' '.join(new).split(' ')

  for i,v in enumerate(text_wo.split(' ')):
    if v == '[sep]':
      phones.insert(i+1,'[sep]')
  
  final_phones = ' '.join(phones)

  f = open( "data/lyrics.txt", "w")
  f.write(text_wo)

  f = open( "data/syllables.txt", "w")
  f.write(final_phones)