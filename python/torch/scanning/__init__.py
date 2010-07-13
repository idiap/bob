from libpytorch_scanning import *

# Brings a more pythonic usability to the several Torch constructions in this
# submodule:
def pattern_str(p):
  return '(x=%d, y=%d, width=%d, height=%d, confidence=%.3e)' % (p.x, p.y, p.width, p.height, p.confidence)
Pattern.__str__ = pattern_str

def patternlist_len(p):
  return p.size()
PatternList.__len__ = patternlist_len

def patternlist_getitem(p, k):
  if k < 0: raise IndexError("Only positive indexes are supported")
  if k >= len(p): raise IndexError("PatternList only has %d entries" % len(p))
  return p.get(k)
PatternList.__getitem__ = patternlist_getitem

class PatternListIterator:
  def __init__(self, p):
    self.list = p
    self.current = 0
  def __iter__(self):
    return PatternListIterator(self.list)
  def next(self):
    self.current += 1
    if self.current > len(self.list): raise StopIteration
    return self.list[self.current-1]

def patternlist_iter(p):
  return PatternListIterator(p)
PatternList.__iter__ = patternlist_iter

def patternlist_str(p):
  return '[' + ','.join([str(k) for k in p]) + ']'
PatternList.__str__ = patternlist_str
