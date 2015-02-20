import numpy

class CSVDOM:
  def __init__(self, path):
    self.ok = False
    self.rows = []
    with open(path, 'r') as f:
      for line in f.readlines():
        if line.startswith(';;'): continue
        self.rows += [line.split(',')]
    self.ok = True
    
