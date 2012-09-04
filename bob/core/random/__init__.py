from ._core_random import *

class variate_generator:
  """A pure-python version of the boost::variate_generator<> class"""

  def __init__(self, engine, distribution):
    self.engine = engine
    self.distribution = distribution

  def seed(self, value):
    self.engine.seed(value)
    self.distribution.reset()

  def __call__(self):
    return self.distribution(self.engine)

__all__ = dir()
