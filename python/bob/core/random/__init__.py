from .. import __from_extension_import__
__from_extension_import__('._core_random', __package__, locals())

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

__all__ = [k for k in dir() if not k.startswith('_')]
