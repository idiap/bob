"""Random number generation and distributions.
"""


from .. import __from_extension_import__
__from_extension_import__('._core_random', __package__, locals())

class variate_generator:
  """A pure-python version of the boost::variate_generator<> class
  
  Keyword parameters:

  engine
    An instance of the RNG you would like to use. This has to be an object of
    the class :py:class:`bob.core.random.mt19937`, already initialized.

  distribution
    The distribution to respect when generating scalars using the engine. The
    distribution object should be previously initialized.
  """

  def __init__(self, engine, distribution):

    self.engine = engine
    self.distribution = distribution

  def seed(self, value):
    """Resets the seed of the ``variate_generator`` with a (integer) value"""

    self.engine.seed(value)
    self.distribution.reset()

  def __call__(self):
    """Use the ``()`` operator to generate a random scalar"""

    return self.distribution(self.engine)

__all__ = [k for k in dir() if not k.startswith('_')]
