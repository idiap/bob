/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 13 Jul 2011 12:29:28 CEST
 *
 * @brief A class that implements data shuffling for multi-class supervised and
 * unsupervised training.
 */

#ifndef TORCH_TRAINER_DATASHUFFLER_H
#define TORCH_TRAINER_DATASHUFFLER_H

#include <vector>
#include <blitz/array.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include "io/Arrayset.h"

namespace Torch { namespace trainer {

  /**
   * A data shuffler is capable of being populated with data from one or
   * multiple classes and matching target values. Once setup, the shuffer can
   * randomly select a number of vectors and accompaning targets for the
   * different classes, filling up user containers.
   *
   * Data shufflers are particular useful for training neural networks.
   */
  class DataShuffler {

    public: //api

      /**
       * Initializes the shuffler with some data classes and corresponding
       * targets. Note that Arraysets must have (for the time being), a shape
       * of (1,) and an element type == double.
       */
      DataShuffler(const std::vector<Torch::io::Arrayset>& data,
          const std::vector<blitz::Array<double,1> >& target);

      /**
       * Copy constructor
       */
      DataShuffler(const DataShuffler& other);

      /**
       * D'tor virtualization
       */
      virtual ~DataShuffler();

      /**
       * Assignment. This will also copy seeds set on the other shuffler.
       */
      DataShuffler& operator= (const DataShuffler& other);

      /**
       * Calculates and returns mean and standard deviation from the input
       * data.
       */
      void getStdNorm(blitz::Array<double,1>& mean,
          blitz::Array<double,1>& stddev) const;

      /**
       * Set automatic standard normalization.
       */
      void setAutoStdNorm(bool s);

      /**
       * Gets current automatic standard normalization settings
       */
      inline bool getAutoStdNorm() const { return m_do_stdnorm; }

      /**
       * The data shape
       */
      inline size_t getDataWidth() const { return m_data[0].getShape()[0]; }

      /**
       * The target shape
       */
      inline size_t getTargetWidth() const { return m_target[0].extent(0); }

      /**
       * Populates the output matrices by randomly selecting N arrays from the
       * input arraysets and matching targets in the most possible fair way.
       * The 'data' and 'target' matrices will contain N rows and the number of
       * columns that are dependent on input arraysets and target arrays.
       *
       * We check don't 'data' and 'target' for size compatibility and is your
       * responsibility to do so.
       *
       * Note this operation is non-const - we do alter the state of our ranges
       * internally.
       */
      void operator() (boost::mt19937& rng, blitz::Array<double,2>& data,
          blitz::Array<double,2>& target);

      /**
       * Populates the output matrices by randomly selecting N arrays from the
       * input arraysets and matching targets in the most possible fair way.
       * The 'data' and 'target' matrices will contain N rows and the number of
       * columns that are dependent on input arraysets and target arrays.
       *
       * We check don't 'data' and 'target' for size compatibility and is your
       * responsibility to do so.
       *
       * This version is a shortcut to the previous declaration of operator()
       * that actually instantiates its own random number generator and seed it
       * a time-based variable. We guarantee two calls will lead to different
       * results if they are at least 1 microsecond appart (procedure uses the
       * machine clock).
       */
      void operator() (blitz::Array<double,2>& data,
          blitz::Array<double,2>& target);

    private: //representation

      std::vector<Torch::io::Arrayset> m_data;
      std::vector<blitz::Array<double,1> > m_target;
      std::vector<boost::uniform_int<size_t> > m_range;
      bool m_do_stdnorm; ///< should we apply standard normalization
      blitz::Array<double,1> m_mean; ///< mean to be used for std. norm.
      blitz::Array<double,1> m_stddev; ///< std.dev for std. norm.

  };

}}

#endif /* TORCH_TRAINER_DATASHUFFLER_H */
