/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 30 May 2011 12:27:03 CEST
 *
 * A machine that implements the liner projection of input to the output using
 * weights, biases and sums:
 *
 * output = sum (weights * inputs) + bias
 *
 * A linear classifier. See C. M. Bishop, "Pattern Recognition and Machine
 * Learning", chapter 4
 */

#ifndef TORCH_MACHINE_LINEARMACHINE_H
#define TORCH_MACHINE_LINEARMACHINE_H

#include <blitz/array.h>
#include "config/Configuration.h"

namespace Torch { namespace machine {

  /**
   * A linear classifier. See C. M. Bishop, "Pattern Recognition and Machine
   * Learning", chapter 4 for more details.
   */
  class LinearMachine {

    public: //api

      /**
       * Constructor, builds a new Linear machine. Weights and biases are
       * not initialized.
       *
       * @param input Size of input vector
       * @param output Size of output vector
       */
      LinearMachine (size_t input, size_t output);

      /**
       * Builds a new machine with a set of given weights and biases. We will
       * check that the number of inputs (second dimension of weights) matches
       * the number of biases and will raise an exception if that is not the
       * case.
       */
      LinearMachine(const blitz::Array<double,2>& weight, const
          blitz::Array<double,1>& bias);

      /**
       * Copies another machine
       */
      LinearMachine (const LinearMachine& other);

      /**
       * Starts a new LinearMachine from an existing Configuration object.
       */
      LinearMachine (const Torch::config::Configuration& config);

      /**
       * Just to virtualise the destructor
       */
      virtual ~LinearMachine();

      /**
       * Assigns from a different machine
       */
      LinearMachine& operator= (const LinearMachine& other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load (const Torch::config::Configuration& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save (Torch::config::Configuration& config) const;

      /**
       * Forwards data through the network, outputs the values of each linear
       * component the input signal is decomposed at.
       *
       * The input and output are checked for compatibility each time the
       * forward method is applied.
       */
      void forward (const blitz::Array<double,1>& input,
          blitz::Array<double,1>& output) const;

      /**
       * Returns the current weight representation. Each row should be
       * considered as a vector from which each of the output values is derived
       * by projecting the input onto such a vector.
       */
      inline const blitz::Array<double, 2> getWeights() const 
      { return m_weight; }

      /**
       * Returns the biases of this classifier.
       */
      inline const blitz::Array<double, 1> getBiases() const 
      { return m_bias; }

      /**
       * Sets the current weight and bias representation. We will check that
       * the number of inputs (second dimension of weights) matches the number
       * of biases and will raise an exception if that is not the case.
       */
      void setWeightsAndBiases(const blitz::Array<double,2>& weight, const
          blitz::Array<double,1>& bias);

    private: //representation

      blitz::Array<double, 2> m_weight; ///< weights
      blitz::Array<double, 1> m_bias; ///< biases
  
  };

}}

#endif /* TORCH_MACHINE_LINEARMACHINE_H */
