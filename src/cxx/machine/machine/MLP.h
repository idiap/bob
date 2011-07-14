/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 06 Jul 2011 16:49:13 CEST
 *
 * @brief The representation of a Multi-Layer Perceptron (MLP).
 */

#ifndef TORCH_MACHINE_MLP_H 
#define TORCH_MACHINE_MLP_H

#include <boost/random.hpp>
#include <blitz/array.h>

#include "io/HDF5File.h"
#include "machine/Activation.h"

namespace Torch { namespace machine {

  /**
   * An MLP object is a representation of a Multi-Layer Perceptron. This
   * implementation is feed-forward and fully-connected. The implementation
   * allows setting of input normalization values and a global activation
   * function. References to fully-connected feed-forward networks: Bishop's
   * Pattern Recognition and Machine Learning, Chapter 5. Figure 5.1 shows what
   * we mean.
   *
   * MLPs normally are multi-layered systems, with 1 or more hidden layers. As
   * a special case, this implementation also supports connecting the input
   * directly to the output by means of a single weight matrix. This is
   * equivalent of a LinearMachine, with the advantage it can be trained by MLP
   * trainers.
   */
  class MLP {
    
    public: //api

      typedef double (*actfun_t)(double); ///< activation function type

      /**
       * Constructor, builds a new MLP. Internal values are uninitialized. In
       * this case, there are no hidden layers and the resulting machine is
       * equivalent to a linear machine except, perhaps for the activation
       * function which is set to be a hyperbolic tangent.
       *
       * @param input Size of input vector
       * @param output Size of output vector
       */
      MLP (size_t input, size_t output);

      /**
       * Constructor, builds a new MLP. Internal values are uninitialized. In
       * this case, the number of hidden layers equals 1 and its size can be
       * defined by the middle parameter. The default activation function will
       * be set to hyperbolic tangent.
       *
       * @param input Size of input vector
       * @param hidden Size of the hidden layer
       * @param output Size of output vector
       */
      MLP (size_t input, size_t hidden, size_t output);

      /**
       * Constructor, builds a new MLP. Internal values are uninitialized. With
       * this constructor you can control the number of hidden layers your MLP
       * will have. The default activation function will be set to hyperbolic
       * tangent.
       *
       * @param input Size of input vector 
       * @param hidden The number and size of each hidden layer 
       * @param output Size of output vector
       */
      MLP (size_t input, const std::vector<size_t>& hidden, size_t output);
      
      /**
       * Builds a new MLP with a shape containing the number of inputs (first
       * element), number of outputs (last element) and the number of neurons
       * in each hidden layer (elements between the first and last element of
       * the vector). The default activation function will be set to hyperbolic
       * tangent.
       */
      MLP (const std::vector<size_t>& shape);
      
      /**
       * Copies another machine
       */
      MLP (const MLP& other);

      /**
       * Starts a new MLP from an existing Configuration object.
       */
      MLP (Torch::io::HDF5File& config);

      /**
       * Just to virtualise the destructor
       */
      virtual ~MLP();

      /**
       * Assigns from a different machine
       */
      MLP& operator= (const MLP& other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load (Torch::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save (Torch::io::HDF5File& config) const;

      /**
       * Forwards data through the network, outputs the values of each output
       * neuron.
       *
       * The input and output are NOT checked for compatibility each time. It
       * is your responsibility to do it.
       */
      void forward_ (const blitz::Array<double,1>& input,
          blitz::Array<double,1>& output) const;

      /**
       * Forwards data through the network, outputs the values of each output
       * neuron. 
       *
       * The input and output are checked for compatibility each time the
       * forward method is applied.
       */
      void forward (const blitz::Array<double,1>& input,
          blitz::Array<double,1>& output) const;

      /**
       * Resizes the machine. This causes this MLP to be completely
       * re-initialized and should be considered invalid for calculation after
       * this operation. Using this method there will be no hidden layers in
       * the resized machine.
       */
      void resize (size_t input, size_t output);

      /**
       * Resizes the machine. This causes this MLP to be completely
       * re-initialized and should be considered invalid for calculation after
       * this operation. Using this method there will be precisely 1 hidden
       * layer in the resized machine.
       */
      void resize (size_t input, size_t hidden, size_t output);

      /**
       * Resizes the machine. This causes this MLP to be completely
       * re-initialized and should be considered invalid for calculation after
       * this operation. Using this method there will be as many hidden layers
       * as there are size_t's in the vector parameter "hidden".
       */
      void resize (size_t input, const std::vector<size_t>& hidden,
          size_t output);

      /**
       * Resizes the machine. This causes this MLP to be completely
       * re-initialized and should be considered invalid for calculation after
       * this operation. Using this method there will be as many hidden layers
       * as there are size_t's in the vector parameter "hidden".
       */
      void resize (const std::vector<size_t>& shape);

      /**
       * Returns the number of inputs expected by this machine
       */
      inline size_t inputSize () const { return m_weight.front().extent(0); }

      /**
       * Returns the number of hidden layers this MLP has
       */
      inline size_t numOfHiddenLayers() const { return m_weight.size() - 1; }

      /**
       * Returns the number of outputs generated by this machine
       */
      inline size_t outputSize () const { return m_weight.back().extent(1); }

      /**
       * Returns the input subtraction factor
       */
      inline const blitz::Array<double, 1>& getInputSubraction() const
      { return m_input_sub; }

      /**
       * Sets the current input subtraction factor. We will check that the
       * number of inputs (first dimension of weights) matches the number of
       * values currently set and will raise an exception if that is not the
       * case.
       */
      void setInputSubtraction(const blitz::Array<double,1>& v);

      /**
       * Sets all input subtraction values to a specific value.
       */
      inline void setInputSubtraction(double v) { m_input_sub = v; }

      /**
       * Returns the input division factor
       */
      inline const blitz::Array<double, 1>& getInputDivision() const
      { return m_input_div; }

      /**
       * Sets the current input division factor. We will check that the number
       * of inputs (first dimension of weights) matches the number of values
       * currently set and will raise an exception if that is not the case.
       */
      void setInputDivision(const blitz::Array<double,1>& v);

      /**
       * Sets all input division values to a specific value.
       */
      inline void setInputDivision(double v) { m_input_div = v; }

      /**
       * Returns the weights of all layers.
       */
      inline const std::vector<blitz::Array<double, 2> >& getWeights() const 
      { return m_weight; }

      /**
       * Sets weights for all layers. The number of inputs, outputs and total
       * number of weights should be the same as set before, or this method
       * will raise.  If you would like to set this MLP to a different weight
       * configuration, consider first using resize().
       */
      void setWeights(const std::vector<blitz::Array<double,2> >& weight);

      /**
       * Sets all weights to a single specific value.
       */
      void setWeights(double v);

      /**
       * Returns the biases of this classifier, for every hidden layer and
       * output layer we have.
       */
      inline const std::vector<blitz::Array<double, 1> >& getBiases() const 
      { return m_bias; }

      /**
       * Sets the current biases. We will check that the number of biases
       * matches the number of weights (first dimension) currently set and
       * will raise an exception if that is not the case.
       */
      void setBiases(const std::vector<blitz::Array<double,1> >& bias);

      /**
       * Sets all output bias values to a specific value.
       */
      void setBiases(double v);

      /**
       * Returns the currently set activation function
       */
      inline Activation getActivation() const { return m_activation; }

      /**
       * Sets the activation function for each of the outputs.
       */
      void setActivation(Activation a);

      /**
       * A pointer to the actual activation function
       */
      inline actfun_t getActivationFunction() const { return m_actfun; }

      /**
       * Reset all weights and biases. You can (optionally) specify the
       * lower and upper bound for the uniform distribution that will be used
       * to draw values from. The default values are the ones recommended by
       * most implementations. Be sure of what you are doing before training to
       * change this too radically.
       *
       * Values are drawn using boost::uniform_real class. Values are taken
       * from the range [lower_bound, upper_bound) according to the
       * boost::random documentation.
       */
      void randomize(boost::mt19937& rng, double lower_bound=-0.1, 
          double upper_bound=+0.1);

      /**
       * This is equivalent to randomize() above, but we will create the boost
       * random number generator ourselves using a time-based seed. Results
       * after each call will be probably different as long as they are
       * separated by at least 1 microsecond (from the machine clock).
       */
      void randomize(double lower_bound=-0.1, double upper_bound=+0.1);

    private: //representation

      blitz::Array<double, 1> m_input_sub; ///< input subtraction
      blitz::Array<double, 1> m_input_div; ///< input division
      std::vector<blitz::Array<double, 2> > m_weight; ///< weights
      std::vector<blitz::Array<double, 1> > m_bias; ///< biases for the output
      Activation m_activation; ///< currently set activation type
      actfun_t m_actfun; ///< currently set activation function

      mutable std::vector<blitz::Array<double, 1> > m_buffer; ///< a buffer for speed
  
  };

}}

#endif /* TORCH_MACHINE_MLP_H */
