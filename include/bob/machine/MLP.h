/**
 * @file bob/machine/MLP.h
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief The representation of a Multi-Layer Perceptron (MLP).
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_MACHINE_MLP_H 
#define BOB_MACHINE_MLP_H

#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>
#include <blitz/array.h>

#include <bob/io/HDF5File.h>
#include <bob/machine/Activation.h>

namespace bob { namespace machine {
  /**
   * @ingroup MACHINE
   * @{
   */

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
      MLP (bob::io::HDF5File& config);

      /**
       * Just to virtualise the destructor
       */
      virtual ~MLP();

      /**
       * Assigns from a different machine
       */
      MLP& operator= (const MLP& other);

      /**
       * @brief Equal to
       */
      bool operator== (const MLP& other) const;

      /**
       * @brief Not equal to
       */
      bool operator!= (const MLP& other) const;

      /**
       * @brief Similar to
       */
      bool is_similar_to(const MLP& other, const double r_epsilon=1e-5,
        const double a_epsilon=1e-8) const;


      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load (bob::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save (bob::io::HDF5File& config) const;

      /**
       * Forwards data through the network, outputs the values of each output
       * neuron.
       *
       * The input and output are NOT checked for compatibility each time. It
       * is your responsibility to do it.
       */
      void forward_ (const blitz::Array<double,1>& input,
          blitz::Array<double,1>& output);

      /**
       * Forwards data through the network, outputs the values of each output
       * neuron. 
       *
       * The input and output are checked for compatibility each time the
       * forward method is applied.
       */
      void forward (const blitz::Array<double,1>& input,
          blitz::Array<double,1>& output);

      /**
       * Forwards data through the network, outputs the values of each output
       * neuron. This variant will take a number of inputs in one single input
       * matrix with inputs arranged row-wise (i.e., every row contains an
       * individual input).
       *
       * The input and output are NOT checked for compatibility each time. It
       * is your responsibility to do it.
       */
      void forward_ (const blitz::Array<double,2>& input,
          blitz::Array<double,2>& output);

      /**
       * Forwards data through the network, outputs the values of each output
       * neuron. This variant will take a number of inputs in one single input
       * matrix with inputs arranged row-wise (i.e., every row contains an
       * individual input).
       *
       * The input and output are checked for compatibility each time the
       * forward method is applied.
       */
      void forward (const blitz::Array<double,2>& input,
          blitz::Array<double,2>& output);

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
      size_t inputSize () const { return m_weight.front().extent(0); }

      /**
       * Returns the number of hidden layers this MLP has
       */
      size_t numOfHiddenLayers() const { return m_weight.size() - 1; }

      /**
       * Returns the number of outputs generated by this machine
       */
      size_t outputSize () const { return m_weight.back().extent(1); }

      /**
       * Returns the input subtraction factor
       */
      const blitz::Array<double, 1>& getInputSubtraction() const
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
      void setInputSubtraction(double v) { m_input_sub = v; }

      /**
       * Returns the input division factor
       */
      const blitz::Array<double, 1>& getInputDivision() const
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
      void setInputDivision(double v) { m_input_div = v; }

      /**
       * Returns the weights of all layers.
       */
      const std::vector<blitz::Array<double, 2> >& getWeights() const 
      { return m_weight; }

      /**
       * @brief Returns the weights of all layers in order to be updated.
       * This method should only be used by trainers.
       */
      std::vector<blitz::Array<double, 2> >& updateWeights()
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
      const std::vector<blitz::Array<double, 1> >& getBiases() const 
      { return m_bias; }

      /**
       * @brief Returns the biases of this classifier, for every hidden layer
       * and output layer we have, in order to be updated.
       * This method should only be used by trainers.
       */
      std::vector<blitz::Array<double, 1> >& updateBiases()
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
       * Returns the currently set activation function for the hidden layers
       */
      boost::shared_ptr<Activation> getHiddenActivation() const 
      { return m_hidden_activation; }

      /**
       * Sets the activation function for each of the hidden layers.
       */
      void setHiddenActivation(boost::shared_ptr<Activation> a) {
        m_hidden_activation = a;
      }

      /**
       * Returns the currently set output activation function
       */
      boost::shared_ptr<Activation> getOutputActivation() const 
      { return m_output_activation; }

      /**
       * Sets the activation function for the outputs of the last layer.
       */
      void setOutputActivation(boost::shared_ptr<Activation> a) {
        m_output_activation = a;
      }

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
      boost::shared_ptr<Activation> m_hidden_activation; ///< currently set activation type
      boost::shared_ptr<Activation> m_output_activation; ///< currently set activation type
      mutable std::vector<blitz::Array<double, 1> > m_buffer; ///< buffer for the outputs of each layer
  
  };

  /**
   * @}
   */
}}

#endif /* BOB_MACHINE_MLP_H */
