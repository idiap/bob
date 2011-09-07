/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed 31 Aug 2011 17:42:07 CEST
 *
 * @brief Implements various spatio-temporal gradients
 */

#ifndef TORCH_IP_SPATIOTEMPORALGRADIENT_H 
#define TORCH_IP_SPATIOTEMPORALGRADIENT_H

#include <blitz/array.h>

namespace Torch { namespace ip {

  /**
   * This class computes the spatio-temporal gradient using the same
   * approximation as the one described by Horn & Schunck in the paper titled
   * "Determining Optical Flow", published in 1981, Artificial Intelligence,
   * Vol. 17, No. 1-3, pp. 185-203.
   *
   * This is equivalent to convolving the image sequence with the following
   * (separate) kernels:
   *
   * u = |N| * ([-1 +1]^T([+1 +1]*(i1)) + [-1 +1]^T([+1 +1]*(i2)))
   * v = |N| * ([+1 +1]^T([-1 +1]*(i1)) + [+1 +1]^T([-1 +1]*(i2)))
   *
   * This will make-up the following convoluted kernel:
   *
   * u = |N| * [ -1 -1 ]   [ -1 -1 ]
   *           [ +1 +1 ] ; [ +1 +1 ] 
   *
   * v = |N| * [ -1 +1 ]   [ -1 +1 ]
   *           [ -1 +1 ] ; [ -1 +1 ] 
   *
   * |N| = 1/(2*sqrt(2))
   *
   * This method returns the matrices u and v that indicate the movement
   * intensity along the 'x' and 'y' directions respectively.
   */
  class ForwardGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat. This shape is used by the internal buffers.
       */
      ForwardGradient(const blitz::TinyVector<int,2>& shape);

      /**
       * Copy constructor
       */
      ForwardGradient(const ForwardGradient& other);

      /**
       * Assignment operator
       */
      ForwardGradient& operator= (const ForwardGradient& other);

      /**
       * Returns the current shape supported
       */
      inline const blitz::TinyVector<int,2>& 
        getShape(const blitz::TinyVector<int,2>& shape) const {
          return m_buffer1.shape();
        }

      /**
       * Re-shape internal buffers
       */
      void setShape(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual D'tor
       */
      virtual ~ForwardGradient();

      /**
       * Call this to run the gradient operator.
       */
      void operator()(const blitz::Array<double,2>& i1,
        const blitz::Array<double,2>& i2, blitz::Array<double,2>& u, 
        blitz::Array<double,2>& v) const;

    private: //representation

      mutable blitz::Array<double,2> m_buffer1;
      mutable blitz::Array<double,2> m_buffer2;
      blitz::Array<double,1> m_kernel1;
      blitz::Array<double,1> m_kernel2;

  };

  /**
   * This class computes the spatio-temporal gradient using a 3-D sobel
   * filter. The gradients are only calculated along the 'x' and 'y'
   * directions. The Sobel operator can be decomposed into 3 1D kernels that
   * are applied in sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 2 1]
   * one can represent the operations like this:
   *
   *                     [+1]             [1]
   * u = h'(x)h(y)h(t) = [ 0] [1 2 1]  [2]
   *                     [-1]        [1]
   *
   *                     [1]              [1]
   * v = h(x)h'(y)h(t) = [2] [-1 0 +1]  [2]
   *                     [1]          [1]
   *
   * The Sobel operator is an edge detector. It calculates the gradient
   * direction in the center of the 3D structure shown above.
   *
   * A normalization factor of 6*sqrt(2) is applied so the transformation
   * remains unitary.
   */
  class CentralGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat. This shape is used by the internal buffers.
       */
      CentralGradient(const blitz::TinyVector<int,2>& shape);

      /**
       * Copy constructor
       */
      CentralGradient(const CentralGradient& other);

      /**
       * Assignment operator
       */
      CentralGradient& operator= (const CentralGradient& other);

      /**
       * Returns the current shape supported
       */
      inline const blitz::TinyVector<int,2>& 
        getShape(const blitz::TinyVector<int,2>& shape) const {
          return m_buffer1.shape();
        }

      /**
       * Re-shape internal buffers
       */
      void setShape(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual D'tor
       */
      virtual ~CentralGradient();

      /**
       * Call this to run the gradient operator.
       */
      void operator() (const blitz::Array<double,2>& i_prev,
          const blitz::Array<double,2>& i,
          const blitz::Array<double,2>& i_after,
          blitz::Array<double,2>& u, blitz::Array<double,2>& v) const;

    private: //representation

      mutable blitz::Array<double,2> m_buffer1;
      mutable blitz::Array<double,2> m_buffer2;
      blitz::Array<double,1> m_kernel1;
      blitz::Array<double,1> m_kernel2;

  };

}}

#endif /* TORCH_IP_SPATIOTEMPORALGRADIENT_H */
