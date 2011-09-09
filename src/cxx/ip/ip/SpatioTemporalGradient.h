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
   * This class computes the spatio-temporal gradient using a 2-term
   * approximation composed of 2 separable kernels (one for the diference term
   * and another one for the averaging term).
   */
  class ForwardGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat and with the kernels to be applied. The shape is used by the
       * internal buffers.
       *
       * @param diff_kernel The kernel that contains the difference operation.
       * Typically, this is [1; -1]. Note the kernel is mirrored during the
       * convolution operation. To obtain a [-1; +1] sliding operator, specify
       * [+1; -1]. This kernel must have a size = 2.
       * 
       * @param avg_kernel  The kernel that contains the spatial averaging
       * operation. This kernel is typically [+1; +1]. This kernel must have
       * a size = 2.
       *
       * @param shape This is the shape of the images to be treated. This has
       * to match the input image height x width specifications (in that
       * order).
       */
      ForwardGradient(const blitz::Array<double,1>& diff_kernel,
          const blitz::Array<double,1>& avg_kernel,
          const blitz::TinyVector<int,2>& shape);

      /**
       * Copy constructor
       */
      ForwardGradient(const ForwardGradient& other);

      /**
       * Virtual D'tor
       */
      virtual ~ForwardGradient();

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
       * Gets the difference kernel
       */
      inline const blitz::Array<double,1>& getDiffKernel() const { 
        return m_diff_kernel;
      }

      /**
       * Sets the difference kernel
       */
      void setDiffKernel(const blitz::Array<double,1>& k);

      /**
       * Gets the averaging kernel
       */
      inline const blitz::Array<double,1>& getAvgKernel() const {
        return m_avg_kernel;
      }

      /**
       * Sets the averaging kernel
       */
      void setAvgKernel(const blitz::Array<double,1>& k);

      /**
       * Call this to run the gradient operator and return Ex, Ey and Et - the
       * spatio temporal gradients for the image pair i1, i2
       */
      void operator()(const blitz::Array<double,2>& i1,
        const blitz::Array<double,2>& i2, blitz::Array<double,2>& Ex,
        blitz::Array<double,2>& Ey, blitz::Array<double,2>& Et) const;

    private: //representation

      blitz::Array<double,1> m_diff_kernel;
      blitz::Array<double,1> m_avg_kernel;
      mutable blitz::Array<double,2> m_buffer1;
      mutable blitz::Array<double,2> m_buffer2;

  };

  /**
   * This class computes the spatio-temporal gradient using the same
   * approximation as the one described by Horn & Schunck in the paper titled
   * "Determining Optical Flow", published in 1981, Artificial Intelligence,
   * Vol. 17, No. 1-3, pp. 185-203.
   *
   * This is equivalent to convolving the image sequence with the following
   * (separate) kernels:
   *
   * Ex = 1/4 * ([-1 +1]^T * ([+1 +1]*(i1)) + [-1 +1]^T * ([+1 +1]*(i2)))
   * Ey = 1/4 * ([+1 +1]^T * ([-1 +1]*(i1)) + [+1 +1]^T * ([-1 +1]*(i2)))
   * Et = 1/4 * ([+1 +1]^T * ([+1 +1]*(i1)) - [+1 +1]^T * ([+1 +1]*(i2)))
   */
  class HornAndSchunckGradient : public virtual ForwardGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat. The shape is used by the internal buffers.
       *
       * The difference kernel for this operator is [+1/4; -1/4]
       * The averaging kernel for this oeprator is [+1; +1]
       */
      HornAndSchunckGradient(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual D'tor
       */
      virtual ~HornAndSchunckGradient();

  };

  /**
   * This class computes the spatio-temporal gradient using a 3-term
   * approximation composed of 2 separable kernels (one for the diference term
   * and another one for the averaging term).
   */
  class CentralGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat and with the kernels to be applied. The shape is used by the
       * internal buffers.
       *
       * @param diff_kernel The kernel that contains the difference operation.
       * Typically, this is [1; -1]. Note the kernel is mirrored during the
       * convolution operation. To obtain a [-1; +1] sliding operator, specify
       * [+1; -1]. This kernel must have a size = 3.
       * 
       * @param avg_kernel  The kernel that contains the spatial averaging
       * operation. This kernel is typically [+1; +1]. This kernel must have
       * a size = 3.
       *
       * @param shape This is the shape of the images to be treated. This has
       * to match the input image height x width specifications (in that
       * order).
       */
      CentralGradient(const blitz::Array<double,1>& diff_kernel,
          const blitz::Array<double,1>& avg_kernel,
          const blitz::TinyVector<int,2>& shape);

      /**
       * Copy constructor
       */
      CentralGradient(const CentralGradient& other);

      /**
       * Virtual D'tor
       */
      virtual ~CentralGradient();

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
       * Gets the difference kernel
       */
      inline const blitz::Array<double,1>& getDiffKernel() const { 
        return m_diff_kernel;
      }

      /**
       * Sets the difference kernel
       */
      void setDiffKernel(const blitz::Array<double,1>& k);

      /**
       * Gets the averaging kernel
       */
      inline const blitz::Array<double,1>& getAvgKernel() const {
        return m_avg_kernel;
      }

      /**
       * Sets the averaging kernel
       */
      void setAvgKernel(const blitz::Array<double,1>& k);

      /**
       * Call this to run the gradient operator.
       */
      void operator() (const blitz::Array<double,2>& i1,
          const blitz::Array<double,2>& i2, const blitz::Array<double,2>& i3,
          blitz::Array<double,2>& Ex, blitz::Array<double,2>& Ey,
          blitz::Array<double,2>& Et) const;

    private: //representation

      blitz::Array<double,1> m_diff_kernel;
      blitz::Array<double,1> m_avg_kernel;
      mutable blitz::Array<double,2> m_buffer1;
      mutable blitz::Array<double,2> m_buffer2;
      mutable blitz::Array<double,2> m_buffer3;

  };

  /**
   * This class computes the spatio-temporal gradient using a 3-D sobel
   * filter. The gradients are calculated along the x, y and t directions. The
   * Sobel operator can be decomposed into 2 1D kernels that are applied in
   * sequence. Considering h'(.) = [+1 0 -1] and h(.) = [1 2 1] one can
   * represent the operations like this:
   *
   *                      [+1]             [1]
   * Ex = h'(x)h(y)h(t) = [ 0] [1 2 1]  [2]
   *                      [-1]        [1]
   *
   *                      [1]              [1]
   * Ey = h(x)h'(y)h(t) = [2] [-1 0 +1]  [2]
   *                      [1]          [1]
   *
   *                      [1]             [-1]
   * Et = h(x)h(y)h'(t) = [2] [1 2 1]   [0]
   *                      [1]        [+1]
   */
  class SobelGradient: public virtual CentralGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat. The shape is used by the internal buffers.
       *
       * The difference kernel for this operator is [+1; 0; -1]
       * The averaging kernel for this oeprator is [+1; +2; +1]
       */
      SobelGradient(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual destructor
       */
      virtual ~SobelGradient();

  };

  /**
   * This class computes the spatio-temporal gradient using a 3-D Prewitt
   * (Smoothed) filter. The gradients are calculated along the x, y and t
   * directions. It is equivalent to a Sobel gradient except the averaging term
   * is all 1's.
   */
  class PrewittGradient: public virtual CentralGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat. The shape is used by the internal buffers.
       *
       * The difference kernel for this operator is [+1; 0; -1]
       * The averaging kernel for this oeprator is [+1; +1; +1]
       */
      PrewittGradient(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual destructor
       */
      virtual ~PrewittGradient();

  };

  /**
   * This class computes the spatio-temporal gradient using a 3-D Isotropic
   * filter. The gradients are calculated along the x, y and t directions. It
   * is equivalent to a Sobel gradient except the averaging middle term is
   * sqrt(2).
   */
  class IsotropicGradient: public virtual CentralGradient {

    public: //api

      /**
       * Constructor. We initialize with the shape of the images we need to
       * treat. The shape is used by the internal buffers.
       *
       * The difference kernel for this operator is [+1; 0; -1]
       * The averaging kernel for this oeprator is [+1; sqrt(2); +1]
       */
      IsotropicGradient(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual destructor
       */
      virtual ~IsotropicGradient();

  };

  /**
   * An approximation to the Laplacian operator. Using the following
   * (non-separable) kernel:
   *
   * [ 0 -1  0]
   * [-1  4 -1]
   * [ 0 -1  0]
   *
   * This is used as the Laplacian operator on OpenCV (multiplied by -1)
   */
  void laplacian_014(const blitz::Array<double,2>& input,
      blitz::Array<double,2>& output);

  /**
   * An approximation to the Laplacian operator. Using the following
   * (non-separable) kernel:
   *
   * [-1 -1 -1]
   * [-1  8 -1]
   * [-1 -1 -1]
   */
  void laplacian_18(const blitz::Array<double,2>& input,
      blitz::Array<double,2>& output);

  /**
   * An approximation to the Laplacian operator. Using the following
   * (non-separable) kernel:
   *
   * [-1 -2 -1]
   * [-2 12 -2]
   * [-1 -2 -1]
   *
   * This is used on the Horn & Schunck paper (multiplied by -1/12)
   */
  void laplacian_12(const blitz::Array<double,2>& input,
      blitz::Array<double,2>& output);
}}

#endif /* TORCH_IP_SPATIOTEMPORALGRADIENT_H */
