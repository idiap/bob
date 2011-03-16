/**
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 * @date Tue  8 Mar 07:41:33 2011 
 *
 * @brief Estimates motion between two sequences of images.
 */

#ifndef TORCH_IP_HORNANDSCHUNCKFLOW_H 
#define TORCH_IP_HORNANDSCHUNCKFLOW_H

#include <cstdlib>
#include <stdint.h>
#include <blitz/array.h>

namespace Torch { namespace ip {

  /**
   * Objects of this class, after configuration, can calculate the Optical Flow
   * between two sequences of images (i1, the starting image and i2, the final
   * image). It does this using the iterative method described by Horn & Schunck
   * in the paper titled "Determining Optical Flow", published in 1981,
   * Artificial Intelligence, Vol. 17, No. 1-3, pp. 185-203.
   *
   * The method constrains the calculation with two assertions that can be made
   * on a natural sequence of images:
   *
   * 1. For the same lighting conditions, the brightness (E) of the shapes in
   * an image do not change and, therefore, the derivative of E w.r.t. time
   * (dE/dt) equals zero. 
   * 2. The relative velocities of adjancent points in an image varies
   * smoothly. The smothness constraint is applied on the image data using the
   * Laplacian operator.
   *
   * It then approximates the calculation of conditions 1 and 2 above using a
   * Taylor series expansion and ignoring terms with order greater or equal 2.
   * This technique is also know as "Finite Differences" and is also applied in
   * other engineering fields such as Fluid Mechanics.
   *
   * The problem is finally posed as an iterative process that simultaneously
   * minimizes conditions 1 and 2 above. A weighting factor (alpha)
   * controls the relative importance of the two above conditions. 
   *
   * This is the set of equations that are implemented:
   *
   * u(n+1) = U(n) - Ex[Ex * U(n) + Ey * V(n) + Et]/(alpha^2 + Ex^2 + Ey^2)
   * v(n+1) = V(n) - Ey[Ey * U(n) + Ey * V(n) + Et]/(alpha^2 + Ex^2 + Ey^2)
   *
   * Where:
   *
   * u(.) - relative velocity in the x direction
   * v(.) - relative velocity in the y direction
   * Ex, Ey and Et - partial derivative of brightness in the x, y and t, which
   *   are estimated using finite differences based on the images i1 and i2
   * U(.) - laplacian estimates for x given equations in section 8 of the paper
   * V(.) - laplacian estimates for y given equations in section 8 of the paper
   *
   * According to paper, alpha^2 should be more or less set to noise in
   * estimating Ex^2 + Ey^2. In practice, many algorithms consider values
   * around 200 a good default. The higher this number is, the more importance
   * on smoothing you will be putting.
   *
   * The initial conditions are set such that u(0) =
   * v(0) = 0, except in the case where you provide them. If you analyzing a
   * video stream, it is a good idea to use the previous estimate as the
   * initial conditions.
   *
   * This is a dense flow estimator and is computed for all pixels in the
   * image. More details are given at the source code for this class.
   */
  class HornAndSchunckFlow {

    public:

      /**
       * Constructs a new HornAndSchunckFlow estimator using a certain weight
       * alpha and pre-programmed to perform a number of iterations. 
       */
      HornAndSchunckFlow(float alpha, size_t iterations);

      /**
       * Copy constructor
       */
      HornAndSchunckFlow(const HornAndSchunckFlow& other);

      /**
       * Destructor virtualization makes inheritance work adequately
       */
      virtual ~HornAndSchunckFlow();

      /**
       * Assignment operator
       */
      HornAndSchunckFlow& operator= (const HornAndSchunckFlow& other);

      /**
       * Getters
       */
      inline float getAlpha() const { return m_alpha; }
      inline size_t getIterations() const { return m_iterations; }

      /**
       * Setters
       */
      inline void setAlpha (float alpha) { m_alpha = alpha; }
      inline void setIterations (size_t iterations) 
      { m_iterations = iterations; }

      /**
       * Call an object of this type to compute the flow. u and v should be
       * initialized or set to zero (if we are to compute the flow from
       * scratch).
       */
      void operator() (const blitz::Array<uint8_t,2>& i1,
          const blitz::Array<uint8_t,2>& i2,
          blitz::Array<double,2>& u, blitz::Array<double,2>& v);

    private: //representation

      float m_alpha; ///< weighting factor
      size_t m_iterations; ///< number of iterations

      //cache variables to improve computation speed
      blitz::Array<double,2> m_ex;
      blitz::Array<double,2> m_ey;
      blitz::Array<double,2> m_et;
      blitz::Array<double,2> m_u0;
      blitz::Array<double,2> m_v0;
      blitz::Array<double,2> m_common_term;

  };

}}

#endif /* TORCH_IP_HORNANDSCHUNCKFLOW_H */
