/**
 * @file cxx/ip/ip/HornAndSchunckFlow.h
 * @date Wed Mar 16 15:01:13 2011 +0100
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Estimates motion between two sequences of images.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#ifndef BOB_IP_HORNANDSCHUNCKFLOW_H 
#define BOB_IP_HORNANDSCHUNCKFLOW_H

#include <cstdlib>
#include <stdint.h>
#include <blitz/array.h>
#include "ip/SpatioTemporalGradient.h"

namespace bob { namespace ip { namespace optflow {

  /**
   * An approximation to the Laplacian (averaging) operator. Using the
   * following (non-separable) kernel for the Laplacian:
   *
   * [ 0 -1  0]
   * [-1  4 -1]
   * [ 0 -1  0]
   *
   * This is used as the Laplacian operator on OpenCV. To calculate the u_bar
   * value we must remove the central mean and multiply by -1/4, yielding:
   *
   * [ 0  1/4  0  ]
   * [1/4  0  1/4 ]
   * [ 0  1/4  0  ]
   *
   * Note that you will get the WRONG results if you use the Laplacian kernel
   * directly...
   */
  void laplacian_avg_hs_opencv(const blitz::Array<double,2>& input,
      blitz::Array<double,2>& output);

  /**
   * An approximation to the Laplacian operator. Using the following
   * (non-separable) kernel:
   *
   * [-1 -2 -1]
   * [-2 12 -2]
   * [-1 -2 -1]
   *
   * This is used on the Horn & Schunck paper. To calculate the u_bar value we
   * must remove the central mean and multiply by -1/12, yielding:
   *
   * [1/12 1/6 1/12]
   * [1/6   0  1/6 ]
   * [1/12 1/6 1/12]
   */
  void laplacian_avg_hs(const blitz::Array<double,2>& input,
      blitz::Array<double,2>& output);

  /**
   * This can calculate the Optical Flow between two sequences of images (i1,
   * the starting image and i2, the final image). It does this using the
   * iterative method described by Horn & Schunck in the paper titled
   * "Determining Optical Flow", published in 1981, Artificial Intelligence,
   * Vol. 17, No. 1-3, pp. 185-203.
   *
   * The method constrains the calculation with two assertions that can be made
   * on a natural sequence of images:
   *
   * 1. For the same lighting conditions, the brightness (E) of the shapes in
   * an image do not change and, therefore, the derivative of E w.r.t. time
   * (dE/dt) equals zero.  2. The relative velocities of adjancent points in an
   * image varies smoothly. The smothness constraint is applied on the image
   * data using the Laplacian operator.
   *
   * It then approximates the calculation of conditions 1 and 2 above using a
   * Taylor series expansion and ignoring terms with order greater or equal 2.
   * This technique is also know as "Finite Differences" and is also applied in
   * other engineering fields such as Fluid Mechanics.
   *
   * The problem is finally posed as an iterative process that simultaneously
   * minimizes conditions 1 and 2 above. A weighting factor (alpha - also
   * sometimes referred as "lambda" in some implementations) controls the
   * relative importance of the two above conditions. The higher it gets, the
   * smoother the field will be. 
   *
   * N.B.: OpenCV sets lambda = alpha^2
   *
   * This is the set of equations that are implemented:
   *
   * u(n+1) = U(n) - Ex[Ex * U(n) + Ey * V(n) + Et]/(alpha^2 + Ex^2 + Ey^2)
   * v(n+1) = V(n) - Ey[Ey * U(n) + Ey * V(n) + Et]/(alpha^2 + Ex^2 + Ey^2)
   *
   * Where:
   *
   * u(.) - relative velocity in the x direction v(.) - relative velocity in
   * the y direction Ex, Ey and Et - partial derivative of brightness in the x,
   * y and t, which are estimated using finite differences based on the images
   * i1 and i2 U(.) - laplacian estimates for x given equations in section 8 of
   * the paper V(.) - laplacian estimates for y given equations in section 8 of
   * the paper
   *
   * According to paper, alpha^2 should be more or less set to noise in
   * estimating Ex^2 + Ey^2. In practice, many algorithms consider values
   * around 200 a good default. The higher this number is, the more importance
   * on smoothing you will be putting.
   *
   * The initial conditions are set such that u(0) = v(0) = 0, except in the
   * case where you provide them. If you analyzing a video stream, it is a good
   * idea to use the previous estimate as the initial conditions.
   *
   * This is a dense flow estimator and is computed for all pixels in the
   * image. More details are given at the source code for this class.
   * Calling it estimates u0 and v0 based on their initial state. If you want
   * to start from scratch, just set u0 and v0 to 0.
   */
  class VanillaHornAndSchunckFlow {

    public: //api
   
      /**
       * Constructor, specify shape of images to be treated
       */
      VanillaHornAndSchunckFlow(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual destructor
       */
      virtual ~VanillaHornAndSchunckFlow();

      /**
       * Returns the current shape supported
       */
      inline const blitz::TinyVector<int,2>&
        getShape(const blitz::TinyVector<int,2>& shape) const {
          return m_ex.shape();
        }

      /**
       * Re-shape internal buffers
       */
      void setShape(const blitz::TinyVector<int,2>& shape);

      /**
       * Calculates the square of the smoothness error (Ec^2) by using the
       * formula described in the paper:
       *
       * Ec^2 = (u_bar - u)^2 + (v_bar - v)^2
       *
       * Sets the input matrix with the discrete values.
       */
      void evalEc2 (const blitz::Array<double,2>& u,
          const blitz::Array<double,2>& v, blitz::Array<double,2>& error) const;

      /**
       * Calculates the brightness error (Eb) as defined in the paper:
       *
       * Eb = (Ex*u + Ey*v + Et)
       *
       * Sets the input matrix with the discrete values
       */
      void evalEb (const blitz::Array<double,2>& i1,
          const blitz::Array<double,2>& i2, const blitz::Array<double,2>& u,
          const blitz::Array<double,2>& v, blitz::Array<double,2>& error) const;

      /**
       * Call this to evaluate the flow
       */
      void operator() (double alpha, size_t iterations, const
          blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
          blitz::Array<double,2>& u0, blitz::Array<double,2>& v0) const;

    private: //representation

      bob::ip::HornAndSchunckGradient m_gradient; ///< Gradient operator
      mutable blitz::Array<double,2> m_ex; ///< Ex buffer
      mutable blitz::Array<double,2> m_ey; ///< Ey buffer
      mutable blitz::Array<double,2> m_et; ///< Et buffer
      mutable blitz::Array<double,2> m_u; ///< U (x velocity) buffer
      mutable blitz::Array<double,2> m_v; ///< V (y velocity) buffer
      mutable blitz::Array<double, 2> m_cterm; ///< common term buffer

  };

  /**
   * This is a clone of the Vanilla HornAndSchunck method that uses a Sobel
   * gradient estimator instead of the forward estimator used by the
   * classical method. The Laplacian operator is also replaced with a more
   * common method.
   */
  class HornAndSchunckFlow {

    public: //api
   
      /**
       * Constructor, specify shape of images to be treated
       */
      HornAndSchunckFlow(const blitz::TinyVector<int,2>& shape);

      /**
       * Virtual destructor
       */
      virtual ~HornAndSchunckFlow();

      /**
       * Returns the current shape supported
       */
      inline const blitz::TinyVector<int,2>&
        getShape(const blitz::TinyVector<int,2>& shape) const {
          return m_ex.shape();
        }

      /**
       * Re-shape internal buffers
       */
      void setShape(const blitz::TinyVector<int,2>& shape);

      /**
       * Calculates the square of the smoothness error (Ec^2) by using the
       * formula described in the paper:
       *
       * Ec^2 = (u_bar - u)^2 + (v_bar - v)^2
       *
       * Sets the input matrix with the discrete values.
       */
      void evalEc2 (const blitz::Array<double,2>& u,
          const blitz::Array<double,2>& v, blitz::Array<double,2>& error) const;

      /**
       * Calculates the brightness error (Eb) as defined in the paper:
       *
       * Eb = (Ex*u + Ey*v + Et)
       *
       * Sets the input matrix with the discrete values
       */
      void evalEb (const blitz::Array<double,2>& i1,
          const blitz::Array<double,2>& i2, const blitz::Array<double,2>& i3, 
          const blitz::Array<double,2>& u, const blitz::Array<double,2>& v,
          blitz::Array<double,2>& error) const;

      /**
       * Call this to evaluate the flow
       */
      void operator() (double alpha, size_t iterations, const
          blitz::Array<double,2>& i1, const blitz::Array<double,2>& i2,
          const blitz::Array<double,2>& i3,
          blitz::Array<double,2>& u0, blitz::Array<double,2>& v0) const;

    private: //representation

      bob::ip::SobelGradient m_gradient; ///< Gradient operator
      mutable blitz::Array<double,2> m_ex; ///< Ex buffer
      mutable blitz::Array<double,2> m_ey; ///< Ey buffer
      mutable blitz::Array<double,2> m_et; ///< Et buffer
      mutable blitz::Array<double,2> m_u; ///< U (x velocity) buffer
      mutable blitz::Array<double,2> m_v; ///< V (y velocity) buffer
      mutable blitz::Array<double, 2> m_cterm; ///< common term buffer

  };

  /**
   * Computes the generalized flow error.
   *
   * E = i2(x-u,y-v) - i1(x,y))
   */
  void flowError (const blitz::Array<double,2>& i1,
      const blitz::Array<double,2>& i2, const blitz::Array<double,2>& u,
      const blitz::Array<double,2>& v, blitz::Array<double,2>& error);

}}}

#endif /* BOB_IP_HORNANDSCHUNCKFLOW_H */
