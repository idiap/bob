/**
 * @file bob/math/interiorpointLP.h
 * @date Thu Mar 31 14:32:14 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines interior point methods which allow to solve a
 *        linear program (LP).
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

#ifndef BOB_MATH_INTERIOR_POINT_LP_H
#define BOB_MATH_INTERIOR_POINT_LP_H

#include <blitz/array.h>

namespace bob {
/**
 * \ingroup libmath_api
 * @{
 *
 */
  namespace math {
    /**
      * @brief Function which solves a linear program using a short-step
      *   interior point method. For more details about this algorithm,
      *   please refer to the following book:
      *   "Primal-Dual Interior-Point Methods", Stephen J. Wright,
      *   ISBN: 978-0898713824, chapter 5: "Path-Following Algorithms"
      *
      *   The primal linear program (LP) is defined as follows:
      *     min transpose(c)*x, s.t. A*x=b, x>=0
      *   The dual formulation is:
      *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
      *
      * @warning The X blitz::array is resized and reindexed.
      * @param A The A squared-matrix of the system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param c The c vector involved in the minimization
      * @param theta Threshold
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      * @param epsilon The expected precision for the algorithm to stop
      */
    void interiorpointShortstepLP(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double theta, blitz::Array<double,1>& x, const double epsilon);
    void interiorpointShortstepNoInitLP(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double theta, blitz::Array<double,1>& x, 
      blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
      const double epsilon);

    /**
      * @brief Function which solves a linear program using a predictor 
      *   corrector interior point method. For more details about this 
      *   algorithm, please refer to the following book:
      *   "Primal-Dual Interior-Point Methods", Stephen J. Wright
      *   ISBN: 978-0898713824, chapter 5: Path-Following Algorithms
      *
      *   The primal linear program (LP) is defined as follows:
      *     min transpose(c)*x, s.t. A*x=b, x>=0
      *   The dual formulation is:
      *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
      *
      * @warning The X blitz::array is resized and reindexed.
      * @param A The A squared-matrix of the system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param c The c vector involved in the minimization
      * @param theta_pred For the prediction
      * @param theta_corr For the correction
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      * @param epsilon The expected precision for the algorithm to stop
      */
    void interiorpointPredictorCorrectorLP(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double theta_pred, const double theta_corr, blitz::Array<double,1>& x, 
      const double epsilon);
    void interiorpointPredictorCorrectorNoInitLP(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double theta_pred, blitz::Array<double,1>& x, 
      blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
      const double epsilon);

    /**
      * @brief Function which solves a linear program using a long-step
      *   interior point method. For more details about this algorithm,
      *   please refer to the following book:
      *   "Primal-Dual Interior-Point Methods", Stephen J. Wright,
      *   ISBN: 978-0898713824, chapter 5: "Path-Following Algorithms"
      *
      *   The primal linear program (LP) is defined as follows:
      *     min transpose(c)*x, s.t. A*x=b, x>=0
      *   The dual formulation is:
      *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
      *
      * @warning The X blitz::array is resized and reindexed.
      * @param A The A squared-matrix of the system A*x=b (size NxN)
      * @param b The b vector of the system A*x=b (size N)
      * @param c The c vector involved in the minimization
      * @param gamma
      * @param sigma
      * @param x The x vector of the system A*x=b which will be updated 
      *   at the end of the function.
      * @param epsilon The expected precision for the algorithm to stop
      */
    void interiorpointLongstepLP(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double gamma, const double sigma, blitz::Array<double,1>& x, 
      const double epsilon);
    void interiorpointLongstepNoInitLP(const blitz::Array<double,2>& A, 
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const double gamma, const double sigma, blitz::Array<double,1>& x, 
      blitz::Array<double,1>& lambda, blitz::Array<double,1>& mu, 
      const double epsilon);


    namespace detail {
      /**
        * @brief Check if a vector is positive (all its elements)
        *
        * @param x The vector to test
        */
      bool isPositive(const blitz::Array<double,1>& x);

      /**
        * @brief Check if a primal-dual point (x,lambda,mu) belongs to the set
        *   of feasible point, i.e. fulfill the constraints:
        *     A*x=b, transpose(A)*lambda+mu=c, x>=0 and mu>=0
        *
        * @param A The A matrix of the linear equalities
        * @param b The b vector of the linear equalities
        * @param c The c vector which defines the linear objective function
        * @param x The x primal variable
        * @param lambda The lambda dual variable
        * @param mu The mu dual variable
        * @param epsilon The precision to determine whether an equality
        *   constraint is fulfilled or not.
        */
      bool isFeasible(const blitz::Array<double,2>& A, 
        const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
        const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda, 
        const blitz::Array<double,1>& mu, const double epsilon);

      /**
        * @brief Check if a primal-dual point (x,lambda,mu) belongs to the
        *   V2(theta) neighborhood of the central path.
        *     /nu ||x.*mu- vu.e|| <= theta
        * @warning This functions does not check if the belongs to the set of 
        *   of feasible points.
        *
        * @param x The x primal variable
        * @param mu The mu dual variable
        * @param theta The value defining the size of the V2 neighborhood
        */
      bool isInV2( const blitz::Array<double,1>& x,
        const blitz::Array<double,1>& mu, const double theta);

      /**
        * @brief Check if a primal-dual point (x,lambda,mu) belongs to the
        *   V-inf(gamma) neighborhood of the central path.
        *     /nu ||x.*mu- vu.e|| <= theta
        * @warning This functions does not check if the belongs to the set of 
        *   of feasible points.
        *
        * @param x The x primal variable
        * @param mu The mu dual variable
        * @param gamma The value defining the size of the V-inf neighborhood
        */
      bool isInVinf( const blitz::Array<double,1>& x,
        const blitz::Array<double,1>& mu, const double gamma);

      /**
        * @brief Check if a primal-dual point (x,lambda,mu) belongs to the
        *   V2(theta) neighborhood of the central path.
        *     /nu ||x.*mu- vu.e|| <= theta
        *   and to the set of feasible points S.
        *
        * @param A The A matrix of the linear equalities
        * @param b The b vector of the linear equalities
        * @param c The c vector which defines the linear objective function
        * @param x The x primal variable
        * @param lambda The lambda dual variable
        * @param mu The mu dual variable
        * @param epsilon The precision to determine whether an equality
        *   constraint is fulfilled or not.
        * @param theta The value defining the size of the V2 neighborhood
        */
      bool isInV2S(const blitz::Array<double,2>& A,
        const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
        const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
        const blitz::Array<double,1>& mu, const double epsilon, 
        const double theta);

      /**
        * @brief Check if a primal-dual point (x,lambda,mu) belongs to the
        *   V-inf(gamma) neighborhood of the central path.
        *     /nu ||x.*mu- vu.e|| <= theta
        *   and to the set of feasible points S.
        *
        * @param A The A matrix of the linear equalities
        * @param b The b vector of the linear equalities
        * @param c The c vector which defines the linear objective function
        * @param x The x primal variable
        * @param lambda The lambda dual variable
        * @param mu The mu dual variable
        * @param epsilon The precision to determine whether an equality
        *   constraint is fulfilled or not.
        * @param gamma The value defining the size of the V-inf neighborhood
        */
      bool isInVinfS(const blitz::Array<double,2>& A,
        const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
        const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
        const blitz::Array<double,1>& mu, const double epsilon, 
        const double gamma);

      /**
        * @brief Compute the value of the logarithmic barrier function for the
        *   given dual variable lambda. This function is called by the method
        *   which looks for an initial solution in S. 
        *
        * @param A The A matrix of the linear equalities
        * @param c The c vector which defines the linear objective function
        * @param lambda The lambda dual variable
        */
      double logBarrierLP(const blitz::Array<double,2>& A,
        const blitz::Array<double,1>& c, blitz::Array<double,1>& lambda);

      /**
        * @brief Compute the gradient of the logarithmic barrier function for 
        *   the given dual variable lambda. This function is called by the 
        *   method which looks for an initial solution in S. 
        *
        * @param A The A matrix of the linear equalities
        * @param c The c vector which defines the linear objective function
        * @param lambda The lambda dual variable
        * @param work_array
        * @param gradient
        */
      void gradientLogBarrierLP(const blitz::Array<double,2>& A,
        const blitz::Array<double,1>& c, blitz::Array<double,1>& lambda,
        blitz::Array<double,1>& work_array, blitz::Array<double,1>& gradient);

      /**
        * @brief Look for an initial solution (lambda,mu) of the dual problem
        *   by minimizing the logarithmic barrier function.
        *
        * @param A The A matrix of the linear equalities
        * @param c The c vector which defines the linear objective function
        * @param lambda The lambda dual variable, which should be initialized
        * @param mu The mu dual variable
        */
      void initializeDualLambdaMuLP(const blitz::Array<double,2>& A,
        const blitz::Array<double,1>& c, blitz::Array<double,1>& lambda, 
        blitz::Array<double,1>& mu);

      /**
        * @brief Apply centering iterations (sigma=1) until we reach a 
        *   a feasible point in the V2 neighborhood.
        *
        * @param A The A matrix of the linear equalities
        * @param theta The value defining the size of the V2 neighborhood
        * @param x The x primal variable
        * @param lambda The lambda dual variable
        * @param mu The mu dual variable
        */
      void centeringV2(const blitz::Array<double,2>& A, const double theta,
        blitz::Array<double,1>& x, blitz::Array<double,1>& lambda, 
        blitz::Array<double,1>& mu);

      /**
        * @brief Apply centering iterations (sigma=1) until we reach a 
        *   a feasible point in the V-inf neighborhood.
        *
        * @param A The A matrix of the linear equalities
        * @param gamma The value defining the size of the V-inf neighborhood
        * @param x The x primal variable
        * @param lambda The lambda dual variable
        * @param mu The mu dual variable
        */
      void centeringVinf(const blitz::Array<double,2>& A, const double gamma,
        blitz::Array<double,1>& x, blitz::Array<double,1>& lambda, 
        blitz::Array<double,1>& mu);

      /**
        * @brief Initialize the large system: 
        *   [A 0 0; 0 A^T I; S 0 X]*[Dx Dlambda Dmu] = [0 0 -x.*mu]
        *
        * @warning X=diag(x), S=diag(mu), x.*mu are not set by this function
        *   The system components A_large and b_large are set using zero base 
        *   indices.
        *
        * @param A The A matrix of the linear equalities
        * @param A_large The large left matrix of the system
        * @param b_large The large right vector of the system
        * @param x_large The large solution vector of the system
        */
      void initializeLargeSystem(const blitz::Array<double,2>& A,
        blitz::Array<double,2>& A_large, blitz::Array<double,1>& b_large,
        blitz::Array<double,1>& x_large);

      /**
        * @brief Update the large system: 
        *   [A 0 0; 0 A^T I; S 0 X]*[Dx Dlambda Dmu] = [0 0 -x.*mu]
        *
        * @warning X=diag(x), S=diag(mu), x.*mu are not set by this function
        *   The system components A_large and b_large are set using zero base 
        *   indices.
        *
        * @param x The current x primal solution of the linear program
        * @param mu The current mu dual solution of the linear program
        * @param sigma The coefficient sigma which quantifies how close we 
        *   want to stay from the central path.
        * @param m
        * @param A_large The large left matrix of the system
        * @param b_large The large right vector of the system
        */
      void updateLargeSystem(const blitz::Array<double,1>& x, 
        const blitz::Array<double,1>& mu, const double sigma, const int m,
        blitz::Array<double,2>& A_large, blitz::Array<double,1>& b_large);
    }

  }
/**
 * @}
 */
}

#endif /* BOB_MATH_INTERIOR_POINT_LP_H */
