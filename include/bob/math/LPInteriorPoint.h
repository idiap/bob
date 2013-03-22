/**
 * @file bob/math/LPInteriorPoint.h
 * @date Thu Mar 31 14:32:14 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines interior point methods which allow to solve a
 *        linear program (LP).
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

#ifndef BOB_MATH_INTERIOR_POINT_LP_H
#define BOB_MATH_INTERIOR_POINT_LP_H

#include <blitz/array.h>

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 */

/**
 * @brief Base class to solve a linear program using interior 
 *   point methods. For more details about the algorithms,
 *   please refer to the following book:
 *   "Primal-Dual Interior-Point Methods", Stephen J. Wright,
 *   ISBN: 978-0898713824, chapter 5: "Path-Following Algorithms"
 *
 *   The primal linear program (LP) is defined as follows:
 *     min transpose(c)*x, s.t. A*x=b, x>=0
 *   The dual formulation is:
 *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
 */
class LPInteriorPoint
{
  public:
    /**
     * @brief Constructor
     * @param M first dimension of the A matrix
     * @param N second dimension of the A matrix
     * @param epsilon The precision to determine whether an equality
     *   constraint is fulfilled or not.
     */
    LPInteriorPoint(const size_t M, const size_t N, const double epsilon);

    /**
     * @brief Copy constructor
     */
    LPInteriorPoint(const LPInteriorPoint& other);

    /**
     * @brief Destructor
     */
    virtual ~LPInteriorPoint() {}

    /**
     * Assigns from a different class instance
     */
    LPInteriorPoint& operator=(const LPInteriorPoint& other);

    /**
      * @brief Equal to
      */
    bool operator==(const LPInteriorPoint& b) const;
    /**
      * @brief Not equal to
      */
    bool operator!=(const LPInteriorPoint& b) const; 

    /**
     * @brief Reset the size of the problem
     */
    void reset(const size_t M, const size_t N);

    /**
     * @brief Getters
     */
    const size_t getDimM() const { return m_M; }
    const size_t getDimN() const { return m_N; }
    const double getEpsilon() const { return m_epsilon; }
    const blitz::Array<double,1>& getLambda() const { return m_lambda; }
    const blitz::Array<double,1>& getMu() const { return m_mu; }

    /**
     * @brief Setters
     */
    void setDimM(const size_t M) 
    { m_M = M; reset(m_M, m_N); }
    void setDimN(const size_t N)
    { m_N = N; reset(m_M, m_N); }
    void setEpsilon(const double epsilon) { m_epsilon = epsilon; }

    /**
     * @brief Solve the linear program
     *   Should be defined in the inherited classes and should check the
     *   dimensionality of the inputs.
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x) = 0;

    /**
     * @brief Solve the linear program using the dual variables lambda and mu
     *   Should be defined in the inherited classes and should check the
     *   dimensionality of the inputs.
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& lambda, const blitz::Array<double,1>& mu) = 0;

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
     */
    virtual bool isFeasible(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
      const blitz::Array<double,1>& mu) const;

    /**
     * @brief Check if a primal-dual point (x,lambda,mu) belongs to the
     *   V2(theta) neighborhood of the central path.
     *     /nu ||x.*mu- vu.e|| <= theta
     * @warning This functions does not check if the point belongs to the set
     *   of feasible points.
     *
     * @param x The x primal variable
     * @param mu The mu dual variable
     * @param theta The value defining the size of the V2 neighborhood
     */
    virtual bool isInV(const blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& mu, const double theta) const;

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
     * @param theta The value defining the size of the V2 neighborhood
     */
    virtual bool isInVS(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      const blitz::Array<double,1>& x, const blitz::Array<double,1>& lambda,
      const blitz::Array<double,1>& mu, const double theta) const;

    /**
     * @brief Look for an initial solution (lambda,mu) of the dual problem
     *   by minimizing the logarithmic barrier function.
     *
     * @param A The A matrix of the linear equalities
     * @param c The c vector which defines the linear objective function
     */
    virtual void initializeDualLambdaMu(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& c);

  protected:
    /**
     * @brief Apply centering iterations (sigma=1) until we reach a 
     *   a feasible point in the V2 neighborhood.
     *
     * @param A The A matrix of the linear equalities
     * @param theta The value defining the size of the V2 neighborhood
     * @param x The x primal variable
     */
    virtual void centeringV(const blitz::Array<double,2>& A,
      const double theta, blitz::Array<double,1>& x);

    /**
     * @brief Initialize the large system: 
     *   [A 0 0; 0 A^T I; S 0 X]*[Dx Dlambda Dmu] = [0 0 -x.*mu]
     *
     * @warning X=diag(x), S=diag(mu), x.*mu are not set by this function
     *   The system components A_large and b_large are set using zero base 
     *   indices.
     *
     * @param A The A matrix of the linear equalities
     */
    virtual void initializeLargeSystem(const blitz::Array<double,2>& A) const;

    /**
     * @brief Update the large system: 
     *   [A 0 0; 0 A^T I; S 0 X]*[Dx Dlambda Dmu] = [0 0 -x.*mu]
     *
     * @warning X=diag(x), S=diag(mu), x.*mu are not set by this function
     *   The system components A_large and b_large are set using zero base 
     *   indices.
     *
     * @param x The current x primal solution of the linear program
     * @param sigma The coefficient sigma which quantifies how close we 
     *   want to stay from the central path.
     * @param m
     */
    virtual void updateLargeSystem(const blitz::Array<double,1>& x, 
      const double sigma, const int m) const;


    /**
     * @brief Compute the value of the logarithmic barrier function for the
     *   given dual variable lambda. This function is called by the method
     *   which looks for an initial solution in S. 
     *
     * @param A_t The (transposed) A matrix of the linear equalities
     * @param c The c vector which defines the linear objective function
     */
    double logBarrierLP(const blitz::Array<double,2>& A_t,
      const blitz::Array<double,1>& c) const;

    /**
     * @brief Compute the gradient of the logarithmic barrier function for 
     *   the given dual variable lambda. This function is called by the 
     *   method which looks for an initial solution in S. 
     *
     * @param A The A matrix of the linear equalities
     * @param c The c vector which defines the linear objective function
     */
    void gradientLogBarrierLP(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& c);

    /**
     * @brief Initialize/reset the arrays in cache
     */
    void resetCache();

    // Attributes
    size_t m_M;
    size_t m_N;
    double m_epsilon;
    blitz::Array<double,1> m_lambda;
    blitz::Array<double,1> m_mu;

    // Cache
    mutable blitz::Array<double,1> m_cache_M;
    mutable blitz::Array<double,1> m_cache_N;
    mutable blitz::Array<double,1> m_cache_x;
    mutable blitz::Array<double,1> m_cache_lambda;
    mutable blitz::Array<double,1> m_cache_mu;
    mutable blitz::Array<double,1> m_cache_gradient;
    mutable blitz::Array<double,2> m_cache_A_large;
    mutable blitz::Array<double,1> m_cache_b_large;
    mutable blitz::Array<double,1> m_cache_x_large;
};

/**
 * @brief Class to solve a linear program using a short-step
 *   interior point method. For more details about this algorithm,
 *   please refer to the following book:
 *   "Primal-Dual Interior-Point Methods", Stephen J. Wright,
 *   ISBN: 978-0898713824, chapter 5: "Path-Following Algorithms"
 *
 *   The primal linear program (LP) is defined as follows:
 *     min transpose(c)*x, s.t. A*x=b, x>=0
 *   The dual formulation is:
 *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
 */
class LPInteriorPointShortstep: public LPInteriorPoint
{
  public:
    /**
     * @brief Constructor
     * @param M first dimension of the A matrix
     * @param N second dimension of the A matrix
     * @param theta Value defining the size of a neighborhood
     * @param epsilon The precision to determine whether an equality
     *   constraint is fulfilled or not.
     */
    LPInteriorPointShortstep(const size_t M, const size_t N, 
      const double theta, const double epsilon);

    /**
     * @brief Copy constructor
     */
    LPInteriorPointShortstep(const LPInteriorPointShortstep& other);

    /**
     * Assigns from a different class instance
     */
    LPInteriorPointShortstep& operator=(const LPInteriorPointShortstep& other);

    /**
      * @brief Equal to
      */
    bool operator==(const LPInteriorPointShortstep& b) const;
    /**
      * @brief Not equal to
      */
    bool operator!=(const LPInteriorPointShortstep& b) const; 

    /**
     * @brief Destructor
     */
    virtual ~LPInteriorPointShortstep() {}

    /**
     * @brief Getter
     */
    const double getTheta() const { return m_theta; }

    /**
     * @brief Setter
     */
    void setTheta(const double theta) { m_theta = theta; }

    /*
     * @brief Solve the linear program
     * @param A The A matrix of the system A*x=b (size MxN)
     * @param b The b vector of the system A*x=b (size M)
     * @param c The c vector involved in the minimization (size N)
     * @param x The x vector of the system A*x=b which will be updated 
     *   at the end of the function (size M)
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x);

    /*
     * @brief Solve the linear program
     * @param A The A matrix of the system A*x=b (size MxN)
     * @param b The b vector of the system A*x=b (size M)
     * @param c The c vector involved in the minimization (size N)
     * @param x The x vector of the system A*x=b which will be updated 
     *   at the end of the function (size M)
     * @param lambda The lambda dual variable
     * @param mu The mu dual variable
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& lambda, const blitz::Array<double,1>& mu);

  protected:
    // Attributes
    double m_theta;
};

/**
 * @brief Class to solve a linear program using a predictor-corrector
 *   interior point method. For more details about this algorithm,
 *   please refer to the following book:
 *   "Primal-Dual Interior-Point Methods", Stephen J. Wright,
 *   ISBN: 978-0898713824, chapter 5: "Path-Following Algorithms"
 *
 *   The primal linear program (LP) is defined as follows:
 *     min transpose(c)*x, s.t. A*x=b, x>=0
 *   The dual formulation is:
 *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
 */
class LPInteriorPointPredictorCorrector: public LPInteriorPoint
{
  public:
    /**
     * @brief Constructor
     * @param M first dimension of the A matrix
     * @param N second dimension of the A matrix
     * @param theta_pred Threshold for the prediction
     * @param theta_corr Threshold for the correction
     * @param epsilon The precision to determine whether an equality
     *   constraint is fulfilled or not.
     */
    LPInteriorPointPredictorCorrector(const size_t M, const size_t N,
      const double theta_pred, const double theta_corr, const double epsilon);

    /**
     * @brief Copy constructor
     */
    LPInteriorPointPredictorCorrector(const LPInteriorPointPredictorCorrector& other);

    /**
     * Assigns from a different class instance
     */
    LPInteriorPointPredictorCorrector& operator=(const LPInteriorPointPredictorCorrector& other);

    /**
      * @brief Equal to
      */
    bool operator==(const LPInteriorPointPredictorCorrector& b) const;
    /**
      * @brief Not equal to
      */
    bool operator!=(const LPInteriorPointPredictorCorrector& b) const; 

    /**
     * @brief Destructor
     */
    virtual ~LPInteriorPointPredictorCorrector() {}

    /**
     * @brief Getters
     */
    const double getThetaPred() const { return m_theta_pred; }
    const double getThetaCorr() const { return m_theta_corr; }

    /**
     * @brief Setters
     */
    void setThetaPred(const double theta_pred) { m_theta_pred = theta_pred; }
    void setThetaCorr(const double theta_corr) { m_theta_corr = theta_corr; }

    /*
     * @brief Solve the linear program
     * @param A The A matrix of the system A*x=b (size MxN)
     * @param b The b vector of the system A*x=b (size M)
     * @param c The c vector involved in the minimization (size N)
     * @param x The x vector of the system A*x=b which will be updated 
     *   at the end of the function (size M)
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x);

    /*
     * @brief Solve the linear program
     * @param A The A matrix of the system A*x=b (size MxN)
     * @param b The b vector of the system A*x=b (size M)
     * @param c The c vector involved in the minimization (size N)
     * @param x The x vector of the system A*x=b which will be updated 
     *   at the end of the function (size M)
     * @param lambda The lambda dual variable
     * @param mu The mu dual variable
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& lambda, const blitz::Array<double,1>& mu);

  protected:
    // Attributes
    double m_theta_pred;
    double m_theta_corr;
};

/**
 * @brief Class to solve a linear program using a long-step
 *   interior point method. For more details about this algorithm,
 *   please refer to the following book:
 *   "Primal-Dual Interior-Point Methods", Stephen J. Wright,
 *   ISBN: 978-0898713824, chapter 5: "Path-Following Algorithms"
 *
 *   The primal linear program (LP) is defined as follows:
 *     min transpose(c)*x, s.t. A*x=b, x>=0
 *   The dual formulation is:
 *     min transpose(b)*lambda, s.t. transpose(A)*lambda+mu=c
 */
class LPInteriorPointLongstep: public LPInteriorPoint
{
  public:
    /**
     * @brief Constructor
     * @param M first dimension of the A matrix
     * @param N second dimension of the A matrix
     * @param gamma Defines the size of the Vinf neighborhood
     * @param sigma
     * @param epsilon The precision to determine whether an equality
     *   constraint is fulfilled or not.
     */
    LPInteriorPointLongstep(const size_t M, const size_t N, 
      const double gamma, const double sigma, const double epsilon);

    /**
     * @brief Copy constructor
     */
    LPInteriorPointLongstep(const LPInteriorPointLongstep& other);

    /**
     * Assigns from a different class instance
     */
    LPInteriorPointLongstep& operator=(const LPInteriorPointLongstep& other);

    /**
      * @brief Equal to
      */
    bool operator==(const LPInteriorPointLongstep& b) const;
    /**
      * @brief Not equal to
      */
    bool operator!=(const LPInteriorPointLongstep& b) const; 

    /**
     * @brief Destructor
     */
    virtual ~LPInteriorPointLongstep() {}

    /**
     * @brief Getters
     */
    const double getGamma() const { return m_gamma; }
    const double getSigma() const { return m_sigma; }

    /**
     * @brief Setter
     */
    void setGamma(const double gamma) { m_gamma = gamma; }
    void setSigma(const double sigma) { m_sigma = sigma; }

    /*
     * @brief Solve the linear program
     * @param A The A matrix of the system A*x=b (size MxN)
     * @param b The b vector of the system A*x=b (size M)
     * @param c The c vector involved in the minimization (size N)
     * @param x The x vector of the system A*x=b which will be updated 
     *   at the end of the function (size M)
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x);

    /*
     * @brief Solve the linear program
     * @param A The A matrix of the system A*x=b (size MxN)
     * @param b The b vector of the system A*x=b (size M)
     * @param c The c vector involved in the minimization (size N)
     * @param x The x vector of the system A*x=b which will be updated 
     *   at the end of the function (size M)
     * @param lambda The lambda dual variable
     * @param mu The mu dual variable
     */
    virtual void solve(const blitz::Array<double,2>& A,
      const blitz::Array<double,1>& b, const blitz::Array<double,1>& c,
      blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& lambda, const blitz::Array<double,1>& mu);

    /**
     * @brief Check if a primal-dual point (x,lambda,mu) belongs to the
     *   V-inf(gamma) neighborhood of the central path.
     *     ||x.*mu|| <= gamma.nu
     * @warning This functions does not check if the belongs to the set of 
     *   of feasible points.
     *
     * @param x The x primal variable
     * @param mu The mu dual variable
     * @param gamma The value defining the size of the V-inf neighborhood
     */
    virtual bool isInV(const blitz::Array<double,1>& x, 
      const blitz::Array<double,1>& mu, const double gamma) const;

  protected:
    // Attributes
    double m_gamma;
    double m_sigma;
};

/**
 * @}
 */
}}

#endif /* BOB_MATH_INTERIOR_POINT_LP_H */
