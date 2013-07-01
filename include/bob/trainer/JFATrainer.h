/**
 * @file bob/trainer/JFATrainer.h
 * @date Tue Jul 19 12:16:17 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief JFA functions
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

#ifndef BOB_TRAINER_JFATRAINER_H
#define BOB_TRAINER_JFATRAINER_H

#include <blitz/array.h>
#include "EMTrainer.h"
#include <bob/machine/GMMStats.h>
#include <bob/machine/JFAMachine.h>
#include <vector>

#include <map>
#include <string>
#include <bob/core/array_copy.h>
#include <boost/shared_ptr.hpp>
#include <bob/core/logging.h>

namespace bob { namespace trainer { 

class FABaseTrainer
{
  public:
    /**
     * @brief Constructor
     */
    FABaseTrainer();

    /**
     * @brief Copy constructor
     */
    FABaseTrainer(const FABaseTrainer& other);

    /**
     * @brief Destructor
     */
    ~FABaseTrainer();

    /**
     * @brief Check that the dimensionality of the statistics match.
     */
    void checkStatistics(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);

    /**
     * @brief Initialize the dimensionality, the UBM, the sums of the 
     * statistics and the number of identities.
     */
    void initUbmNidSumStatistics(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);

    /**
     * @brief Precomputes the sums of the zeroth order statistics over the
     * sessions for each client
     */
    void precomputeSumStatisticsN(const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Precomputes the sums of the first order statistics over the
     * sessions for each client
     */
    void precomputeSumStatisticsF(const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);

    /**
     * @brief Initializes (allocates and sets to zero) the x, y, z speaker
     * factors
     */
    void initializeXYZ(const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);

    /**
     * @brief Resets the x, y, z speaker factors to zero values
     */
    void resetXYZ();


    /**** Y and V functions ****/
    /**
     * @brief Computes Vt * diag(sigma)^-1
     */
    void computeVtSigmaInv(const bob::machine::FABase& m);
    /**
     * @brief Computes Vt_{c} * diag(sigma)^-1 * V_{c} for each Gaussian c
     */
    void computeVProd(const bob::machine::FABase& m);
    /**
     * @brief Computes (I+Vt*diag(sigma)^-1*Ni*V)^-1 which occurs in the y 
     * estimation for the given person
     */
    void computeIdPlusVProd_i(const size_t id);
    /**
     * @brief Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h}) 
     * which occurs in the y estimation of the given person
     */
    void computeFn_y_i(const bob::machine::FABase& m,
      const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& stats,
      const size_t id);
    /**
     * @brief Updates y_i (of the current person) and the accumulators to
     * compute V with the cache values m_cache_IdPlusVprod_i, m_VtSigmaInv and
     * m_cache_Fn_y_i
     */
    void updateY_i(const size_t id);
    /**
     * @brief Updates y and the accumulators to compute V 
     */
    void updateY(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Computes the accumulators m_acc_V_A1 and m_acc_V_A2 for V
     * V = A2 * A1^-1
     */
    void computeAccumulatorsV(const bob::machine::FABase& m, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Updates V from the accumulators m_acc_V_A1 and m_acc_V_A2 
     */
    void updateV(blitz::Array<double,2>& V);


    /**** X and U functions ****/
    /**
     * @brief Computes Ut * diag(sigma)^-1
     */
    void computeUtSigmaInv(const bob::machine::FABase& m);
    /**
     * @brief Computes Ut_{c} * diag(sigma)^-1 * U_{c} for each Gaussian c
     */
    void computeUProd(const bob::machine::FABase& m);
    /**
     * @brief Computes (I+Ut*diag(sigma)^-1*Ni*U)^-1 which occurs in the x 
     * estimation
     */
    void computeIdPlusUProd_ih(const boost::shared_ptr<bob::machine::GMMStats>& stats);
    /**
     * @brief Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - D*z_{i} - U*x_{i,h})
     * which occurs in the y estimation of the given person
     */
    void computeFn_x_ih(const bob::machine::FABase& m, 
      const boost::shared_ptr<bob::machine::GMMStats>& stats, const size_t id);
    /**
     * @brief Updates x_ih (of the current person/session) and the 
     * accumulators to compute U with the cache values m_cache_IdPlusVprod_i, 
     * m_VtSigmaInv and m_cache_Fn_y_i
     */
    void updateX_ih(const size_t id, const size_t h);
    /**
     * @brief Updates x
     */
    void updateX(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Computes the accumulators m_acc_U_A1 and m_acc_U_A2 for U
     * U = A2 * A1^-1
     */
    void computeAccumulatorsU(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Updates U from the accumulators m_acc_U_A1 and m_acc_U_A2
     */
    void updateU(blitz::Array<double,2>& U);


    /**** z and D functions ****/
    /**
     * @brief Computes diag(D) * diag(sigma)^-1
     */
    void computeDtSigmaInv(const bob::machine::FABase& m);
    /**
     * @brief Computes Dt_{c} * diag(sigma)^-1 * D_{c} for each Gaussian c
     */
    void computeDProd(const bob::machine::FABase& m);
    /**
     * @brief Computes (I+diag(d)t*diag(sigma)^-1*Ni*diag(d))^-1 which occurs
     * in the z estimation for the given person
     */
    void computeIdPlusDProd_i(const size_t id);
    /**
     * @brief Computes sum_{sessions h}(N_{i,h}*(o_{i,h} - m - V*y_{i} - U*x_{i,h})
     * which occurs in the y estimation of the given person
     */
    void computeFn_z_i(const bob::machine::FABase& m,
      const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& stats, const size_t id);
    /**
     * @brief Updates z_i (of the current person) and the accumulators to 
     * compute D with the cache values m_cache_IdPlusDProd_i, m_VtSigmaInv
     * and m_cache_Fn_z_i
     */
    void updateZ_i(const size_t id);
    /**
     * @brief Updates z and the accumulators to compute D
     */
    void updateZ(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Computes the accumulators m_acc_D_A1 and m_acc_D_A2 for d
     * d = A2 * A1^-1
     */
    void computeAccumulatorsD(const bob::machine::FABase& m,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& stats);
    /**
     * @brief Updates d from the accumulators m_acc_D_A1 and m_acc_D_A2
     */
    void updateD(blitz::Array<double,1>& d);


    /**
     * @brief Initializes randomly a 1D vector
     */
    static void initializeRandom(blitz::Array<double,1>& vector);
    /**
     * @brief Initializes randomly a 2D matrix
     */
    static void initializeRandom(blitz::Array<double,2>& matrix);


    /**
     * @brief Get the zeroth order statistics
     */
    const std::vector<blitz::Array<double,1> >& getNacc() const
    { return m_Nacc; }
    /**
     * @brief Get the first order statistics
     */
    const std::vector<blitz::Array<double,1> >& getFacc() const
    { return m_Facc; }
    /**
     * @brief Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_x; }
    /**
     * @brief Get the y speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getY() const
    { return m_y; }
    /**
     * @brief Get the z speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getZ() const
    { return m_z; }
    /**
     * @brief Set the x speaker factors
     */
    void setX(const std::vector<blitz::Array<double,2> >& X)
    { m_x = X; }
    /**
     * @brief Set the y speaker factors
     */
    void setY(const std::vector<blitz::Array<double,1> >& y)
    { m_y = y; }
    /**
     * @brief Set the z speaker factors
     */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_z = z; }

    /**
     * @brief Initializes the cache to process the given statistics
     */
    void initCache();

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccVA1() const
    { return m_acc_V_A1; }
    const blitz::Array<double,2>& getAccVA2() const
    { return m_acc_V_A2; }
    const blitz::Array<double,3>& getAccUA1() const
    { return m_acc_U_A1; }
    const blitz::Array<double,2>& getAccUA2() const
    { return m_acc_U_A2; }
    const blitz::Array<double,1>& getAccDA1() const
    { return m_acc_D_A1; }
    const blitz::Array<double,1>& getAccDA2() const
    { return m_acc_D_A2; }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccVA1(const blitz::Array<double,3>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_V_A1);
      m_acc_V_A1 = acc; }
    void setAccVA2(const blitz::Array<double,2>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_V_A2);
      m_acc_V_A2 = acc; }
    void setAccUA1(const blitz::Array<double,3>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_U_A1);
      m_acc_U_A1 = acc; }
    void setAccUA2(const blitz::Array<double,2>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_U_A2);
      m_acc_U_A2 = acc; }
    void setAccDA1(const blitz::Array<double,1>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_D_A1);
      m_acc_D_A1 = acc; }
    void setAccDA2(const blitz::Array<double,1>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_D_A2);
      m_acc_D_A2 = acc; }


  private:
    size_t m_Nid; // Number of identities 
    size_t m_dim_C; // Number of Gaussian components of the UBM GMM
    size_t m_dim_D; // Dimensionality of the feature space
    size_t m_dim_ru; // Rank of the U subspace
    size_t m_dim_rv; // Rank of the V subspace

    std::vector<blitz::Array<double,2> > m_x; // matrix x of speaker factors for eigenchannels U, for each client
    std::vector<blitz::Array<double,1> > m_y; // vector y of spealer factors for eigenvoices V, for each client
    std::vector<blitz::Array<double,1> > m_z; // vector z of spealer factors for eigenvoices Z, for each client

    std::vector<blitz::Array<double,1> > m_Nacc; // Sum of the zeroth order statistics over the sessions for each client, dimension C
    std::vector<blitz::Array<double,1> > m_Facc; // Sum of the first order statistics over the sessions for each client, dimension CD

    // Accumulators for the M-step
    blitz::Array<double,3> m_acc_V_A1;
    blitz::Array<double,2> m_acc_V_A2;
    blitz::Array<double,3> m_acc_U_A1;
    blitz::Array<double,2> m_acc_U_A2;
    blitz::Array<double,1> m_acc_D_A1;
    blitz::Array<double,1> m_acc_D_A2;

    // Cache/Precomputation
    blitz::Array<double,2> m_cache_VtSigmaInv; // Vt * diag(sigma)^-1
    blitz::Array<double,3> m_cache_VProd; // first dimension is the Gaussian id
    blitz::Array<double,2> m_cache_IdPlusVProd_i;
    blitz::Array<double,1> m_cache_Fn_y_i;

    blitz::Array<double,2> m_cache_UtSigmaInv; // Ut * diag(sigma)^-1
    blitz::Array<double,3> m_cache_UProd; // first dimension is the Gaussian id
    blitz::Array<double,2> m_cache_IdPlusUProd_ih;
    blitz::Array<double,1> m_cache_Fn_x_ih;

    blitz::Array<double,1> m_cache_DtSigmaInv; // Dt * diag(sigma)^-1
    blitz::Array<double,1> m_cache_DProd; // supervector length dimension
    blitz::Array<double,1> m_cache_IdPlusDProd_i;
    blitz::Array<double,1> m_cache_Fn_z_i;

    // Working arrays
    mutable blitz::Array<double,2> m_tmp_ruru;
    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_rvrv;
    mutable blitz::Array<double,2> m_tmp_rvD;
    mutable blitz::Array<double,1> m_tmp_rv;
    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,1> m_tmp_CD;
    mutable blitz::Array<double,1> m_tmp_CD_b;
};


class JFATrainer
{
  public:
    /**
     * @brief Constructor
     */
    JFATrainer(const size_t max_iterations=10);

    /**
     * @brief Copy onstructor
     */
    JFATrainer(const JFATrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~JFATrainer();

    /**
     * @brief Assignment operator
     */
    JFATrainer& operator=(const JFATrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const JFATrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const JFATrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const JFATrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Sets the maximum number of EM-like iterations (for each subspace)
     */
    void setMaxIterations(const size_t max_iterations) 
    { m_max_iterations = max_iterations; }

    /**
     * @brief Gets the maximum number of EM-like iterations (for each subspace)
     */
    size_t getMaxIterations() const 
    { return m_max_iterations; }

    /**
     * @brief This methods performs some initialization before the EM loop.
     */
    virtual void initialize(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);

    /**
     * @brief This methods performs the e-Step to train the first subspace V
     */
    virtual void eStep1(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the m-Step to train the first subspace V
     */
    virtual void mStep1(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the finalization after training the first 
     * subspace V
     */
    virtual void finalize1(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the e-Step to train the second subspace U
     */
    virtual void eStep2(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the m-Step to train the second subspace U
     */
    virtual void mStep2(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the finalization after training the second 
     * subspace U
     */
    virtual void finalize2(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the e-Step to train the third subspace d
     */
    virtual void eStep3(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the m-Step to train the third subspace d
     */
    virtual void mStep3(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs the finalization after training the third
     * subspace d
     */
    virtual void finalize3(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);

    /**
     * @brief This methods performs the main loops to train the subspaces U, V and d
     */
    virtual void train_loop(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods trains the subspaces U, V and d
     */
    virtual void train(bob::machine::JFABase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
   
    /**
     * @brief Enrol a client
     */
    void enrol(bob::machine::JFAMachine& machine,
      const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& features,
      const size_t n_iter);

    /** 
     * @brief Sets the Random Number Generator
     */
    void setRng(const boost::shared_ptr<boost::mt19937> rng)
    { m_rng = rng; }

    /** 
     * @brief Gets the Random Number Generator
     */
    const boost::shared_ptr<boost::mt19937> getRng() const
    { return m_rng; }

    /**
     * @brief Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_base_trainer.getX(); }
    /**
     * @brief Get the y speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getY() const
    { return m_base_trainer.getY(); }
    /**
     * @brief Get the z speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getZ() const
    { return m_base_trainer.getZ(); }
    /**
     * @brief Set the x speaker factors
     */
    void setX(const std::vector<blitz::Array<double,2> >& X)
    { m_base_trainer.setX(X); }
    /**
     * @brief Set the y speaker factors
     */
    void setY(const std::vector<blitz::Array<double,1> >& y)
    { m_base_trainer.setY(y); }
    /**
     * @brief Set the z speaker factors
     */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_base_trainer.setZ(z); }

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccVA1() const
    { return m_base_trainer.getAccVA1(); }
    const blitz::Array<double,2>& getAccVA2() const
    { return m_base_trainer.getAccVA2(); }
    const blitz::Array<double,3>& getAccUA1() const
    { return m_base_trainer.getAccUA1(); }
    const blitz::Array<double,2>& getAccUA2() const
    { return m_base_trainer.getAccUA2(); }
    const blitz::Array<double,1>& getAccDA1() const
    { return m_base_trainer.getAccDA1(); }
    const blitz::Array<double,1>& getAccDA2() const
    { return m_base_trainer.getAccDA2(); }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccVA1(const blitz::Array<double,3>& acc)
    { m_base_trainer.setAccVA1(acc); }
    void setAccVA2(const blitz::Array<double,2>& acc)
    { m_base_trainer.setAccVA2(acc); }
    void setAccUA1(const blitz::Array<double,3>& acc)
    { m_base_trainer.setAccUA1(acc); }
    void setAccUA2(const blitz::Array<double,2>& acc)
    { m_base_trainer.setAccUA2(acc); }
    void setAccDA1(const blitz::Array<double,1>& acc)
    { m_base_trainer.setAccDA1(acc); }
    void setAccDA2(const blitz::Array<double,1>& acc)
    { m_base_trainer.setAccDA2(acc); }


  private:
    // Attributes
    size_t m_max_iterations;
    boost::shared_ptr<boost::mt19937> m_rng; ///< The random number generator for the inialization
    bob::trainer::FABaseTrainer m_base_trainer;
};


class ISVTrainer: public EMTrainer<bob::machine::ISVBase, std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > > >
{
  public:
    /**
     * @brief Constructor
     */
    ISVTrainer(const size_t max_iterations=10, const double relevance_factor=4.);

    /**
     * @brief Copy onstructor
     */
    ISVTrainer(const ISVTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~ISVTrainer();

    /**
     * @brief Assignment operator
     */
    ISVTrainer& operator=(const ISVTrainer& other);

    /**
     * @brief Equal to
     */
    bool operator==(const ISVTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const ISVTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ISVTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief This methods performs some initialization before the EM loop.
     */
    virtual void initialize(bob::machine::ISVBase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    /**
     * @brief This methods performs some actions after the EM loop.
     */
    virtual void finalize(bob::machine::ISVBase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);
    
    /**
     * @brief Calculates and saves statistics across the dataset
     * The statistics will be used in the mStep() that follows.
     */
    virtual void eStep(bob::machine::ISVBase& machine, 
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);

    /**
     * @brief Performs a maximization step to update the parameters of the
     * factor analysis model.
     */
    virtual void mStep(bob::machine::ISVBase& machine,
      const std::vector<std::vector<boost::shared_ptr<bob::machine::GMMStats> > >& ar);

    /**
     * @brief Computes the average log likelihood using the current estimates
     * of the latent variables.
     */
    virtual double computeLikelihood(bob::machine::ISVBase& machine);

    /**
     * @brief Enrol a client
     */
    void enrol(bob::machine::ISVMachine& machine,
      const std::vector<boost::shared_ptr<bob::machine::GMMStats> >& features,
      const size_t n_iter);

    /**
     * @brief Get the x speaker factors
     */
    const std::vector<blitz::Array<double,2> >& getX() const
    { return m_base_trainer.getX(); }
    /**
     * @brief Get the z speaker factors
     */
    const std::vector<blitz::Array<double,1> >& getZ() const
    { return m_base_trainer.getZ(); }
    /**
     * @brief Set the x speaker factors
     */
    void setX(const std::vector<blitz::Array<double,2> >& X)
    { m_base_trainer.setX(X); }
    /**
     * @brief Set the z speaker factors
     */
    void setZ(const std::vector<blitz::Array<double,1> >& z)
    { m_base_trainer.setZ(z); }

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccUA1() const
    { return m_base_trainer.getAccUA1(); }
    const blitz::Array<double,2>& getAccUA2() const
    { return m_base_trainer.getAccUA2(); }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccUA1(const blitz::Array<double,3>& acc)
    { m_base_trainer.setAccUA1(acc); }
    void setAccUA2(const blitz::Array<double,2>& acc)
    { m_base_trainer.setAccUA2(acc); }


  private:
    /**
     * @brief Initialize D to sqrt(ubm_var/relevance_factor)
     */
    void initializeD(bob::machine::ISVBase& machine) const;

    // Attributes
    bob::trainer::FABaseTrainer m_base_trainer;
    double m_relevance_factor;
};


}}

#endif /* BOB_TRAINER_FATRAINER_H */
