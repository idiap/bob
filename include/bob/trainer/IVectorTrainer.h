/**
* @file bob/trainer/IVectorTrainer.h
* @date Sat Mar 30 20:55:00 2013 +0200
* @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
*
* Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3 of the License.
*
* This program is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef BOB_TRAINER_IVECTOR_H
#define BOB_TRAINER_IVECTOR_H

#include <blitz/array.h>
#include "EMTrainer.h"
#include <bob/machine/IVectorMachine.h>
#include <bob/machine/GMMStats.h>
#include <boost/shared_ptr.hpp>
#include <boost/random.hpp>
#include <vector>

namespace bob { namespace trainer {
/**
 * @ingroup TRAINER
 * @{
 */

/**
 * @brief An IVectorTrainer to learn a Total Variability subspace \f$T\f$
 *  (and eventually a covariance matrix \f$\Sigma\f$).\n
 * Reference:\n
 * "Front-End Factor Analysis For Speaker Verification",
 *    N. Dehak, P. Kenny, R. Dehak, P. Dumouchel, P. Ouellet, 
 *   IEEE Trans. on Audio, Speech and Language Processing
 */
class IVectorTrainer: public bob::trainer::EMTrainer<bob::machine::IVectorMachine, std::vector<bob::machine::GMMStats> >
{
  public:
    /**
     * @brief Default constructor. Builds an IVectorTrainer
     */
    IVectorTrainer(const bool update_sigma=false, 
      const double convergence_threshold=0.001,
      const size_t max_iterations=10, const bool compute_likelihood=false);

    /**
     * @brief Copy constructor
     */
    IVectorTrainer(const IVectorTrainer& other);

    /**
     * @brief Destructor
     */
    virtual ~IVectorTrainer();

    /**
     * @brief Initialization before the EM loop
     */
    virtual void initialize(bob::machine::IVectorMachine& ivector, 
      const std::vector<bob::machine::GMMStats>& data);
    
    /**
     * @brief Calculates statistics across the dataset, 
     * and saves these as:
     * - m_acc_Nij_wij2
     * - m_acc_Fnormij_wij
     * - m_acc_Nij (only if update_sigma is enabled)
     * - m_acc_Snormij (only if update_sigma is enabled)
     * 
     * These statistics will be used in the mStep() that follows.
     */
    virtual void eStep(bob::machine::IVectorMachine& ivector, 
      const std::vector<bob::machine::GMMStats>& data);

    /**
     * @brief Maximisation step: Update the Total Variability matrix \f$T\f$
     * and \f$\Sigma\f$ if update_sigma is enabled.
     */
    virtual void mStep(bob::machine::IVectorMachine& ivector, 
      const std::vector<bob::machine::GMMStats>& data);

    /**
     * @brief Computes the likelihood using current estimates 
     * @warning (currently unsupported)
     */
    virtual double computeLikelihood(bob::machine::IVectorMachine& ivector);

    /**
     * @brief Finalization after the EM loop
     */
    virtual void finalize(bob::machine::IVectorMachine& ivector, 
      const std::vector<bob::machine::GMMStats>& data);

    /**
     * @brief Assigns from a different IVectorTrainer
     */
    IVectorTrainer& operator=(const IVectorTrainer &other);

    /**
     * @brief Equal to
     */
    bool operator==(const IVectorTrainer& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const IVectorTrainer& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const IVectorTrainer& b, const double r_epsilon=1e-5,
      const double a_epsilon=1e-8) const;

    /**
     * @brief Getters for the accumulators
     */
    const blitz::Array<double,3>& getAccNijWij2() const 
    { return m_acc_Nij_wij2; }
    const blitz::Array<double,3>& getAccFnormijWij() const
    { return m_acc_Fnormij_wij; }
    const blitz::Array<double,1>& getAccNij() const
    { return m_acc_Nij; }
    const blitz::Array<double,2>& getAccSnormij() const
    { return m_acc_Snormij; }

    /**
     * @brief Setters for the accumulators, Very useful if the e-Step needs
     * to be parallelized.
     */
    void setAccNijWij2(const blitz::Array<double,3>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_Nij_wij2);
      m_acc_Nij_wij2 = acc; }
    void setAccFnormijWij(const blitz::Array<double,3>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_Fnormij_wij);
      m_acc_Fnormij_wij = acc; }
    void setAccNij(const blitz::Array<double,1>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_Nij);
      m_acc_Nij = acc; }
    void setAccSnormij(const blitz::Array<double,2>& acc)
    { bob::core::array::assertSameShape(acc, m_acc_Snormij);
      m_acc_Snormij = acc; }

  protected:
    // Attributes
    bool m_update_sigma;

    // Acccumulators
    blitz::Array<double,3> m_acc_Nij_wij2;
    blitz::Array<double,3> m_acc_Fnormij_wij;
    blitz::Array<double,1> m_acc_Nij;
    blitz::Array<double,2> m_acc_Snormij;
    
    // Working arrays
    mutable blitz::Array<double,1> m_tmp_wij;
    mutable blitz::Array<double,2> m_tmp_wij2;
    mutable blitz::Array<double,1> m_tmp_d1;
    mutable blitz::Array<double,1> m_tmp_t1;
    mutable blitz::Array<double,2> m_tmp_dd1;
    mutable blitz::Array<double,2> m_tmp_dt1;
    mutable blitz::Array<double,2> m_tmp_tt1;
    mutable blitz::Array<double,2> m_tmp_tt2;
};

/**
 * @}
 */
}}

#endif // BOB_TRAINER_IVECTORTRAINER_H
