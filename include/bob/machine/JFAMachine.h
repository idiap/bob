/**
 * @file bob/machine/JFAMachine.h
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief A base class for Joint Factor Analysis-like machines
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

#ifndef BOB_MACHINE_FABASE_H
#define BOB_MACHINE_FABASE_H

#include "Machine.h"
#include "GMMMachine.h"
#include "JFAMachineException.h"
#include "LinearScoring.h"

#include <bob/io/HDF5File.h>
#include <boost/shared_ptr.hpp>

namespace bob { namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

/**
 * @brief A FA Base class which contains U, V and D matrices
 * TODO: add a reference to the journal articles
 */
class FABase
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 FABase
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    FABase();

    /**
     * @brief Constructor. Builds a new FABase.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     *
     * @param ubm The Universal Background Model
     * @param ru size of U (CD x ru)
     * @param rv size of U (CD x rv)
     * @warning ru and rv SHOULD BE  >= 1. Just set U/V/D to zero if you want
     *   to ignore one subspace. This is the case for ISV.
     */ 
    FABase(const boost::shared_ptr<bob::machine::GMMMachine> ubm, const size_t ru=1, const size_t rv=1);

    /**
     * @brief Copy constructor
     */
    FABase(const FABase& other);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~FABase(); 

    /**
     * @brief Assigns from a different JFA machine
     */
    FABase& operator=(const FABase &other);

    /**
     * @brief Equal to
     */
    bool operator==(const FABase& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const FABase& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const FABase& b, const double r_epsilon=1e-5, 
      const double a_epsilon=1e-8) const; 

    /**
     * @brief Returns the UBM
     */
    const boost::shared_ptr<bob::machine::GMMMachine> getUbm() const 
    { return m_ubm; }

    /**
     * @brief Returns the U matrix
     */
    const blitz::Array<double,2>& getU() const 
    { return m_U; }

    /**
     * @brief Returns the V matrix
     */
    const blitz::Array<double,2>& getV() const 
    { return m_V; }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector)
     */
    const blitz::Array<double,1>& getD() const 
    { return m_d; }

    /**
     * @brief Returns the UBM mean supervector (as a 1D vector)
     */
    const blitz::Array<double,1>& getUbmMean() const 
    { return m_cache_mean; }

    /**
     * @brief Returns the UBM variance supervector (as a 1D vector)
     */
    const blitz::Array<double,1>& getUbmVariance() const 
    { return m_cache_sigma; }

    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimC() const 
    { if(!m_ubm) throw bob::machine::JFABaseNoUBMSet(); 
      return m_ubm->getNGaussians(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimD() const 
    { if(!m_ubm) throw bob::machine::JFABaseNoUBMSet();
      return m_ubm->getNInputs(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimCD() const 
    { if(!m_ubm) throw bob::machine::JFABaseNoUBMSet(); 
      return m_ubm->getNInputs()*m_ubm->getNGaussians(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const 
    { return m_ru; }

    /**
     * @brief Returns the size/rank rv of the V matrix
     */
    const size_t getDimRv() const 
    { return m_rv; }

    /**
     * @brief Resets the dimensionality of the subspace U and V
     * U and V are hence uninitialized.
     */ 
    void resize(const size_t ru, const size_t rv);

    /**
     * @brief Resets the dimensionality of the subspace U and V,
     * assuming that no UBM has yet been set
     * U and V are hence uninitialized.
     */ 
    void resize(const size_t ru, const size_t rv, const size_t cd);

    /**
     * @brief Returns the U matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateU() 
    { return m_U; }

    /**
     * @brief Returns the V matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateV() 
    { return m_V; }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector) in order
     * to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,1>& updateD() 
    { return m_d; }


    /**
     * @brief Sets (the mean supervector of) the Universal Background Model
     * U, V and d are uninitialized in case of dimensions update (C or D)
     */
    void setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm);

    /**
     * @brief Sets the U matrix
     */
    void setU(const blitz::Array<double,2>& U);

    /**
     * @brief Sets the V matrix
     */
    void setV(const blitz::Array<double,2>& V);

    /**
     * @brief Sets the diagonal matrix diag(d) 
     * (a 1D vector is expected as an argument)
     */
    void setD(const blitz::Array<double,1>& d);


    /**
     * @brief Estimates x from the GMM statistics considering the LPT 
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& x) const;

    /**
     * @brief Compute and put U^{T}.Sigma^{-1} matrix in cache
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    void updateCacheUbmUVD();


  private:
    /**
     * @brief Update cache arrays/variables
     */
    void updateCache();
    /**
     * @brief Put GMM mean/variance supervector in cache
     */
    void updateCacheUbm();
    /**
     * @brief Resize working arrays
     */
    void resizeTmp();
    /**
     * @brief Computes (Id + U^T.Sigma^-1.U.N_{i,h}.U)^-1 = 
     *   (Id + sum_{c=1..C} N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c})^-1
     */
    void computeIdPlusUSProdInv(const bob::machine::GMMStats& gmm_stats, 
      blitz::Array<double,2>& out) const;
    /**
     * @brief Computes Fn_x = sum_{sessions h}(N*(o - m))
     * (Normalised first order statistics)
     */
    void computeFn_x(const bob::machine::GMMStats& gmm_stats,
      blitz::Array<double,1>& out) const;
    /**
     * @brief Estimates the value of x from the passed arguments 
     * (IdPlusUSProdInv and Fn_x), considering the LPT assumption
     */
    void estimateX(const blitz::Array<double,2>& IdPlusUSProdInv,
      const blitz::Array<double,1>& Fn_x, blitz::Array<double,1>& x) const;


    // UBM
    boost::shared_ptr<bob::machine::GMMMachine> m_ubm;

    // dimensionality
    size_t m_ru; // size of U (CD x ru)
    size_t m_rv; // size of V (CD x rv)

    // U, V, D matrices
    // D is assumed to be diagonal, and only the diagonal is stored
    blitz::Array<double,2> m_U;
    blitz::Array<double,2> m_V;
    blitz::Array<double,1> m_d; 

    // Vectors/Matrices precomputed in cache
    blitz::Array<double,1> m_cache_mean;
    blitz::Array<double,1> m_cache_sigma;
    blitz::Array<double,2> m_cache_UtSigmaInv;

    mutable blitz::Array<double,2> m_tmp_IdPlusUSProdInv;
    mutable blitz::Array<double,1> m_tmp_Fn_x;
    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_ruru;
};


/**
 * @brief A JFA Base class which contains U, V and D matrices
 * TODO: add a reference to the journal articles
 */
class JFABase
{
  public:
    /**
     * @brief Default constructor. Builds a 1 x 1 JFABase
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    JFABase();

    /**
     * @brief Constructor. Builds a new JFABase.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     *
     * @param ubm The Universal Background Model
     * @param ru size of U (CD x ru)
     * @param rv size of U (CD x rv)
     * @warning ru and rv SHOULD BE  >= 1.
     */ 
    JFABase(const boost::shared_ptr<bob::machine::GMMMachine> ubm, const size_t ru=1, const size_t rv=1);

    /**
     * @brief Copy constructor
     */
    JFABase(const JFABase& other);

    /**
     * @deprecated Starts a new JFAMachine from an existing Configuration object.
     */
    JFABase(bob::io::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~JFABase(); 

    /**
     * @brief Assigns from a different JFA machine
     */
    JFABase& operator=(const JFABase &other);

    /**
     * @brief Equal to
     */
    bool operator==(const JFABase& b) const
    { return m_base.operator==(b.m_base); }

    /**
     * @brief Not equal to
     */
    bool operator!=(const JFABase& b) const
    { return m_base.operator!=(b.m_base); }

    /**
     * @brief Similar to
     */
    bool is_similar_to(const JFABase& b, const double r_epsilon=1e-5, 
      const double a_epsilon=1e-8) const
    { return m_base.is_similar_to(b.m_base, r_epsilon, a_epsilon); }

    /**
     * @brief Saves model to an HDF5 file
     */
    void save(bob::io::HDF5File& config) const;

    /**
     * @brief Loads data from an existing configuration object. Resets
     * the current state.
     */
    void load(bob::io::HDF5File& config);

    /**
     * @brief Returns the UBM
     */
    const boost::shared_ptr<bob::machine::GMMMachine> getUbm() const 
    { return m_base.getUbm(); }

    /**
     * @brief Returns the U matrix
     */
    const blitz::Array<double,2>& getU() const 
    { return m_base.getU(); }

    /**
     * @brief Returns the V matrix
     */
    const blitz::Array<double,2>& getV() const 
    { return m_base.getV(); }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector)
     */
    const blitz::Array<double,1>& getD() const 
    { return m_base.getD(); }

    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimC() const 
    { return m_base.getDimC(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimD() const 
    { return m_base.getDimD(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimCD() const 
    { return m_base.getDimCD(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const 
    { return m_base.getDimRu(); }

    /**
     * @brief Returns the size/rank rv of the V matrix
     */
    const size_t getDimRv() const 
    { return m_base.getDimRv(); }

    /**
     * @brief Resets the dimensionality of the subspace U and V
     * U and V are hence uninitialized.
     */ 
    void resize(const size_t ru, const size_t rv)
    { m_base.resize(ru, rv); }

    /**
     * @brief Returns the U matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateU() 
    { return m_base.updateU(); }

    /**
     * @brief Returns the V matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateV() 
    { return m_base.updateV(); }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector) in order
     * to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,1>& updateD() 
    { return m_base.updateD(); }


    /**
     * @brief Sets (the mean supervector of) the Universal Background Model
     * U, V and d are uninitialized in case of dimensions update (C or D)
     */
    void setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm)
    { m_base.setUbm(ubm); }

    /**
     * @brief Sets the U matrix
     */
    void setU(const blitz::Array<double,2>& U)
    { m_base.setU(U); }

    /**
     * @brief Sets the V matrix
     */
    void setV(const blitz::Array<double,2>& V)
    { m_base.setV(V); }

    /**
     * @brief Sets the diagonal matrix diag(d) 
     * (a 1D vector is expected as an argument)
     */
    void setD(const blitz::Array<double,1>& d)
    { m_base.setD(d); }

    /**
     * @brief Estimates x from the GMM statistics considering the LPT 
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_base.estimateX(gmm_stats, x); }

    /**
     * @brief Precompute (put U^{T}.Sigma^{-1} matrix in cache)
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    void precompute()
    { m_base.updateCacheUbmUVD(); }

    /**
     * @brief Returns the FABase member
     */
    const bob::machine::FABase& getBase() const
    { return m_base; }


  private:
    // FABase
    bob::machine::FABase m_base;
};


/**
 * @brief An ISV Base class which contains U and D matrices
 * TODO: add a reference to the journal articles
 */
class ISVBase
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 ISVBase
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    ISVBase();

    /**
     * @brief Constructor. Builds a new ISVBase.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     *
     * @param ubm The Universal Background Model
     * @param ru size of U (CD x ru)
     * @warning ru SHOULD BE >= 1.
     */ 
    ISVBase(const boost::shared_ptr<bob::machine::GMMMachine> ubm, const size_t ru=1);

    /**
     * @brief Copy constructor
     */
    ISVBase(const ISVBase& other);

    /**
     * @deprecated Starts a new JFAMachine from an existing Configuration object.
     */
    ISVBase(bob::io::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~ISVBase(); 

    /**
     * @brief Assigns from a different JFA machine
     */
    ISVBase& operator=(const ISVBase &other);

    /**
     * @brief Equal to
     */
    bool operator==(const ISVBase& b) const
    { return m_base.operator==(b.m_base); }

    /**
     * @brief Not equal to
     */
    bool operator!=(const ISVBase& b) const
    { return m_base.operator!=(b.m_base); }

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ISVBase& b, const double r_epsilon=1e-5, 
      const double a_epsilon=1e-8) const
    { return m_base.is_similar_to(b.m_base, r_epsilon, a_epsilon); }

    /**
     * @brief Saves machine to an HDF5 file
     */
    void save(bob::io::HDF5File& config) const;

    /**
     * @brief Loads data from an existing configuration object. Resets
     * the current state.
     */
    void load(bob::io::HDF5File& config);

    /**
     * @brief Returns the UBM
     */
    const boost::shared_ptr<bob::machine::GMMMachine> getUbm() const 
    { return m_base.getUbm(); }

    /**
     * @brief Returns the U matrix
     */
    const blitz::Array<double,2>& getU() const 
    { return m_base.getU(); }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector)
     */
    const blitz::Array<double,1>& getD() const 
    { return m_base.getD(); }

    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimC() const 
    { return m_base.getDimC(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimD() const 
    { return m_base.getDimD(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimCD() const 
    { return m_base.getDimCD(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const 
    { return m_base.getDimRu(); }

    /**
     * @brief Resets the dimensionality of the subspace U and V
     * U and V are hence uninitialized.
     */ 
    void resize(const size_t ru)
    { m_base.resize(ru, 1); 
      blitz::Array<double,2>& V = m_base.updateV();
      V = 0;
     }

    /**
     * @brief Returns the U matrix in order to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,2>& updateU() 
    { return m_base.updateU(); }

    /**
     * @brief Returns the diagonal matrix diag(d) (as a 1D vector) in order
     * to update it
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    blitz::Array<double,1>& updateD() 
    { return m_base.updateD(); }


    /**
     * @brief Sets (the mean supervector of) the Universal Background Model
     * U, V and d are uninitialized in case of dimensions update (C or D)
     */
    void setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm)
    { m_base.setUbm(ubm); }

    /**
     * @brief Sets the U matrix
     */
    void setU(const blitz::Array<double,2>& U)
    { m_base.setU(U); }

    /**
     * @brief Sets the diagonal matrix diag(d) 
     * (a 1D vector is expected as an argument)
     */
    void setD(const blitz::Array<double,1>& d)
    { m_base.setD(d); }

    /**
     * @brief Estimates x from the GMM statistics considering the LPT 
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_base.estimateX(gmm_stats, x); }

    /**
     * @brief Precompute (put U^{T}.Sigma^{-1} matrix in cache)
     * @warning Should only be used by the trainer for efficiency reason, 
     *   or for testing purpose.
     */
    void precompute()
    { m_base.updateCacheUbmUVD(); }

    /**
     * @brief Returns the FABase member
     */
    const bob::machine::FABase& getBase() const
    { return m_base; }


  private:
    // FABase
    bob::machine::FABase m_base;
};


/**
 * @brief A JFAMachine which is associated to a JFABase that contains 
 *   U, V and D matrices. The JFAMachine describes the identity part 
 *   (latent variables y and z)
 * TODO: add a reference to the journal articles
 */
class JFAMachine: public Machine<bob::machine::GMMStats, double>
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 JFAMachine
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    JFAMachine();

    /**
     * @brief Constructor. Builds a new JFAMachine.
     *
     * @param jfa_base The JFABase associated with this machine
     */ 
    JFAMachine(const boost::shared_ptr<bob::machine::JFABase> jfa_base);

    /**
     * @brief Copy constructor
     */
    JFAMachine(const JFAMachine& other);

    /**
     * @deprecated Starts a new JFAMachine from an existing Configuration object.
     */
    JFAMachine(bob::io::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~JFAMachine(); 

    /**
     * @brief Assigns from a different JFA machine
     */
    JFAMachine& operator=(const JFAMachine &other);

    /**
     * @brief Equal to
     */
    bool operator==(const JFAMachine& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const JFAMachine& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const JFAMachine& b, const double r_epsilon=1e-5, 
      const double a_epsilon=1e-8) const;

    /**
     * @brief Saves machine to an HDF5 file
     */
    void save(bob::io::HDF5File& config) const;

    /**
     * @brief Loads data from an existing configuration object. Resets
     * the current state.
     */
    void load(bob::io::HDF5File& config);

    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimC() const 
    { return m_jfa_base->getDimC(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimD() const 
    { return m_jfa_base->getDimD(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimCD() const 
    { return m_jfa_base->getDimCD(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const 
    { return m_jfa_base->getDimRu(); }

    /**
     * @brief Returns the size/rank rv of the V matrix
     */
    const size_t getDimRv() const 
    { return m_jfa_base->getDimRv(); }

    /**
     * @brief Returns the x session factor
     */
    const blitz::Array<double,1>& getX() const 
    { return m_cache_x; }

    /**
     * @brief Returns the y speaker factor
     */
    const blitz::Array<double,1>& getY() const 
    { return m_y; }

    /**
     * @brief Returns the z speaker factor
     */
    const blitz::Array<double,1>& getZ() const 
    { return m_z; }

    /**
     * @brief Returns the y speaker factors in order to update it
     */
    blitz::Array<double,1>& updateY() 
    { return m_y; }

    /**
     * @brief Returns the z speaker factors in order to update it
     */
    blitz::Array<double,1>& updateZ()
    { return m_z; }

    /**
     * @brief Returns the y speaker factors
     */
    void setY(const blitz::Array<double,1>& y);

    /**
     * @brief Returns the V matrix
     */
    void setZ(const blitz::Array<double,1>& z);

    /**
     * @brief Returns the JFABase
     */
    const boost::shared_ptr<bob::machine::JFABase> getJFABase() const
    { return m_jfa_base; }

    /**
     * @brief Sets the JFABase
     */
    void setJFABase(const boost::shared_ptr<bob::machine::JFABase> jfa_base);


    /**
     * @brief Estimates x from the GMM statistics considering the LPT 
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_jfa_base->estimateX(gmm_stats, x); }
    /**
     * @brief Estimates Ux from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateUx(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& Ux);

   /**
    * @brief Execute the machine
    *
    * @param input input data used by the machine
    * @param score value computed by the machine
    * @warning Inputs are checked
    */
    void forward(const bob::machine::GMMStats& input, double& score) const;
    /**
     * @brief Computes a score for the given UBM statistics and given the 
     * Ux vector
     */
    void forward(const bob::machine::GMMStats& gmm_stats, 
      const blitz::Array<double,1>& Ux, double& score) const;

    /**
     * @brief Execute the machine
     *
     * @param input input data used by the machine
     * @param score value computed by the machine
     * @warning Inputs are NOT checked
     */
    void forward_(const bob::machine::GMMStats& input, double& score) const;

  protected:
    /**
     * @brief Resize latent variable according to the JFABase
     */ 
    void resize();
    /**
     * @brief Resize working arrays
     */
    void resizeTmp();
    /**
     * @brief Update the cache
     */
    void updateCache();

    // UBM
    boost::shared_ptr<bob::machine::JFABase> m_jfa_base;

    // y and z vectors/factors learned during the enrolment procedure
    blitz::Array<double,1> m_y;
    blitz::Array<double,1> m_z;

    // cache
    blitz::Array<double,1> m_cache_mVyDz;
    mutable blitz::Array<double,1> m_cache_x;

    // x vector/factor in cache when computing scores
    mutable blitz::Array<double,1> m_tmp_Ux;
};

/**
 * @brief A ISVMachine which is associated to a ISVBase that contains 
 *   U D matrices. 
 * TODO: add a reference to the journal articles
 */
class ISVMachine: public Machine<bob::machine::GMMStats, double>
{
  public:
    /**
     * @brief Default constructor. Builds an otherwise invalid 0 x 0 ISVMachine
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    ISVMachine();

    /**
     * @brief Constructor. Builds a new ISVMachine.
     *
     * @param isv_base The ISVBase associated with this machine
     */ 
    ISVMachine(const boost::shared_ptr<bob::machine::ISVBase> isv_base);

    /**
     * @brief Copy constructor
     */
    ISVMachine(const ISVMachine& other);

    /**
     * @brief Starts a new ISVMachine from an existing Configuration object.
     */
    ISVMachine(bob::io::HDF5File& config);

    /**
     * @brief Just to virtualise the destructor
     */
    virtual ~ISVMachine(); 

    /**
     * @brief Assigns from a different ISV machine
     */
    ISVMachine& operator=(const ISVMachine &other);

    /**
     * @brief Equal to
     */
    bool operator==(const ISVMachine& b) const;

    /**
     * @brief Not equal to
     */
    bool operator!=(const ISVMachine& b) const;

    /**
     * @brief Similar to
     */
    bool is_similar_to(const ISVMachine& b, const double r_epsilon=1e-5, 
      const double a_epsilon=1e-8) const;

    /**
     * @brief Saves machine to an HDF5 file
     */
    void save(bob::io::HDF5File& config) const;

    /**
     * @brief Loads data from an existing configuration object. Resets
     * the current state.
     */
    void load(bob::io::HDF5File& config);


    /**
     * @brief Returns the number of Gaussian components C
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimC() const 
    { return m_isv_base->getDimC(); }

    /**
     * @brief Returns the feature dimensionality D
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimD() const 
    { return m_isv_base->getDimD(); }

    /**
     * @brief Returns the supervector length CD
     * (CxD: Number of Gaussian components by the feature dimensionality)
     * @warning An exception is thrown if no Universal Background Model has 
     *   been set yet.
     */
    const size_t getDimCD() const 
    { return m_isv_base->getDimCD(); }

    /**
     * @brief Returns the size/rank ru of the U matrix
     */
    const size_t getDimRu() const 
    { return m_isv_base->getDimRu(); }

    /**
     * @brief Returns the x session factor
     */
    const blitz::Array<double,1>& getX() const 
    { return m_cache_x; }

    /**
     * @brief Returns the z speaker factor
     */
    const blitz::Array<double,1>& getZ() const 
    { return m_z; }

    /**
     * @brief Returns the z speaker factors in order to update it
     */
    blitz::Array<double,1>& updateZ()
    { return m_z; }

    /**
     * @brief Returns the V matrix
     */
    void setZ(const blitz::Array<double,1>& z);

    /**
     * @brief Returns the ISVBase
     */
    const boost::shared_ptr<bob::machine::ISVBase> getISVBase() const
    { return m_isv_base; }

    /**
     * @brief Sets the ISVBase
     */
    void setISVBase(const boost::shared_ptr<bob::machine::ISVBase> isv_base);


    /**
     * @brief Estimates x from the GMM statistics considering the LPT 
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateX(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& x) const
    { m_isv_base->estimateX(gmm_stats, x); }
    /**
     * @brief Estimates Ux from the GMM statistics considering the LPT
     * assumption, that is the latent session variable x is approximated
     * using the UBM
     */
    void estimateUx(const bob::machine::GMMStats& gmm_stats, blitz::Array<double,1>& Ux);

   /**
    * @brief Execute the machine
    *
    * @param input input data used by the machine
    * @param score value computed by the machine
    * @warning Inputs are checked
    */
    void forward(const bob::machine::GMMStats& input, double& score) const;
    /**
     * @brief Computes a score for the given UBM statistics and given the 
     * Ux vector
     */
    void forward(const bob::machine::GMMStats& gmm_stats, 
      const blitz::Array<double,1>& Ux, double& score) const;

    /**
     * @brief Execute the machine
     *
     * @param input input data used by the machine
     * @param score value computed by the machine
     * @warning Inputs are NOT checked
     */
    void forward_(const bob::machine::GMMStats& input, double& score) const;

  protected:
    /**
     * @brief Resize latent variable according to the ISVBase
     */ 
    void resize();
    /**
     * @ Update cache
     */
    void updateCache();
    /**
     * @brief Resize working arrays
     */
    void resizeTmp();

    // UBM
    boost::shared_ptr<bob::machine::ISVBase> m_isv_base;

    // y and z vectors/factors learned during the enrolment procedure
    blitz::Array<double,1> m_z;

    // cache
    blitz::Array<double,1> m_cache_mDz;
    mutable blitz::Array<double,1> m_cache_x;

    // x vector/factor in cache when computing scores
    mutable blitz::Array<double,1> m_tmp_Ux;
};


/**
 * @}
 */
}}

#endif // BOB_MACHINE_FABASE_H
