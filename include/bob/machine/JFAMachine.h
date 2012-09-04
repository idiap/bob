/**
 * @file cxx/machine/machine/JFAMachine.h
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Machines that implements the Joint Factor Analysis and 
 *   Inter Session Variability models
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

#ifndef BOB_MACHINE_JFAMACHINE_H
#define BOB_MACHINE_JFAMACHINE_H

#include "bob/io/Arrayset.h"
#include "bob/io/HDF5File.h"
#include "bob/machine/GMMMachine.h"
#include "bob/machine/JFAMachineException.h"
#include "bob/machine/LinearScoring.h"

#include <boost/shared_ptr.hpp>

namespace bob { namespace machine {
  
/**
 * A JFA Base machine which contains U, V and D matrices
 * TODO: add a reference to the journal article
 */
class JFABaseMachine 
{
  public:
    /**
     * Default constructor. Builds an otherwise invalid 0 x 0 JFA machine.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     */
    JFABaseMachine();

    /**
     * Constructor. Builds a new JFA machine.
     * The Universal Background Model and the matrices U, V and diag(d) are
     * not initialized.
     *
     * @param ubm The Universal Background Model
     * @param ru size of U (CD x ru)
     * @param rv size of U (CD x rv)
     * @warning ru and rv SHOULD BE  >= 1. Just set U/V to zero if you want
     *   to ignore one subspace. This is the case for ISV.
     */ 
    JFABaseMachine(const boost::shared_ptr<bob::machine::GMMMachine> ubm, const size_t ru=1, const size_t rv=1);

    /**
     * Copies another machine
     */
    JFABaseMachine(const JFABaseMachine& other);

    /**
     * Starts a new JFAMachine from an existing Configuration object.
     */
    JFABaseMachine(bob::io::HDF5File& config);

    /**
     * Just to virtualise the destructor
     */
    virtual ~JFABaseMachine(); 

    /**
     * Assigns from a different JFA machine
     */
    JFABaseMachine& operator=(const JFABaseMachine &other);

    /**
     * Equal to
     */
    bool operator==(const JFABaseMachine& b) const;    

    /**
     * Loads data from an existing configuration object. Resets the current
     * state.
     */
    void load(bob::io::HDF5File& config);

    /**
     * Saves an existing JFA machine to a Configuration object.
     */
    void save(bob::io::HDF5File& config) const;

    /**
      * Returns the UBM
      */
    const boost::shared_ptr<bob::machine::GMMMachine> getUbm() const 
    { return m_ubm; }

    /**
      * Returns the U matrix
      */
    const blitz::Array<double,2>& getU() const 
    { return m_U; }

    /**
      * Returns the V matrix
      */
    const blitz::Array<double,2>& getV() const 
    { return m_V; }

    /**
      * Returns the diagonal matrix diag(d) (as a 1D vector)
      */
    const blitz::Array<double,1>& getD() const 
    { return m_d; }

    /**
      * Returns the number of Gaussian components C
      * @warning An exception is thrown if no Universal Background Model has 
      *   been set yet.
      */
    const size_t getDimC() const 
    { if(!m_ubm) throw bob::machine::JFABaseNoUBMSet(); 
      return m_ubm->getNGaussians(); }

    /**
      * Returns the feature dimensionality D
      * @warning An exception is thrown if no Universal Background Model has 
      *   been set yet.
      */
    const size_t getDimD() const 
    { if(!m_ubm) throw bob::machine::JFABaseNoUBMSet();
      return m_ubm->getNInputs(); }

    /**
      * Returns the supervector length CD
      * (CxD: Number of Gaussian components by the feature dimensionality)
      * @warning An exception is thrown if no Universal Background Model has 
      *   been set yet.
      */
    const size_t getDimCD() const 
    { if(!m_ubm) throw bob::machine::JFABaseNoUBMSet(); 
      return m_ubm->getNInputs()*m_ubm->getNGaussians(); }

    /**
      * Returns the size/rank ru of the U matrix
      */
    const size_t getDimRu() const 
    { return m_ru; }

    /**
      * Returns the size/rank rv of the V matrix
      */
    const size_t getDimRv() const 
    { return m_rv; }

    /**
     * Resets the dimensionality of the subspace U and V
     * U and V are hence uninitialized.
     */ 
    void resize(const size_t ru, const size_t rv);

    /**
      * Returns the U matrix in order to update it
      * @warning Should only be used by the trainer for efficiency reason, 
      *   or for testing purpose.
      */
    blitz::Array<double,2>& updateU() 
    { return m_U; }

    /**
      * Returns the V matrix in order to update it
      * @warning Should only be used by the trainer for efficiency reason, 
      *   or for testing purpose.
      */
    blitz::Array<double,2>& updateV() 
    { return m_V; }

    /**
      * Returns the diagonal matrix diag(d) (as a 1D vector) in order to 
      * update it
      * @warning Should only be used by the trainer for efficiency reason, 
      *   or for testing purpose.
      */
    blitz::Array<double,1>& updateD() 
    { return m_d; }


    /**
      * Sets (the mean supervector of) the Universal Background Model
      * U, V and d are uninitialized in case of dimensions update (C or D)
      */
    void setUbm(const boost::shared_ptr<bob::machine::GMMMachine> ubm);

    /**
      * Sets the U matrix
      */
    void setU(const blitz::Array<double,2>& U);

    /**
      * Sets the V matrix
      */
    void setV(const blitz::Array<double,2>& V);

    /**
      * Sets the diagonal matrix diag(d) 
      * (a 1D vector is expected as an argument)
      */
    void setD(const blitz::Array<double,1>& d);


  private:
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
};


/**
 * A JFA machine which contains y and z vectors/factors, and which can be 
 * used to process data. 
 * It should be used in combination with a JFABaseMachine and a Universal 
 * Background Model (UBM). Please note that several JFAMachines might share 
 * the same JFABaseMachine and UBM.
 */
class JFAMachine 
{
  public:
    /**
     * Default constructor. Builds an otherwise invalid 0 x 0 jfa machine.
     */
    JFAMachine();

    /**
     * Constructor, builds a new jfa machine, setting a JFABaseMachine. 
     * y and z are not initialized.
     */ 
    JFAMachine(const boost::shared_ptr<bob::machine::JFABaseMachine> jfabase);

    /**
     * Copies another machine
     */
    JFAMachine(const JFAMachine& other);

    /**
     * Starts a new JFAMachine from an existing Configuration object.
     */
    JFAMachine(bob::io::HDF5File& config);

    /**
     * Just to virtualise the destructor
     */
    virtual ~JFAMachine(); 

    /**
     * Assigns from a different machine
     */
    JFAMachine& operator=(const JFAMachine &other);

    /**
     * Equal to
     * @warning Also checks that the UBM are equal if any
     */
    bool operator==(const JFAMachine &m) const;    

    /**
     * Loads data from an existing configuration object. Resets the current
     * state.
     */
    void load(bob::io::HDF5File& config);

    /**
     * Saves an existing machine to a Configuration object.
     */
    void save(bob::io::HDF5File& config) const;

    /**
      * Returns the JFABaseMachine
      */
    const boost::shared_ptr<bob::machine::JFABaseMachine> getJFABase() const 
    { return m_jfa_base; }

    /**
      * Returns the number of Gaussian components C
      */
    const size_t getDimC() const 
    { if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet(); 
      return m_jfa_base->getUbm()->getNGaussians(); }

    /**
      * Returns the feature dimensionality D
      */
    const size_t getDimD() const 
    { if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet(); 
      return m_jfa_base->getUbm()->getNInputs(); }

    /**
      * Returns the supervector length CxD
      */
    const size_t getDimCD() const 
    { if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet(); 
      return m_jfa_base->getUbm()->getNInputs()*m_jfa_base->getUbm()->getNGaussians(); }

    /**
      * Returns the size/rank ru of the U matrix
      */
    const size_t getDimRu() const 
    { if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet(); 
      return m_jfa_base->getDimRu(); }

    /**
      * Returns the size/rank rv of the V matrix
      */
    const size_t getDimRv() const 
    { if(!m_jfa_base) throw bob::machine::JFAMachineNoJFABaseSet(); 
      return m_jfa_base->getDimRv(); }

    /**
      * Returns the last estimated x speaker factor
      * @warning For test purpose only
      */
    const blitz::Array<double,1>& getX() const 
    { return m_x; }

    /**
      * Returns the y speaker factor
      */
    const blitz::Array<double,1>& getY() const 
    { return m_y; }

    /**
      * Returns the z speaker factor
      */
    const blitz::Array<double,1>& getZ() const 
    { return m_z; }


    /**
      * Returns the y speaker factors in order to update it
      */
    blitz::Array<double,1>& updateY() 
    { return m_y; }

    /**
      * Returns the z speaker factors in order to update it
      */
    blitz::Array<double,1>& updateZ()
    { return m_z; }

    /**
      * Returns the y speaker factors
      */
    void setY(const blitz::Array<double,1>& y);

    /**
      * Returns the V matrix
      */
    void setZ(const blitz::Array<double,1>& z);

    /**
      * Returns the JFABaseMachine
      */
    void setJFABase(const boost::shared_ptr<bob::machine::JFABaseMachine> jfa_base);

    /**
      * Estimates from a 2D blitz::Array
      */
    void estimateX(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats);

    /**
     * Estimates x and computes a score for the given UBM statistics
     */
    void forward(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats, double& score);
    void forward(const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& samples, blitz::Array<double,1>& scores);


  private:
    /**
     * Resizes the arrays (both members and cache)
     */ 
    void resize();
    /**
     * Resizes the arrays in cache
     */ 
    void resizeCache();
    /**
     * Put the mean and variance supervectors of the UBM in cache
     */
    void cacheSupervectors();
    /**
     * Computes U^T.Sigma^-1
     */
    void computeUtSigmaInv();
    /**
     * Computes (Id + U^T.Sigma^-1.U.N_{i,h}.U)^-1 = (Id + sum_{c=1..C} N_{i,h}.U_{c}^T.Sigma_{c}^-1.U_{c})^-1
     */
    void computeIdPlusUSProdInv(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats);
    /**
     * Computes Fn_x = sum_{sessions h}(N*(o - m) (Normalised first order statistics)
     */
    void computeFn_x(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats);
    /**
     * Estimates the value of x from the cache (Fn_x, U^T.Sigma^-1, etc.)
     */
    void updateX_fromCache();


    /**
     * the JFABaseMachine which contains the matrices U, V and D
     */
    boost::shared_ptr<bob::machine::JFABaseMachine> m_jfa_base;

    /**
      * y and z vectors/factors learned during the enrolment procedure
      */
    blitz::Array<double,1> m_y;
    blitz::Array<double,1> m_z;

    /**
      * x vectors/factors estimated when forwarding data
      * y and z vectors/factors used to estimate x when processing data
      */
    blitz::Array<double,1> m_y_for_x;
    blitz::Array<double,1> m_z_for_x;
    blitz::Array<double,1> m_x;

    /**
      * data cached used to improve performance
      */
    mutable blitz::Array<double,1> m_cache_Ux;
    mutable blitz::Array<double,1> m_cache_mVyDz;

    mutable blitz::Array<double,1> m_cache_mean;
    mutable blitz::Array<double,1> m_cache_sigma;
    mutable blitz::Array<double,2> m_cache_UtSigmaInv;
    mutable blitz::Array<double,2> m_cache_IdPlusUSProdInv;
    mutable blitz::Array<double,1> m_cache_Fn_x;

    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,2> m_tmp_ruru;
    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_ruCD;
};


}}

#endif
