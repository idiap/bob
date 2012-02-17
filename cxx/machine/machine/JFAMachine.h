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

#include "io/Arrayset.h"
#include "io/HDF5File.h"
#include "machine/GMMMachine.h"
#include "machine/JFAMachineException.h"
#include "machine/LinearScoring.h"

#include <boost/shared_ptr.hpp>

namespace bob { namespace machine {
  
/**
 * A JFA Base machine which contains U, V and D matrices
 * TODO: add a reference to the journal article
 */
class JFABaseMachine {

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
     * @param UBM The Universal Backgroud Model
     * @param ru size of U (CD x ru)
     * @param rv size of U (CD x rv)
     */ 
    JFABaseMachine(const boost::shared_ptr<bob::machine::GMMMachine> ubm, int ru, int rv);

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
    const int getDimC() const 
    { if(!m_ubm) throw bob::machine::JFAMachineNoUBM(); 
      return m_ubm->getNGaussians(); }

    /**
      * Returns the feature dimensionality D
      * @warning An exception is thrown if no Universal Background Model has 
      *   been set yet.
      */
    const int getDimD() const 
    { if(!m_ubm) throw bob::machine::JFAMachineNoUBM();
      return m_ubm->getNInputs(); }

    /**
      * Returns the supervector length CD
      * (CxD: Number of Gaussian components by the feature dimensionality)
      * @warning An exception is thrown if no Universal Background Model has 
      *   been set yet.
      */
    const int getDimCD() const 
    { if(!m_ubm) throw bob::machine::JFAMachineNoUBM(); 
      return m_ubm->getNInputs()*m_ubm->getNGaussians(); }

    /**
      * Returns the size/rank ru of the U matrix
      */
    const int getDimRu() const 
    { return m_ru; }

    /**
      * Returns the size/rank rv of the V matrix
      */
    const int getDimRv() const 
    { return m_rv; }


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
    int m_ru; // size of U (CD x ru)
    int m_rv; // size of V (CD x rv)

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
class JFAMachine {

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
    JFAMachine& operator= (const JFAMachine &other);

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
    const int getDimC() const 
    { return m_jfa_base->getUbm()->getNGaussians(); }

    /**
      * Returns the feature dimensionality D
      */
    const int getDimD() const 
    { return m_jfa_base->getUbm()->getNInputs(); }

    /**
      * Returns the supervector length CxD
      */
    const int getDimCD() const 
    { return m_jfa_base->getUbm()->getNInputs()*m_jfa_base->getUbm()->getNGaussians(); }

    /**
      * Returns the size/rank ru of the U matrix
      */
    const int getDimRu() const 
    { return m_jfa_base->getDimRu(); }

    /**
      * Returns the size/rank rv of the V matrix
      */
    const int getDimRv() const 
    { return m_jfa_base->getDimRv(); }

    /**
      * Returns the y speaker factors
      */
    const blitz::Array<double,1>& getY() const 
    { return m_y; }

    /**
      * Returns the z speaker factors
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
    void forward(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats, double& score);
    void forward(const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& samples, blitz::Array<double,1>& scores);
    void estimateX(boost::shared_ptr<const bob::machine::GMMStats> gmm_stats);
    void updateX(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F);
    void computeUtSigmaInv();
    void computeUProd();
    void computeIdPlusUProd(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F);
    void computeFn_x(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F);
    void updateX_fromCache();


  private:
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

    mutable blitz::Array<double,2> m_cache_UtSigmaInv;
    mutable blitz::Array<double,1> m_cache_sigma;
    mutable blitz::Array<double,1> m_cache_mean;
    mutable blitz::Array<double,3> m_cache_UProd;
    mutable blitz::Array<double,2> m_cache_IdPlusUProd;
    mutable blitz::Array<double,1> m_cache_Fn_x;

    mutable blitz::Array<double,2> m_tmp_ruD;
    mutable blitz::Array<double,2> m_tmp_ruru;
    mutable blitz::Array<double,1> m_tmp_ru;
    mutable blitz::Array<double,1> m_tmp_CD;
    mutable blitz::Array<double,1> m_tmp_CD_b;
};


}}

#endif
