/**
 * @file cxx/machine/machine/JFAMachine.h
 * @date Sat Jul 23 21:41:15 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Machines that implements the Joint Factor Analysis model
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef TORCH_MACHINE_JFAMACHINE_H
#define TORCH_MACHINE_JFAMACHINE_H

#include <boost/shared_ptr.hpp>

#include "io/Arrayset.h"
#include "io/HDF5File.h"
#include "machine/GMMMachine.h"
#include "machine/LinearScoring.h"

namespace Torch { namespace machine {
  
  /**
   * A JFA Base machine which contains U, V and D matrices
   */
  class JFABaseMachine {

    public:

      /**
       * Default constructor. Builds an otherwise invalid 0 x 0 jfa machine.
       * This is equivalent to construct a JFA with five int parameters set 
       * to 0.
       */
      JFABaseMachine();

      /**
       * Constructor, builds a new jfa machine. UBM, U, V and d are
       * not initialized.
       *
       * @param C Number of gaussian components
       * @param D Dimensionality of the feature vector
       * @param ru size of U (CD x ru)
       * @param rv size of U (CD x rv)
       */ 
      JFABaseMachine(const boost::shared_ptr<Torch::machine::GMMMachine> ubm, int ru, int rv);

      /**
       * Copies another machine
       */
      JFABaseMachine(const JFABaseMachine& other);

      /**
       * Starts a new JFAMachine from an existing Configuration object.
       */
      JFABaseMachine(Torch::io::HDF5File& config);

      /**
       * Just to virtualise the destructor
       */
      virtual ~JFABaseMachine(); 

      /**
       * Assigns from a different machine
       */
      JFABaseMachine& operator= (const JFABaseMachine &other);

      /**
       * Loads data from an existing configuration object. Resets the current
       * state.
       */
      void load(Torch::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save(Torch::io::HDF5File& config) const;

      /**
        * Get the UBM
        */
      const boost::shared_ptr<Torch::machine::GMMMachine> getUbm() const 
      { return m_ubm; }

      /**
        * Get the U matrix
        */
      const blitz::Array<double,2>& getU() const 
      { return m_U; }

      /**
        * Get the V matrix
        */
      const blitz::Array<double,2>& getV() const 
      { return m_V; }

      /**
        * Get the d diagonal matrix/vector
        */
      const blitz::Array<double,1>& getD() const 
      { return m_d; }

      /**
        * Get the number of Gaussian components
        */
      const int getDimC() const 
      { return m_ubm->getNGaussians(); }

      /**
        * Get the feature dimensionality
        */
      const int getDimD() const 
      { return m_ubm->getNInputs(); }

      /**
        * Get the supervector length
        */
      const int getDimCD() const 
      { return m_ubm->getNInputs()*m_ubm->getNGaussians(); }

      /**
        * Get the size/rank ru of the U matrix
        */
      const int getDimRu() const 
      { return m_ru; }

      /**
        * Get the size/rank rv of the V matrix
        */
      const int getDimRv() const 
      { return m_rv; }


      /**
        * Get the U matrix in order to update it
        */
      blitz::Array<double,2>& updateU() 
      { return m_U; }

      /**
        * Get the V matrix in order to update it
        */
      blitz::Array<double,2>& updateV() 
      { return m_V; }

      /**
        * Get the d diagonal matrix/vector in order to update it
        */
      blitz::Array<double,1>& updateD() 
      { return m_d; }


      /**
        * Set the mean supervector of the UBM
        */
      void setUbm(const boost::shared_ptr<Torch::machine::GMMMachine> ubm);

      /**
        * Set the U matrix
        */
      void setU(const blitz::Array<double,2>& U);

      /**
        * Set the V matrix
        */
      void setV(const blitz::Array<double,2>& V);

      /**
        * Set the d diagonal matrix/vector
        */
      void setD(const blitz::Array<double,1>& d);

    private:

      // UBM
      boost::shared_ptr<Torch::machine::GMMMachine> m_ubm;

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
   * A JFA machine which contains y and z vectors, and can be used to process
   * data.
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
      JFAMachine(const boost::shared_ptr<Torch::machine::JFABaseMachine> jfabase);

      /**
       * Copies another machine
       */
      JFAMachine(const JFAMachine& other);

      /**
       * Starts a new JFAMachine from an existing Configuration object.
       */
      JFAMachine(Torch::io::HDF5File& config);

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
      void load(Torch::io::HDF5File& config);

      /**
       * Saves an existing machine to a Configuration object.
       */
      void save(Torch::io::HDF5File& config) const;

      /**
        * Get the JFABaseMachine
        */
      const boost::shared_ptr<Torch::machine::JFABaseMachine> getJFABase() const 
      { return m_jfa_base; }

      /**
        * Get the number of Gaussian components
        */
      const int getDimC() const 
      { return m_jfa_base->getUbm()->getNGaussians(); }

      /**
        * Get the feature dimensionality
        */
      const int getDimD() const 
      { return m_jfa_base->getUbm()->getNInputs(); }

      /**
        * Get the supervector length
        */
      const int getDimCD() const 
      { return m_jfa_base->getUbm()->getNInputs()*m_jfa_base->getUbm()->getNGaussians(); }

      /**
        * Get the size/rank ru of the U matrix
        */
      const int getDimRu() const 
      { return m_jfa_base->getDimRu(); }

      /**
        * Get the size/rank rv of the V matrix
        */
      const int getDimRv() const 
      { return m_jfa_base->getDimRv(); }

      /**
        * Get the y speaker factors
        */
      const blitz::Array<double,1>& getY() const 
      { return m_y; }

      /**
        * Get the z speaker factors
        */
      const blitz::Array<double,1>& getZ() const 
      { return m_z; }


      /**
        * Get the y speaker factors in order to update it
        */
      blitz::Array<double,1>& updateY() 
      { return m_y; }

      /**
        * Get the z speaker factors in order to update it
        */
      blitz::Array<double,1>& updateZ()
      { return m_z; }

      /**
        * Set the y speaker factors
        */
      void setY(const blitz::Array<double,1>& y);

      /**
        * Set the V matrix
        */
      void setZ(const blitz::Array<double,1>& z);

      /**
        * Set the JFABaseMachine
        */
      void setJFABase(const boost::shared_ptr<Torch::machine::JFABaseMachine> jfa_base);


      /**
        * Estimates from a 2D blitz::Array
        */
      void forward(const Torch::machine::GMMStats* gmm_stats, double& score);
      void forward(const std::vector<const Torch::machine::GMMStats*>& samples, blitz::Array<double,1>& scores);
      void estimateX(const Torch::machine::GMMStats* gmm_stats);
      void updateX(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F);
      void computeUtSigmaInv();
      void computeUProd();
      void computeIdPlusUProd(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F);
      void computeFn_x(const blitz::Array<double,1>& N, const blitz::Array<double,1>& F);
      void updateX_fromCache();

    private:

      boost::shared_ptr<Torch::machine::JFABaseMachine> m_jfa_base;

      /**
        * y and z vectors learned during the enrolment procedure
        */
      blitz::Array<double,1> m_y;
      blitz::Array<double,1> m_z;

      /**
        * x vectors estimated when forwarding data
        * y and z vectors used to estimate x when processing data
        */
      blitz::Array<double,1> m_y_for_x;
      blitz::Array<double,1> m_z_for_x;
      blitz::Array<double,1> m_x;

      /**
        * data cached used to improve performance
        */
      mutable Torch::machine::GMMStats m_cache_gmmstats;
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
