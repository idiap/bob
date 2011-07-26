/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Wed 20 Jul 2011 19:20:50
 *
 * A machine that implements the Joint Factor Analysis model
 */

#ifndef TORCH_MACHINE_JFAMACHINE_H
#define TORCH_MACHINE_JFAMACHINE_H

#include "io/Arrayset.h"
#include "io/HDF5File.h"
#include "machine/GMMMachine.h"
#include <boost/shared_ptr.hpp>

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

      boost::shared_ptr<Torch::machine::GMMMachine> m_ubm;

      // dimensionality
      int m_ru; // size of U (CD x ru)
      int m_rv; // size of V (CD x rv)

      blitz::Array<double,2> m_U;
      blitz::Array<double,2> m_V;
      blitz::Array<double,1> m_d;
  };


  /**
   * A JFA Base machine which contains U, V and D matrices
   */
  class JFAMachine {

    public:

      /**
       * Default constructor. Builds an otherwise invalid 0 x 0 jfa machine.
       * This is equivalent to construct a JFA with five int parameters set 
       * to 0.
       */
      JFAMachine();

      /**
       * Constructor, builds a new jfa machine. UBM, U, V and d are
       * not initialized.
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

    private:

      boost::shared_ptr<Torch::machine::JFABaseMachine> m_jfa_base;

      blitz::Array<double,1> m_y;
      blitz::Array<double,1> m_z;
  };


}}

#endif
