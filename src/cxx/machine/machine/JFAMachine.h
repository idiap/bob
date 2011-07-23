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

namespace Torch { namespace machine {
  
  /**
   * A JFA machine
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
       *
       * @param C Number of gaussian components
       * @param D Dimensionality of the feature vector
       * @param ru size of U (CD x ru)
       * @param rv size of U (CD x rv)
       */ 
      JFAMachine(int C, int D, int ru, int rv);

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
        * Get the mean supervector of the UBM
        */
      const blitz::Array<double,1>& getUbmMean() const 
      { return m_ubm_mean; }

      /**
        * Get the variance supervector of the UBM
        */
      const blitz::Array<double,1>& getUbmVar() const 
      { return m_ubm_var; }

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
      { return m_C; }

      /**
        * Get the feature dimensionality
        */
      const int getDimD() const 
      { return m_D; }

      /**
        * Get the supervector length
        */
      const int getDimCD() const 
      { return m_CD; }

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
        * Get the mean supervector of the UBM in order to update it
        */
      blitz::Array<double,1>& updateUbmMean()
      { return m_ubm_mean; }

      /**
        * Get the variance supervector of the UBM in order to update it
        */
      blitz::Array<double,1>& updateUbmVar() 
      { return m_ubm_var; }

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
      void setUbmMean(const blitz::Array<double,1>& mean);

      /**
        * Set the variance supervector of the UBM
        */
      void setUbmVar(const blitz::Array<double,1>& var);

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

      // dimensionality
      int m_C; // Number of Gaussian components
      int m_D; // Feature dimensionality
      int m_CD; // Product CD
      int m_ru; // size of U (CD x ru)
      int m_rv; // size of V (CD x rv)

      blitz::Array<double,1> m_ubm_mean;
      blitz::Array<double,1> m_ubm_var;
      blitz::Array<double,2> m_U;
      blitz::Array<double,2> m_V;
      blitz::Array<double,1> m_d;

      blitz::Array<double,2> m_X;
      blitz::Array<double,2> m_Y;
      blitz::Array<double,2> m_Z;
  };

}}

#endif
