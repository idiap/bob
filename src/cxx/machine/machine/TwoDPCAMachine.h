/**
  * @file src/cxx/machine/machine/TwoDPCAMachine.h
  * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
  *
  * @brief This class defines a 2DPCA machine.
  */

#ifndef TORCH5SPRO_MACHINE_TWODPCAMACHINE_H
#define TORCH5SPRO_MACHINE_TWODPCAMACHINE_H

#include <blitz/array.h>
#include "machine/Machine.h"

namespace Torch {
/**
  * \ingroup libmachine_api
  * @{
  *
  */
  namespace machine {
  
  
    /**
      * @brief Class which implements subspace projection.
      */
    class TwoDPCAMachine : public Machine<blitz::Array<double,2>, blitz::Array<double,2> > {
      public:
        /**
         * Default constructor
         */
        TwoDPCAMachine();

        /**
          * Constructors
          */
        TwoDPCAMachine(int dim_outputs);
        TwoDPCAMachine(int dim_outputs, const blitz::Array<double,1>& eigenvalues,
          const blitz::Array<double,2>& eigenvectors);
        TwoDPCAMachine(int dim_outputs, const blitz::Array<double,1>& eigenvalues, 
          const blitz::Array<double,2>& eigenvectors, int n_outputs);
        TwoDPCAMachine(int dim_outputs, const blitz::Array<double,1>& eigenvalues, 
          const blitz::Array<double,2>& eigenvectors, double p_variance);

        /**
         * Copy constructor
         */
        TwoDPCAMachine(const TwoDPCAMachine& other);

        /**
         * Assigment
         */
        TwoDPCAMachine & operator= (const TwoDPCAMachine &other);

        /**
         * Destructor
         */
        virtual ~TwoDPCAMachine(); 

        /// Set the feature dimensionality
        /// Overrides Machine::setNInputs
        void setDimOutputs(int dim_outputs);
        void setNOutputs(int n_outputs);
        void setPVariance(double p_variance);

        /// Get number of inputs
        int getDimOutputs() const;
        int getNOutputs() const;
        double getPVariance() const;

        void setEigenvaluesvectors(const blitz::Array<double,1>& eigenvalues,
          const blitz::Array<double,2>& eigenvectors);

        const blitz::Array<double,1>& getEigenvalues() const;
        const blitz::Array<double,2>& getEigenvectors() const;

        /// Remove means before the projection (in particular for PCA)
        /// Should be set after the Eigenvalues/Eigenvectors
        void setPreMean(const blitz::Array<double,2>& pre_mean);

        const blitz::Array<double,2>& getPreMean() const;

        /// Output the projected sample, x 
        /// (overrides Machine::forward)
        void forward(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const;

        /// Print the parameters of the GMM
        void print() const;

      protected:
        /// Copy another TwoDPCAMachine
        void copy(const TwoDPCAMachine&);

        // The dimensionality of the outputs (number of rows of the inputs)
        int m_dim_outputs;
        // The percentage of variance to keep
        double m_p_variance;
        // The number of output to keep
        int m_n_outputs;

        // The eigenvalues and eigenvectors
        blitz::Array<double,1> m_eigenvalues;
        blitz::Array<double,2> m_eigenvectors;

        // Mean to be removed before the projection
        blitz::Array<double,2> m_pre_mean;
    };

  }
}

#endif
