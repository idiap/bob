/**
  * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
  * @date Fri 8 Jul 2011 11:25:10
  *
  * A machine that combines the posterior scores of a client and 
  * a UBM models and generates log likelihood ratios.
  */
 
#ifndef TORCH_MACHINE_GMMLLRMACHINE_H
#define TORCH_MACHINE_GMMLLRMACHINE_H

#include "io/Arrayset.h"
#include "machine/Machine.h"
#include "machine/GMMMachine.h"
#include "machine/GMMLLRMachine.h"
#include "io/HDF5File.h"
#include <iostream>

namespace Torch { namespace machine {


    
  class GMMLLRMachine : public Machine<blitz::Array<double,1>, double> {
    public:

      /// Constructor from a Configuration
      GMMLLRMachine(Torch::io::HDF5File& config);
      GMMLLRMachine(Torch::io::HDF5File& client, Torch::io::HDF5File& ubm);

      /// Constructor from two GMMMachines
      GMMLLRMachine(const Torch::machine::GMMMachine& client, const Torch::machine::GMMMachine& ubm);

      /// Copy constructor
      /// (Needed because the GMM points to its constituent Gaussian members)
      GMMLLRMachine(const GMMLLRMachine& other);

      /// Assigment
      GMMLLRMachine & operator= (const GMMLLRMachine &other);

      /// Equal to
      bool operator ==(const GMMLLRMachine& b) const;
      
      /// Destructor
      virtual ~GMMLLRMachine(); 

      /// Get number of inputs
      int getNInputs() const;

      /// Output the log likelihood ratio for the sample input
      void forward(const blitz::Array<double,1>& input, double& output) const;

      /// Get a pointer to the client or UBM GMMMachine
      GMMMachine* getGMMClient() const;
      GMMMachine* getGMMUBM() const;

        /// Save to a Configuration
      void save(Torch::io::HDF5File& config) const;
      
      /// Load from a Configuration
      void load(Torch::io::HDF5File& config);
      
      friend std::ostream& operator<<(std::ostream& os, const GMMLLRMachine& machine);
      
    protected:

      /// Copy another GMMLLRMachine
      void copy(const GMMLLRMachine&);
      
      /// The feature dimensionality
      int64_t m_n_inputs;

      /// The GMM models for the client and the UBM
      GMMMachine *m_gmm_client;
      GMMMachine *m_gmm_ubm;
  };

}}

#endif /* TORCH_MACHINE_GMMLLRMACHINE_H */
