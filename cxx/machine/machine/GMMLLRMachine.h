/**
 * @file cxx/machine/machine/GMMLLRMachine.h
 * @date Fri Jul 8 13:01:03 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
  * A machine that combines the posterior scores of a client and
  * a UBM models and generates log likelihood ratios.
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
 
#ifndef BOB_MACHINE_GMMLLRMACHINE_H
#define BOB_MACHINE_GMMLLRMACHINE_H

#include "io/Arrayset.h"
#include "machine/Machine.h"
#include "machine/GMMMachine.h"
#include "machine/GMMLLRMachine.h"
#include "io/HDF5File.h"
#include <iostream>

namespace bob { namespace machine {


    
  class GMMLLRMachine : public Machine<blitz::Array<double,1>, double> {
    public:

      /// Constructor from a Configuration
      GMMLLRMachine(bob::io::HDF5File& config);
      GMMLLRMachine(bob::io::HDF5File& client, bob::io::HDF5File& ubm);

      /// Constructor from two GMMMachines
      GMMLLRMachine(const bob::machine::GMMMachine& client, const bob::machine::GMMMachine& ubm);

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
      void forward_(const blitz::Array<double,1>& input, double& output) const;

      /// Get a pointer to the client or UBM GMMMachine
      GMMMachine* getGMMClient() const;
      GMMMachine* getGMMUBM() const;

        /// Save to a Configuration
      void save(bob::io::HDF5File& config) const;
      
      /// Load from a Configuration
      void load(bob::io::HDF5File& config);
      
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

#endif /* BOB_MACHINE_GMMLLRMACHINE_H */
