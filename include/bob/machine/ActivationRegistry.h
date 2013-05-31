/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 17 May 2013 14:34:17 CEST
 *
 * @brief A registration system for new activation routines
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

#ifndef BOB_MACHINE_ACTIVATIONREGISTRY_H
#define BOB_MACHINE_ACTIVATIONREGISTRY_H

#include <map>
#include <string>
#include <boost/shared_ptr.hpp>

#include "bob/machine/Activation.h"

namespace bob { namespace machine {

  /**
   * The ActivationRegistry holds registered loaders for different types of
   * Activation functions. 
   */
  class ActivationRegistry {

    public: //static access
      
      /**
       * Returns the singleton
       */
      static boost::shared_ptr<ActivationRegistry> instance();

      static const std::map<std::string, activation_factory_t>& getFactories ()
      {
        boost::shared_ptr<ActivationRegistry> ptr = instance();
        return ptr->s_id2factory;
      }
 
    public: //object access

      void registerActivation(const std::string& unique_identifier,
          activation_factory_t factory);

      void deregisterFactory(const std::string& unique_identifier);

      activation_factory_t find(const std::string& unique_identifier);

      bool isRegistered(const std::string& unique_identifier);

    private:

      ActivationRegistry (): s_id2factory() {}

      // Not implemented
      ActivationRegistry (const ActivationRegistry&);

      std::map<std::string, activation_factory_t> s_id2factory;
    
  };

}}

#endif /* BOB_MACHINE_ACTIVATIONREGISTRY_H */

