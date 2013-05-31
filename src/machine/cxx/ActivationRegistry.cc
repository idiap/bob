/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 17 May 2013 16:02:05 CEST
 *
 * @brief Implementation of the activation registry
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

#include "bob/machine/ActivationRegistry.h"

boost::shared_ptr<bob::machine::ActivationRegistry> bob::machine::ActivationRegistry::instance() {
  static boost::shared_ptr<bob::machine::ActivationRegistry> s_instance(new ActivationRegistry());
  return s_instance;
}
    
void bob::machine::ActivationRegistry::deregisterFactory(const std::string& id) {
  s_id2factory.erase(id);
}

void bob::machine::ActivationRegistry::registerActivation(const std::string& id,
    bob::machine::activation_factory_t factory) {

  auto it = s_id2factory.find(id);

  if (it == s_id2factory.end()) {
    s_id2factory[id] = factory;
  }
  else {
    boost::format m("factory for activation function %s is being registered twice");
    throw std::runtime_error(m.str());
  }

}

bool bob::machine::ActivationRegistry::isRegistered(const std::string& id) {
  return (s_id2factory.find(id) != s_id2factory.end());
}

bob::machine::activation_factory_t bob::machine::ActivationRegistry::find
(const std::string& id) {

  auto it = s_id2factory.find(id);

  if (it == s_id2factory.end()) {
    boost::format m("unregistered activation function: %s");
    m % id;
    throw std::runtime_error(m.str());
  }

  return it->second;

}
