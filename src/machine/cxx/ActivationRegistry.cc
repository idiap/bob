/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Fri 17 May 2013 16:02:05 CEST
 *
 * @brief Implementation of the activation registry
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
    m % id;
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
