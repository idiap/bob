/**
 * @file cxx/trainer/src/Exception.cc
 * @date Wed May 18 16:14:44 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Implementation of trainer exceptions
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

#include "trainer/Exception.h"
#include <boost/format.hpp>

namespace train = Torch::trainer;

train::Exception::Exception() throw() {
}

train::Exception::~Exception() throw() {
}

const char* train::Exception::Exception::what() const throw() {
 static const char* what_string = "Generic trainer::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}


train::NoPriorGMM::NoPriorGMM() throw() {
}

train::NoPriorGMM::~NoPriorGMM() throw() {
}

const char* train::NoPriorGMM::what() const throw() {
  return "MAP_GMMTrainer: Prior GMM has not been set";
}

train::WrongNumberOfClasses::WrongNumberOfClasses(size_t got) throw() :
  m_got(got)
{
}

train::WrongNumberOfClasses::~WrongNumberOfClasses() throw() { }

const char* train::WrongNumberOfClasses::what() const throw() {
  try {
    boost::format message("Cannot operate with '%u' classes");
    message % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "trainer::WrongNumberOfClasses: cannot format, exception raised";
    return emergency;
  }
}

train::WrongNumberOfFeatures::WrongNumberOfFeatures(size_t got,
    size_t expected, size_t classid) throw() :
  m_got(got),
  m_expected(expected),
  m_classid(classid)
{
}

train::WrongNumberOfFeatures::~WrongNumberOfFeatures() throw() { }

const char* train::WrongNumberOfFeatures::what() const throw() {
  try {
    boost::format message("Number of features for class '%u' is not compatible with the remaining classes. Class '%u' has '%u' features and the previous classes have '%u' features.");
    message % m_classid % m_classid % m_got % m_expected;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "trainer::WrongNumberOfFeatures: cannot format, exception raised";
    return emergency;
  }
}

train::IncompatibleMachine::IncompatibleMachine() throw() {
}

train::IncompatibleMachine::~IncompatibleMachine() throw() {
}

const char* train::IncompatibleMachine::IncompatibleMachine::what() const throw() {
 static const char* what_string = "trainer::IncompatibleMachine: you are trying to provide a machine this trained was not tailored for";
 return what_string;
}

train::EmptyTrainingSet::EmptyTrainingSet() throw() {
}

train::EmptyTrainingSet::~EmptyTrainingSet() throw() {
}

const char* train::EmptyTrainingSet::EmptyTrainingSet::what() const throw() {
 static const char* what_string = "trainer::EmptyTrainingSet: you are trying to train a machine without any data";
 return what_string;
}
