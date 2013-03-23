/**
 * @file trainer/cxx/Exception.cc
 * @date Wed May 18 16:14:44 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Implementation of trainer exceptions
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

#include <bob/trainer/Exception.h>
#include <boost/format.hpp>

bob::trainer::Exception::Exception() throw() {
}

bob::trainer::Exception::~Exception() throw() {
}

const char* bob::trainer::Exception::what() const throw() {
 static const char* what_string = "Generic trainer::Exception: user specialization has not overwritten what() or is throwing an object of this class (in which case, please fix it!)";
 return what_string;
}


bob::trainer::NoPriorGMM::NoPriorGMM() throw() {
}

bob::trainer::NoPriorGMM::~NoPriorGMM() throw() {
}

const char* bob::trainer::NoPriorGMM::what() const throw() {
  return "MAP_GMMTrainer: Prior GMM has not been set";
}

bob::trainer::WrongNumberOfClasses::WrongNumberOfClasses(size_t got) throw() :
  m_got(got)
{
}

bob::trainer::WrongNumberOfClasses::~WrongNumberOfClasses() throw() { }

const char* bob::trainer::WrongNumberOfClasses::what() const throw() {
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

bob::trainer::WrongNumberOfFeatures::WrongNumberOfFeatures(size_t got,
    size_t expected, size_t classid) throw() :
  m_got(got),
  m_expected(expected),
  m_classid(classid)
{
}

bob::trainer::WrongNumberOfFeatures::~WrongNumberOfFeatures() throw() { }

const char* bob::trainer::WrongNumberOfFeatures::what() const throw() {
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

bob::trainer::IncompatibleMachine::IncompatibleMachine() throw() {
}

bob::trainer::IncompatibleMachine::~IncompatibleMachine() throw() {
}

const char* bob::trainer::IncompatibleMachine::what() const throw() {
 static const char* what_string = "trainer::IncompatibleMachine: you are trying to provide a machine this trained was not tailored for";
 return what_string;
}

bob::trainer::EmptyTrainingSet::EmptyTrainingSet() throw() {
}

bob::trainer::EmptyTrainingSet::~EmptyTrainingSet() throw() {
}

const char* bob::trainer::EmptyTrainingSet::what() const throw() {
 static const char* what_string = "trainer::EmptyTrainingSet: you are trying to train a machine without any data";
 return what_string;
}

bob::trainer::LLRPriorNotInRange::LLRPriorNotInRange(const double got) throw() :
  m_got(got)
{
}

bob::trainer::LLRPriorNotInRange::~LLRPriorNotInRange() throw() { }

const char* bob::trainer::LLRPriorNotInRange::what() const throw() {
  try {
    boost::format message("Cannot set prior to '%f', as it should be in the range ]0.,1.[.");
    message % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "bob::trainer::LLRPriorNotInRange: cannot format, exception raised";
    return emergency;
  }
}

bob::trainer::KMeansInitializationFailure::KMeansInitializationFailure() throw() {
}

bob::trainer::KMeansInitializationFailure::~KMeansInitializationFailure() throw() {
}

const char* bob::trainer::KMeansInitializationFailure::what() const throw() {
 static const char* what_string = "bob::trainer::KMeansInitializationFailure: this usually happens when many samples are identical, as the initial means should all be different.";
 return what_string;
}
