/**
 * @file cxx/ip/src/histo.cc
 * @date Mon Apr 18 16:08:34 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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
#include "ip/histo.h"

Torch::ip::UnsupportedTypeForHistogram::UnsupportedTypeForHistogram(tca::ElementType elementType)  throw(): elementType(elementType) {
  sprintf(description, "The source type \"%s\" is not supported", Torch::core::array::stringize(elementType));
}
Torch::ip::UnsupportedTypeForHistogram::UnsupportedTypeForHistogram(const UnsupportedTypeForHistogram& other) throw(): elementType(other.elementType) {
  sprintf(description, "The source type \"%s\" is not supported", Torch::core::array::stringize(elementType));
}

Torch::ip::UnsupportedTypeForHistogram::~UnsupportedTypeForHistogram() throw() {
  
}

const char* Torch::ip::UnsupportedTypeForHistogram::what() const throw() {
  return description;
}

Torch::ip::InvalidArgument::InvalidArgument()  throw() {
  
}

Torch::ip::InvalidArgument::InvalidArgument(const InvalidArgument& other) throw() {
  
}

Torch::ip::InvalidArgument::~InvalidArgument() throw() {
  
}

const char* Torch::ip::InvalidArgument::what() const throw() {
  return "Invalid argument";
}
