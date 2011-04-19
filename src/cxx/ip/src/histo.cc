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
