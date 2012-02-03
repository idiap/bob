#include "daq/OutputWriter.h"

namespace bob { namespace daq {

OutputWriter::OutputWriter() {
  dir = std::string(".");
  name = std::string("output");
}

OutputWriter::~OutputWriter() {

}

void OutputWriter::setOutputDir(std::string dir) {
  this->dir = dir;
}

void OutputWriter::setOutputName(std::string name) {
  this->name = name;
}

}}