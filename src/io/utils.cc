/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Wed  3 Oct 08:36:48 2012
 *
 * @brief Implementation of some compile-time I/O utitlites
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/io/utils.h>
#include <bob/io/CodecRegistry.h>

boost::shared_ptr<bob::io::File> bob::io::open (const char* filename,
    char mode, const char* pretend_extension) {
  boost::shared_ptr<bob::io::CodecRegistry> instance = bob::io::CodecRegistry::instance();
  return instance->findByExtension(pretend_extension)(filename, mode);
}

boost::shared_ptr<bob::io::File> bob::io::open (const char* filename, char mode) {
  boost::shared_ptr<bob::io::CodecRegistry> instance = bob::io::CodecRegistry::instance();
  return instance->findByFilenameExtension(filename)(filename, mode);
}

bob::core::array::typeinfo bob::io::peek (const char* filename) {
  return open(filename, 'r')->type();
}

bob::core::array::typeinfo bob::io::peek_all (const char* filename) {
  return open(filename, 'r')->type_all();
}
