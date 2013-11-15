/**
 * @file visioner/programs/max_threads.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <iostream>
#include <boost/thread.hpp>

int main(int, char* []) {	
  std::cout << boost::thread::hardware_concurrency() << "\n";
  return EXIT_SUCCESS;
}
