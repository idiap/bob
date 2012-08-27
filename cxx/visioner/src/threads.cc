/**
 * @file visioner/src/threads.cc
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 06 Aug 2012 14:31:27 CEST
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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

#include "visioner/util/threads.h"

// Split some objects to process using multiple threads
void bob::visioner::thread_split(uint64_t n_objects, 
    std::vector<uint64_t>& sbegins, std::vector<uint64_t>& sends, 
    size_t num_of_threads=boost::thread::hardware_concurrency()) {

  uint64_t n_objects_per_thread = n_objects / num_of_threads + 1;

  for (uint64_t ith = 0, sbegin = 0; ith < num_of_threads; ith ++) {
    sbegins.push_back(sbegin);
    sbegin = std::min(sbegin + n_objects_per_thread, n_objects);
    sends.push_back(sbegin);
  }

}
