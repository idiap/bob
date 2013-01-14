/**
 * @file bob/visioner/util/threads.h
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Mon 06 Aug 2012 14:29:59 CEST
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
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

#ifndef BOB_VISIONER_UTIL_THREADS_H 
#define BOB_VISIONER_UTIL_THREADS_H

#include <vector>

#include <boost/thread.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/shared_array.hpp>

namespace bob { namespace visioner {

  // Split some objects to process using multiple threads
  void thread_split(uint64_t n_objects, std::vector<uint64_t>& sbegins, 
      std::vector<uint64_t>& sends, size_t num_of_threads);

  // Split a loop computation of the given size using multiple threads
  // NB: Stateless threads: op(<begin, end>)
  template <typename TOp> void thread_loop(TOp op, uint64_t size,
      size_t num_of_threads=boost::thread::hardware_concurrency()) {
    
    boost::shared_array<boost::thread> threads(new boost::thread[num_of_threads]);
    std::vector<uint64_t> th_begins; th_begins.reserve(num_of_threads);
    std::vector<uint64_t> th_ends; th_ends.reserve(num_of_threads);

    thread_split(size, th_begins, th_ends, num_of_threads);		

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      std::pair<uint64_t, uint64_t> range(th_begins[ith], th_ends[ith]);
      boost::thread t(boost::bind(op, range));
      threads[ith] = boost::move(t);
    }

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      threads[ith].join();
    }

  }

  // Split a loop computation of the given size using multiple threads
  // NB: Stateless threads: op(thread_index, <begin, end>)
  template <typename TOp> void thread_iloop(TOp op, uint64_t size,
      size_t num_of_threads=boost::thread::hardware_concurrency()) {

    boost::shared_array<boost::thread> threads(new boost::thread[num_of_threads]);
    std::vector<uint64_t> th_begins; th_begins.reserve(num_of_threads);
    std::vector<uint64_t> th_ends; th_ends.reserve(num_of_threads);

    thread_split(size, th_begins, th_ends, num_of_threads);		

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      std::pair<uint64_t, uint64_t> range(th_begins[ith], th_ends[ith]);
      boost::thread t(boost::bind(op, ith, range));
      threads[ith] = boost::move(t);
    }

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      threads[ith].join();
    }

  }

  // Split a loop computation of the given size using multiple threads
  // NB: State threads: op(<begin, end>, result&)
  template <typename TOp, typename TResult> void thread_loop(TOp op, uint64_t size, std::vector<TResult>& results, size_t num_of_threads=boost::thread::hardware_concurrency()) {

    boost::shared_array<boost::thread> threads(new boost::thread[num_of_threads]);
    std::vector<uint64_t> th_begins; th_begins.reserve(num_of_threads);
    std::vector<uint64_t> th_ends; th_ends.reserve(num_of_threads);

    thread_split(size, th_begins, th_ends, num_of_threads);		

    results.resize(num_of_threads);

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      std::pair<uint64_t, uint64_t> range(th_begins[ith], th_ends[ith]);
      boost::thread t(boost::bind(op, range, boost::ref(results[ith])));
      threads[ith] = boost::move(t);
    }

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      threads[ith].join();
    }

  }

  // Split a loop computation of the given size using multiple threads
  // NB: State threads: op(thread_index, <begin, end>, result&)
  template <typename TOp, typename TResult> void thread_iloop(TOp op, uint64_t size, std::vector<TResult>& results, size_t num_of_threads=boost::thread::hardware_concurrency()) {

    boost::shared_array<boost::thread> threads(new boost::thread[num_of_threads]);
    std::vector<uint64_t> th_begins; th_begins.reserve(num_of_threads);
    std::vector<uint64_t> th_ends; th_ends.reserve(num_of_threads);

    thread_split(size, th_begins, th_ends, num_of_threads);		

    results.resize(num_of_threads);

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      std::pair<uint64_t, uint64_t> range(th_begins[ith], th_ends[ith]);
      boost::thread t(boost::bind(op, ith, range, boost::ref(results[ith])));
      threads[ith] = boost::move(t);
    }

    for (uint64_t ith = 0; ith < num_of_threads; ith ++) {
      threads[ith].join();
    }

  }

}}

#endif /* BOB_VISIONER_UTIL_THREADS_H */
