/**
 * @file visioner/visioner/util/util.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
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

#ifndef BOB_VISIONER_UTIL_H
#define BOB_VISIONER_UTIL_H

#include <string>
#include <cmath>
#include <algorithm>

#include <boost/lexical_cast.hpp>
#include <boost/thread.hpp>
#include <boost/lambda/bind.hpp>
#include <boost/shared_array.hpp>

#include "visioner/util/matrix.h"

namespace bob { namespace visioner {

  /**
   * Utility functions
   */

  // Force a value in a given range
  template <typename T, typename TMin, typename TMax>
    T range(T value, TMin min_value, TMax max_value)
    {
      return value < (T)min_value ? (T)min_value : (value > (T)max_value ? (T)max_value : value);
    }

  // Force a value in a given range
  template <typename TIt, typename TMin, typename TMax>
    void range(TIt begin, TIt end, TMin min_value, TMax max_value)
    {
      for (TIt it = begin; it != end; ++ it)
      {
        *it = range(*it, min_value, max_value);
      }
    }

  // Safely compute the inverse of some value
  template <typename T>
    double inverse(T value)
    {
      static const T zero = (T)0;
      return value == zero ? 1.0 : 1.0 / (0.0 + value);
    }	

  // Math expensive routines
  inline double my_exp(double x) { return std::exp(x); }
  inline double my_log(double x) { return std::log(x); }
  inline double my_atan2(double y, double x) { return std::atan2(y, x); }
  inline double my_sqrt(double x) { return std::sqrt(x); }
  inline double my_abs(double x) { return std::abs(x); }

  // Remove duplicates from a given vector
  template <typename T>
    void unique(std::vector<T>& data)
    {
      std::sort(data.begin(), data.end());
      data.erase(std::unique(data.begin(), data.end()), data.end());
    }

  // Bitwise functors
  template <typename Arg1, typename Arg2, typename Result>
    struct bitwise_and : public std::binary_function<Arg1, Arg2, Result>
  {
    Result operator()(Arg1 arg1, Arg2 arg2) const
    {
      return arg1 & arg2;
    }
  };

  template <typename Arg1, typename Arg2, typename Result>
    struct bitwise_or : public std::binary_function<Arg1, Arg2, Result>
  {
    Result operator()(Arg1 arg1, Arg2 arg2) const
    {
      return arg1 | arg2;
    }
  };

  template <typename Arg1, typename Arg2, typename Result>
    struct bitwise_xor : public std::binary_function<Arg1, Arg2, Result>
  {
    Result operator()(Arg1 arg1, Arg2 arg2) const
    {
      return arg1 ^ arg2;
    }
  };

  // Total number of threads to use
  inline uint64_t n_threads()
  {
    return boost::thread::hardware_concurrency();
  }

  // Split some objects to process using multiple threads
  void thread_split(uint64_t n_objects, uint64_t* sbegins, uint64_t* sends);

  // Split a loop computation of the given size using multiple threads
  // NB: Stateless threads: op(<begin, end>)
  template <typename TOp>
    void thread_loop(TOp op, uint64_t size)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<uint64_t> th_begins(new uint64_t[n_threads()]);
      boost::shared_array<uint64_t> th_ends(new uint64_t[n_threads()]);

      thread_split(size, th_begins.get(), th_ends.get());		
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, std::pair<uint64_t, uint64_t>(th_begins[ith], th_ends[ith])));
      }
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  // Split a loop computation of the given size using multiple threads
  // NB: Stateless threads: op(thread_index, <begin, end>)
  template <typename TOp>
    void thread_iloop(TOp op, uint64_t size)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<uint64_t> th_begins(new uint64_t[n_threads()]);
      boost::shared_array<uint64_t> th_ends(new uint64_t[n_threads()]);

      thread_split(size, th_begins.get(), th_ends.get());		
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, ith, std::pair<uint64_t, uint64_t>(th_begins[ith], th_ends[ith])));
      }
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  // Split a loop computation of the given size using multiple threads
  // NB: State threads: op(<begin, end>, result&)
  template <typename TOp, typename TResult>
    void thread_loop(TOp op, uint64_t size, std::vector<TResult>& results)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<uint64_t> th_begins(new uint64_t[n_threads()]);
      boost::shared_array<uint64_t> th_ends(new uint64_t[n_threads()]);

      results.resize(n_threads());

      thread_split(size, th_begins.get(), th_ends.get());
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, std::pair<uint64_t, uint64_t>(th_begins[ith], th_ends[ith]), 
              boost::ref(results[ith])));
      }
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  // Split a loop computation of the given size using multiple threads
  // NB: State threads: op(thread_index, <begin, end>, result&)
  template <typename TOp, typename TResult>
    void thread_iloop(TOp op, uint64_t size, std::vector<TResult>& results)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<uint64_t> th_begins(new uint64_t[n_threads()]);
      boost::shared_array<uint64_t> th_ends(new uint64_t[n_threads()]);

      results.resize(n_threads());

      thread_split(size, th_begins.get(), th_ends.get());
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, ith, std::pair<uint64_t, uint64_t>(th_begins[ith], th_ends[ith]), 
              boost::ref(results[ith])));
      }
      for (uint64_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  /**
   * String processing
   */

  // Split a string given some separators
  std::vector<std::string> split(const std::string& str, const char* delim_chars = " \n\t\r");

  // Resize the string to obtain the given size
  //  (   if larger - characters will be removed from the end, 
  //      if smaller - <app_char> will be inserted at the end)        
  std::string resize(const std::string& str, uint64_t size, char app_char = ' ');

  // Trim a float value to the given digits
  std::string round(double value, uint64_t digits);

  // Split a string to a list of values	
  template <typename T>
    std::vector<T> split2values(const std::string& value, const char* delim_chars = " \n\t\r")
    {
      std::vector<T> result;

      const std::vector<std::string> tokens = visioner::split(value, delim_chars);	
      for (std::vector<std::string>::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
      {
        result.push_back(boost::lexical_cast<T>(it->c_str()));
      }

      return result;
    }

  /**
   * File processing
   */

  // Parse an .ini stream (lines with: "attribute = values [#xxx]", ignoring "#xxx" ones)
  bool load_initext(const std::string& text, std::vector<std::string>& attributes, std::vector<std::string>& values);
  bool load_inifile(const std::string& path, std::vector<std::string>& attributes, std::vector<std::string>& values);

  // Retrieves the value associated to some attribute
  // If the attribute is not found, the program exits
  std::string attribute_value(const std::vector<std::string>& attributes, const std::vector<std::string>& values, const char* attribute);

  // Parse a (sef of) list file(s)
  bool load_listfiles(const std::string& files, std::vector<std::string>& ifiles, std::vector<std::string>& gfiles);

  // Load the content of a text file into a string
  bool load_file(const std::string& path, std::string& text);

  // Extract the name of the file, with (filename) and without (basename) extension, from a path
  std::string filename(const std::string& path);
  std::string basename(const std::string& path);
  std::string extname(const std::string& path);

  // Extract the name of the directory from a path
  std::string dirname(const std::string& path);	

}}

#endif // BOB_VISIONER_UTIL_H
