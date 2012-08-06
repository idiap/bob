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

  typedef u_int64_t       		index_t;
  typedef std::vector<index_t>		indices_t;
  typedef Matrix<index_t>			index_mat_t;

  typedef std::pair<index_t, index_t>     index_pair_t;

  typedef std::string			string_t;
  typedef std::vector<string_t>		strings_t;

  typedef double				scalar_t;
  typedef std::vector<scalar_t>		scalars_t;
  typedef Matrix<scalar_t>		scalar_mat_t;
  typedef std::vector<scalar_mat_t>       scalar_mats_t;

  typedef float				fscalar_t;
  typedef std::vector<fscalar_t>		fscalars_t;
  typedef Matrix<fscalar_t>		fscalar_mat_t;
  typedef std::vector<fscalar_mat_t>      fscalar_mats_t;

  typedef u_int16_t     			discrete_t;
  typedef std::vector<discrete_t>		discretes_t;
  typedef Matrix<discrete_t>		discrete_mat_t;

  typedef int     			bool_t;
  typedef std::vector<bool_t>		bools_t;
  typedef Matrix<bool_t>			bool_mat_t;

  /////////////////////////////////////////////////////////////////////////////////////////
  // Utility functions
  /////////////////////////////////////////////////////////////////////////////////////////

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
    scalar_t inverse(T value)
    {
      static const T zero = (T)0;
      return value == zero ? 1.0 : 1.0 / (0.0 + value);
    }	

  // Math expensive routines
  inline scalar_t my_exp(scalar_t x) { return std::exp(x); }
  inline scalar_t my_log(scalar_t x) { return std::log(x); }
  inline scalar_t my_atan2(scalar_t y, scalar_t x) { return std::atan2(y, x); }
  inline scalar_t my_sqrt(scalar_t x) { return std::sqrt(x); }
  inline scalar_t my_abs(scalar_t x) { return std::abs(x); }

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
  inline index_t n_threads()
  {
    return boost::thread::hardware_concurrency();
  }

  // Split some objects to process using multiple threads
  void thread_split(index_t n_objects, index_t* sbegins, index_t* sends);

  // Split a loop computation of the given size using multiple threads
  // NB: Stateless threads: op(<begin, end>)
  template <typename TOp>
    void thread_loop(TOp op, index_t size)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<index_t> th_begins(new index_t[n_threads()]);
      boost::shared_array<index_t> th_ends(new index_t[n_threads()]);

      thread_split(size, th_begins.get(), th_ends.get());		
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, index_pair_t(th_begins[ith], th_ends[ith])));
      }
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  // Split a loop computation of the given size using multiple threads
  // NB: Stateless threads: op(thread_index, <begin, end>)
  template <typename TOp>
    void thread_iloop(TOp op, index_t size)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<index_t> th_begins(new index_t[n_threads()]);
      boost::shared_array<index_t> th_ends(new index_t[n_threads()]);

      thread_split(size, th_begins.get(), th_ends.get());		
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, ith, index_pair_t(th_begins[ith], th_ends[ith])));
      }
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  // Split a loop computation of the given size using multiple threads
  // NB: State threads: op(<begin, end>, result&)
  template <typename TOp, typename TResult>
    void thread_loop(TOp op, index_t size, std::vector<TResult>& results)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<index_t> th_begins(new index_t[n_threads()]);
      boost::shared_array<index_t> th_ends(new index_t[n_threads()]);

      results.resize(n_threads());

      thread_split(size, th_begins.get(), th_ends.get());
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, index_pair_t(th_begins[ith], th_ends[ith]), 
              boost::ref(results[ith])));
      }
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  // Split a loop computation of the given size using multiple threads
  // NB: State threads: op(thread_index, <begin, end>, result&)
  template <typename TOp, typename TResult>
    void thread_iloop(TOp op, index_t size, std::vector<TResult>& results)
    {
      boost::shared_array<boost::thread> threads(new boost::thread[n_threads()]);
      boost::shared_array<index_t> th_begins(new index_t[n_threads()]);
      boost::shared_array<index_t> th_ends(new index_t[n_threads()]);

      results.resize(n_threads());

      thread_split(size, th_begins.get(), th_ends.get());
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith] = boost::thread(
            boost::bind(op, ith, index_pair_t(th_begins[ith], th_ends[ith]), 
              boost::ref(results[ith])));
      }
      for (index_t ith = 0; ith < n_threads(); ith ++)
      {
        threads[ith].join();
      }
    }

  /////////////////////////////////////////////////////////////////////////////////////////
  // String processing
  /////////////////////////////////////////////////////////////////////////////////////////

  // Split a string given some separators
  strings_t split(const string_t& str, const char* delim_chars = " \n\t\r");

  // Resize the string to obtain the given size
  //  (   if larger - characters will be removed from the end, 
  //      if smaller - <app_char> will be inserted at the end)        
  string_t resize(const string_t& str, index_t size, char app_char = ' ');

  // Trim a float value to the given digits
  string_t round(scalar_t value, index_t digits);

  // Split a string to a list of values	
  template <typename T>
    std::vector<T> split2values(const string_t& value, const char* delim_chars = " \n\t\r")
    {
      std::vector<T> result;

      const visioner::strings_t tokens = visioner::split(value, delim_chars);	
      for (visioner::strings_t::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
      {
        result.push_back(boost::lexical_cast<T>(it->c_str()));
      }

      return result;
    }

  /////////////////////////////////////////////////////////////////////////////////////////
  // File processing
  /////////////////////////////////////////////////////////////////////////////////////////

  // Parse an .ini stream (lines with: "attribute = values [#xxx]", ignoring "#xxx" ones)
  bool load_initext(const string_t& text, strings_t& attributes, strings_t& values);
  bool load_inifile(const string_t& path, strings_t& attributes, strings_t& values);

  // Retrieves the value associated to some attribute
  // If the attribute is not found, the program exits
  string_t attribute_value(const strings_t& attributes, const strings_t& values, const char* attribute);

  // Parse a (sef of) list file(s)
  bool load_listfiles(const string_t& files, strings_t& ifiles, strings_t& gfiles);

  // Load the content of a text file into a string
  bool load_file(const string_t& path, string_t& text);

  // Extract the name of the file, with (filename) and without (basename) extension, from a path
  string_t filename(const string_t& path);
  string_t basename(const string_t& path);
  string_t extname(const string_t& path);

  // Extract the name of the directory from a path
  string_t dirname(const string_t& path);	

}}

#endif // BOB_VISIONER_UTIL_H
