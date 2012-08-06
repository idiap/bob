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
#include <algorithm>
#include <vector>
#include <stdint.h>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#include <boost/lexical_cast.hpp>

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
  template <typename T> double inverse(T value) {
    static const T zero = (T)0;
    return value == zero ? 1.0 : 1.0 / (0.0 + value);
  }	

  // Remove duplicates from a given vector
  template <typename T> void unique(std::vector<T>& data) {
    std::sort(data.begin(), data.end());
    data.erase(std::unique(data.begin(), data.end()), data.end());
  }

  /**
   * String processing
   */

  /**
   * Resize the string to obtain the given size
   *  (   if larger - characters will be removed from the end, 
   *      if smaller - <app_char> will be inserted at the end)        
   */
  std::string resize(const std::string& str, uint64_t size, 
      char app_char = ' ');

  /**
   * Trims a float value to the given digits
   */
  std::string round(double value, uint64_t digits);

  /**
   * Splits a string to a list of values	
   */
  template <typename T> std::vector<T> split2values(const std::string& value, 
      const char* delim_chars = " \n\t\r")
  {
    std::vector<T> result;
    std::vector<std::string> tokens;
    boost::split(tokens, value, boost::is_any_of(delim_chars));
    result.reserve(tokens.size());
    for (std::vector<std::string>::const_iterator
        it = tokens.begin(); it != tokens.end(); ++ it) {
      result.push_back(boost::lexical_cast<T>(it->c_str()));
    }
    return result;
  }

  /**
   * File processing
   */

  /**
   * Parse a (sef of) list file(s)
   */
  bool load_listfiles(const std::string& files,
      std::vector<std::string>& ifiles, std::vector<std::string>& gfiles);

  /**
   * Load the content of a text file into a string
   */
  bool load_file(const std::string& path, std::string& text);

  /**
   * Returns boost::filesystem::path::stem().
   */
  std::string basename(const std::string& path);

}}

#endif // BOB_VISIONER_UTIL_H
