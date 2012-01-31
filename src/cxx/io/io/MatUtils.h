/**
 * @file cxx/io/io/MatUtils.h
 * @date Wed Jun 22 17:50:08 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief Utilities to read and write .mat (Matlab) binary files
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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

#ifndef BOB_IO_MATUTILS_H 
#define BOB_IO_MATUTILS_H

#include "core/array_type.h"
#include <blitz/array.h>

#include <matio.h>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>

#include "core/array.h"
#include "io/Exception.h"

namespace bob { namespace io { namespace detail {

  /**
   * This method will create a new boost::shared_ptr to mat_t that knows how to
   * delete itself
   */
  boost::shared_ptr<mat_t> make_matfile(const std::string& filename, int flags);

  /**
   * Retrieves information about the first variable found on a file. 
   */
  void mat_peek(const std::string& filename,
      bob::core::array::typeinfo& info);

  /**
   * Retrieves information about the first variable with a certain name
   * (array_%d) that exists in a .mat file (if it exists)
   */
  void mat_peek_set(const std::string& filename,
      bob::core::array::typeinfo& info);

  /**
   * Retrieves information about all variables with a certain name (array_%d)
   * that exist in a .mat file
   */
  boost::shared_ptr<std::map<size_t, 
    std::pair<std::string, bob::core::array::typeinfo> > > 
      list_variables(const std::string& filename);

  /**
   * Reads a variable on the (already opened) mat_t file. If you don't
   * specify the variable name, I'll just read the next one. Re-allocates the
   * buffer if required.
   */
  void read_array (boost::shared_ptr<mat_t> file,
      bob::core::array::interface& buf, const std::string& varname="");

  /**
   * Appends a single Array into the given matlab file and with a given name
   */
  void write_array(boost::shared_ptr<mat_t> file, 
      const std::string& varname, const bob::core::array::interface& buf);
 
}}}

#endif /* BOB_IO_MATUTILS_H */
