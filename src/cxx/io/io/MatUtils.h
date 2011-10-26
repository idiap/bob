/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 21 Feb 13:51:53 2011 
 *
 * @brief Utilities to read and write .mat (Matlab) binary files
 */

#ifndef TORCH_IO_MATUTILS_H 
#define TORCH_IO_MATUTILS_H

#include "core/array_type.h"
#include <blitz/array.h>

#include <matio.h>
#include <boost/format.hpp>
#include <boost/shared_ptr.hpp>
#include <map>
#include <string>

#include "io/buffer.h"
#include "io/utils.h"
#include "io/Exception.h"

namespace Torch { namespace io { namespace detail {

  /**
   * This method will create a new boost::shared_ptr to mat_t that knows how to
   * delete itself
   */
  boost::shared_ptr<mat_t> make_matfile(const std::string& filename, int flags);

  /**
   * Retrieves information about the first variable found on a file. 
   */
  void mat_peek(const std::string& filename, Torch::io::typeinfo& info);

  /**
   * Retrieves information about the first variable with a certain name
   * (array_%d) that exists in a .mat file (if it exists)
   */
  void mat_peek_set(const std::string& filename, Torch::io::typeinfo& info);

  /**
   * Retrieves information about all variables with a certain name (array_%d)
   * that exist in a .mat file
   */
  boost::shared_ptr<std::map<size_t, 
    std::pair<std::string, Torch::io::typeinfo> > > 
      list_variables(const std::string& filename);

  /**
   * Reads a variable on the (already opened) mat_t file. If you don't
   * specify the variable name, I'll just read the next one. Re-allocates the
   * buffer if required.
   */
  void read_array (boost::shared_ptr<mat_t> file, Torch::io::buffer& buf,
      const std::string& varname="");

  /**
   * Appends a single Array into the given matlab file and with a given name
   */
  void write_array(boost::shared_ptr<mat_t> file, 
      const std::string& varname, const Torch::io::buffer& buf);
 
}}}

#endif /* TORCH_IO_MATUTILS_H */
