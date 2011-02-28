/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 26 Feb 18:54:51 2011 
 *
 * @brief Helps us search for files in different paths.
 */

#ifndef TORCH_DATABASE_PATHLIST_H 
#define TORCH_DATABASE_PATHLIST_H

#include <list>
#include <string>
#include <boost/filesystem.hpp>

namespace Torch { namespace database {

  class PathList {
    
    public:

      /**
       * Default constructor, starts with an empty path list
       */
      PathList();

      /**
       * Constructs a PathList object starting from a UNIX like path (separated
       * by ':'. E.g. ".:/my/path1:/my/path2"
       */
      PathList(const std::string& unixpath);

      /**
       * Copy constructor
       */
      PathList(const PathList& other);

      /**
       * Virtualized destructor
       */
      ~PathList();

      /**
       * Assignment
       */
      PathList& operator= (const PathList& other);

      /**
       * Appends another searchable path, if it is not already there. If the
       * path is not complete, it is completed with
       * boost::filesystem::complete()
       */
      void append(const boost::filesystem::path& path);

      /**
       * Prepends another searchable path, if it is not already there. If the
       * path is not complete, it is completed with
       * boost::filesystem::complete()
       */
      void prepend(const boost::filesystem::path& path);

      /**
       * Removes a path if it is listed inside. If the path is not complete, it
       * is completed with boost::filesystem::complete()
       */
      void remove(const boost::filesystem::path& path);

      /**
       * Returns all paths defined inside
       */
      inline const std::list<boost::filesystem::path>& paths() const 
      { return m_list; }

      /**
       * Searches a file or directory in all the paths and returns the first
       * match. If the searched path is not found, it returns an empty path
       * (i.e. path.empty() will return 'true'). If the input path is absolute
       * (i.e. contains the filesystem root), the output path will be the same
       * as the input path iff the file or directory pointed by the input path
       * exists.
       */
      boost::filesystem::path locate(const boost::filesystem::path& path) const;

      /**
       * This method will search internally for the longest possible path
       * prefix to an input path. If that is found, I return only the bit of
       * the input path that exceeds that match. Otherwise, I just return the
       * input path.
       *
       * E.g.: internal list is ['/path1', '/path1/other', /path2']
       *       input path is '/path1/other/arrays/data54.bin'
       *       return value: 'arrays/data54.bin'
       */
      boost::filesystem::path reduce(const boost::filesystem::path& input) const;

    private:

      std::list<boost::filesystem::path> m_list; ///< all added paths

  };

}}

#endif /* TORCH_DATABASE_PATHLIST_H */
