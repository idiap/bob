/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Mon 28 Feb 11:14:36 2011 
 *
 * @brief Bindings for the PathList type
 */

#include <boost/python.hpp>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>

#include "database/PathList.h"

using namespace boost::python;
namespace fs = boost::filesystem;
namespace db = Torch::database;

static void pathlist_append(db::PathList& pl, const char* path) {
  pl.append(path);
}

static void pathlist_prepend(db::PathList& pl, const char* path) {
  pl.prepend(path);
}

static void pathlist_remove(db::PathList& pl, const char* path) {
  pl.remove(path);
}

static str pathlist_locate(const db::PathList& pl, const char* path) {
  return str(pl.locate(path).string());
}

static bool pathlist_contains(const db::PathList& pl, const char* path) {
  return pl.contains(path);
}

static str pathlist_reduce(const db::PathList& pl, const char* path) {
  return str(pl.reduce(path).string());
}

static str pathlist_getcurrent(const db::PathList& pl) {
  return str(pl.getCurrentPath().string());
}

static void pathlist_setcurrent(db::PathList& pl, const char* path) {
  pl.setCurrentPath(path);
}

static tuple pathlist_paths(const db::PathList& pl) {
  list retval;
  for (std::list<fs::path>::const_iterator it = pl.paths().begin();
      it != pl.paths().end(); ++it) retval.append(str(it->string()));
  return tuple(retval);
}

void bind_database_pathlist() {
  class_<db::PathList, boost::shared_ptr<db::PathList> >("PathList", "Holds a list of searcheable paths", init<>("Initializes a new, empty PathList"))
    .def(init<const std::string&>((arg("unixpath")), "Constructs a PathList object starting from a UNIX like path (separated by ':'. E.g. '.:/my/path1:/my/path2'"))
    .add_property("current_path", &pathlist_getcurrent, &pathlist_setcurrent, "The absolute path to use for resolving contained relative paths")
    .def("append", &pathlist_append, (arg("self"), arg("path")), "Appends another searchable path, if it is not already there. If the path is not complete, it is completed with boost::filesystem::complete().")
    .def("prepend", &pathlist_prepend, (arg("self"), arg("path")), "Prepends another searchable path, if it is not already there. If the path is not complete, it is completed with boost::filesystem::complete().")
    .def("remove", &pathlist_remove, (arg("self"), arg("path")), "Removes a path if it is listed inside. If the path is not complete, it is completed with boost::filesystem::complete()")
    .def("__contains__", &pathlist_contains, (arg("self"), arg("path")), "Tells if  certain path exists internally.")
    .def("existing", &db::PathList::existing, (arg("self")), "Filters my own list, so I only keep existing paths", return_self<>())
    .def("locate", &pathlist_locate, (arg("self"), arg("path")), "Searches a file or directory in all the paths and returns the first match. If the searched path is not found, it returns an empty path. If the input path is absolute (i.e. contains the filesystem root), the output path will be the same as the input path iff the file or directory pointed by the input path exists.")
    .def("reduce", &pathlist_reduce, (arg("self"), arg("path")), "This method will search internally for the longest possible path prefix to an input path. If that is found, I return only the bit of the input path that exceeds that match. Otherwise, I just return the input path. E.g.: internal list is ['/path1', '/path1/other', /path2'] and input path is '/path1/other/arrays/data54.bin' then the return value: 'arrays/data54.bin'")
    .def("paths", &pathlist_paths, (arg("self")), "An immutable handle to all internally stored paths")
    ;
}
