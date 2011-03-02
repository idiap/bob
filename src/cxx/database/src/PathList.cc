/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 26 Feb 19:03:51 2011 
 *
 * @brief Implements path search for Torch::database 
 */

#include <algorithm>
#include <boost/tokenizer.hpp>
#include "database/PathList.h"
#include "database/Exception.h"

namespace db = Torch::database;
namespace fs = boost::filesystem;

db::PathList::PathList() : 
  m_list(),
  m_current_path(fs::current_path())
{
}

db::PathList::PathList(const std::string& unixpath) : 
  m_list(), 
  m_current_path(fs::current_path())
{
  typedef boost::tokenizer<boost::char_separator<char> > tok_t;
  static boost::char_separator<char> sep(":");
  tok_t tokens(unixpath, sep);
  for (tok_t::iterator it = tokens.begin(); it != tokens.end(); ++it) 
    append(*it);
}

db::PathList::PathList(const db::PathList& other) : 
  m_list(other.m_list),
  m_current_path(other.m_current_path)
{
}

db::PathList::~PathList() { }

db::PathList& db::PathList::operator= (const db::PathList& other) {
  m_list = other.m_list;
  m_current_path = other.m_current_path;
  return *this;
}

void db::PathList::setCurrentPath(const fs::path& path) {
  if (!path.is_complete()) throw db::PathIsNotAbsolute(path.string()); 
  m_current_path = path;
}

void db::PathList::append(const fs::path& path) {
  m_list.remove(path);
  m_list.push_back(path);
}

void db::PathList::prepend(const fs::path& path) {
  m_list.remove(path);
  m_list.push_front(path);
}

void db::PathList::remove(const fs::path& path) {
  m_list.remove(path);
}

bool db::PathList::contains(const fs::path& path) const {
  return std::find(m_list.begin(), m_list.end(), path) != m_list.end();
}

db::PathList& db::PathList::existing() {
  for (std::list<fs::path>::iterator
      it=m_list.begin(); it!=m_list.end();) { //N.B. we don't increment it!
    if (!fs::exists(*it)) m_list.erase(it++);
    else ++it;
  }
  return *this;
}

/**
 * Removes the last component from the path, supposing it is complete. If it is
 * only root_path(), just return it.
 */
static fs::path trim_one(const fs::path& p) {
  if (p == p.root_path()) return p;

  fs::path retval;
  for (fs::path::iterator it = p.begin(); it!=p.end(); ++it) {
    fs::path::iterator next = it;
    ++next; //< for the lack of better support in boost::filesystem V2
    if (next == p.end()) break; //< == skip the last bit
    retval /= *it;
  }
  return retval;
}

/**
 * We wrote this method because we are using boost::filesystem v2 and the name
 * resolution in this version sucks. If you find yourself maintaining this
 * method, just re-think about using boost::filesystem::absolute, if v3 is
 * already available.
 */
static fs::path absolute(const fs::path& p, const fs::path& current) {
  fs::path completed = fs::complete(p, current);
  fs::path retval;
  for (fs::path::iterator it = completed.begin(); it != completed.end(); ++it) {
    if (*it == "..") {
      retval = trim_one(retval); 
      continue;
    }
    if (*it == ".") { //ignore '.'
      continue;
    }
    retval /= *it;
  }
  return retval;
}

fs::path db::PathList::locate(const fs::path& path) const {
  if (path.is_complete()) return path; //can only locate relative paths
  for (std::list<fs::path>::const_iterator 
      it=m_list.begin(); it!=m_list.end(); ++it) {
    fs::path check = absolute(*it / path, m_current_path);
    if (fs::exists(check)) return check;
  }
  return fs::path(); //emtpy
}

static bool starts_with(const fs::path& input, const fs::path& path) {
  return (input.string().find(path.string()) == 0);
}

fs::path db::PathList::reduce(const fs::path& input) const {
  if (!input.is_complete()) return input; //can only reduce absolute paths
  fs::path abs_input = absolute(input, m_current_path);
  const fs::path* best_match = 0; //holds the best match so far
  for (std::list<fs::path>::const_iterator 
      it=m_list.begin(); it!=m_list.end(); ++it) {
    fs::path abs_path = absolute(*it, m_current_path);
    if (starts_with(abs_input, abs_path)) {
      if (best_match) {
        if (it->string().size() > best_match->string().size()) best_match = &(*it);
        //otherwise, we prefer the first entry in the path list, naturally
      }
      else {
        //no if/else required in this case, we just got our first match
        best_match = &(*it);
      }
    }
  }
  if (!best_match) return input; //no match found
  
  //if you get to this point, you have found a match, return "input-best_match"
  size_t psize = absolute(*best_match, m_current_path).string().size();
  if (absolute(*best_match, m_current_path) != abs_input.root_path()) {
    //if we are anywhere but in the root directory, we need to remove the extra
    //slash after the path name being clipped.
    psize += 1;
  }
  const std::string& istr = abs_input.string();
  return istr.substr(psize, istr.size()-psize);
}
