/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 26 Feb 19:03:51 2011 
 *
 * @brief Implements path search for Torch::database 
 */

#include <algorithm>
#include <boost/tokenizer.hpp>
#include "database/PathList.h"

namespace db = Torch::database;
namespace fs = boost::filesystem;

db::PathList::PathList() : m_list() {
}

db::PathList::PathList(const std::string& unixpath) : m_list() {
  typedef boost::tokenizer<boost::char_separator<char> > tok_t;
  static boost::char_separator<char> sep(":");
  tok_t tokens(unixpath, sep);
  for (tok_t::iterator it = tokens.begin(); it != tokens.end(); ++it) 
    append(*it);
}

db::PathList::PathList(const db::PathList& other) : m_list(other.m_list) {
}

db::PathList::~PathList() { }

db::PathList& db::PathList::operator= (const db::PathList& other) {
  m_list = other.m_list;
  return *this;
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

fs::path db::PathList::locate(const fs::path& path) const {
  if (path.is_complete()) return path; //can only locate relative paths
  for (std::list<fs::path>::const_iterator 
      it=m_list.begin(); it!=m_list.end(); ++it) {
    if (fs::exists(*it / path)) return fs::complete(*it / path);
  }
  return fs::path(); //emtpy
}

static bool starts_with(const fs::path& input, const fs::path& path) {
  return (input.string().find(fs::complete(path).string()) == 0);
}

fs::path db::PathList::reduce(const fs::path& input) const {
  if (!input.is_complete()) return input; //can only reduce absolute paths
  const fs::path* best_match = 0; //holds the best match so far
  for (std::list<fs::path>::const_iterator 
      it=m_list.begin(); it!=m_list.end(); ++it) {
    if (starts_with(input, *it)) {
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
  return input.string().substr(fs::complete(*best_match).string().size()+1);
}
