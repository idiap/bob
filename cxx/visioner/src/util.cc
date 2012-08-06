/**
 * @file visioner/src/util.cc
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

#include <iostream>
#include <fstream>
#include <boost/algorithm/string/trim.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>

#include <QObject>

#include "core/logging.h"

#include "visioner/util/util.h"

namespace bob { namespace visioner {

  // Resize the string to obtain the given size
  //  (   if larger - characters will be removed from the end, 
  //      if smaller - <app_char> will be inserted at the end)        
  std::string resize(const std::string& str, uint64_t size, char app_char)
  {
    std::string result = str;
    if (result.size() < size)
    {
      result.insert(result.end(), size - result.size(), app_char);
    }
    else if (result.size() > size)
    {
      result.erase(result.begin() + size, result.end());  
    }

    return result;
  }

  // Trim a float value to the given digits
  std::string round(double value, uint64_t digits)
  {
    return QObject::tr("%1").arg(value, 0, 'f', digits).toStdString();
  }        

  // Load the content of a text file into a string
  bool load_file(const std::string& path, std::string& text)
  {
    std::ifstream is(path.c_str());
    if (is.is_open() == false)
    {
      return false;
    }

    static const int buff_size = 4096;
    char buff[buff_size];

    text.clear();
    while (is.getline(buff, buff_size))
    {
      text += buff;
      text += "\n";
    }

    is.close();
    return true;
  }

  // Parse a list file
  static bool load_listfile(const std::string& path, std::vector<std::string>& ifiles, std::vector<std::string>& gfiles)
  {
    static const int buff_size = 4096;
    char buff[buff_size];

    ifiles.clear();
    gfiles.clear();

    // Open the file
    std::ifstream is(path.c_str());
    if (is.is_open() == false)
    {
      return false;
    }

    // Parse the file
    std::string dir;
    while (is.getline(buff, buff_size))
    {
      std::vector<std::string> tokens;
      boost::split(tokens, buff, boost::is_any_of("#"));
      if (tokens.size() == 1)
      {
        dir = boost::filesystem::path(path).parent_path().string() + boost::trim_copy(tokens[0]);
      }
      else if (tokens.size() == 2)
      {
        ifiles.push_back(dir + boost::trim_copy(tokens[0]));
        gfiles.push_back(dir + boost::trim_copy(tokens[1]));
      }
      else
      {
        is.close();
        return false;
      }
    }

    //                ifiles.erase(ifiles.begin() + std::min(ifiles.size(), (std::size_t)128), ifiles.end());
    //                gfiles.erase(gfiles.begin() + std::min(gfiles.size(), (std::size_t)128), gfiles.end());

    // OK
    is.close();
    return true;
  }	

  // Parse a (sef of) list file(s)
  bool load_listfiles(const std::string& files, std::vector<std::string>& ifiles, std::vector<std::string>& gfiles)
  {
    ifiles.clear(), gfiles.clear();

    std::vector<std::string> tokens;
    boost::split(tokens, files, boost::is_any_of(":\n\t\r"));
    for (std::vector<std::string>::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
    {
      std::vector<std::string> ifiles_crt, gfiles_crt;
      if (	load_listfile(*it, ifiles_crt, gfiles_crt) == false ||
          ifiles_crt.empty() || ifiles_crt.size() != gfiles_crt.size())
      {
        return false;
      }

      ifiles.insert(ifiles.end(), ifiles_crt.begin(), ifiles_crt.end());
      gfiles.insert(gfiles.end(), gfiles_crt.begin(), gfiles_crt.end());
    }

    return !ifiles.empty() && ifiles.size() == gfiles.size();
  }	

  std::string basename(const std::string& path) {
    return boost::filesystem::path(path).stem().c_str();
  }

}}
