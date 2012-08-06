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

#include <QObject>

#include "core/logging.h"

#include "visioner/util/util.h"

namespace bob { namespace visioner {

  // Split some objects to process using multiple threads
  void thread_split(uint64_t n_objects, uint64_t* sbegins, uint64_t* sends)
  {
    const uint64_t n_objects_per_thread = n_objects / n_threads() + 1;
    for (uint64_t ith = 0, sbegin = 0; ith < n_threads(); ith ++)
    {
      sbegins[ith] = sbegin;
      sbegin = std::min(sbegin + n_objects_per_thread, n_objects);
      sends[ith] = sbegin;
    }
  }

  // Split a string given some separators
  std::vector<std::string> split(const std::string& str, const char* delim_chars)
  {
    std::vector<std::string> parse_strs;

    // Find the beginning of the splitted std::vector<std::string> ...
    uint64_t pos_beg = str.find_first_not_of(delim_chars);
    while (pos_beg != std::string::npos)
    {
      // Find the end of the splitted std::vector<std::string> ...
      uint64_t pos_end = str.find_first_of(delim_chars, pos_beg + 1);
      if (pos_end == std::string::npos)
        pos_end = str.size();
      if (pos_end != pos_beg)
        parse_strs.push_back(str.substr(pos_beg, pos_end - pos_beg));

      // Continue to iterate for the next splitted string
      pos_beg = str.find_first_not_of(delim_chars, pos_end);
    }

    if (parse_strs.empty())
    {
      parse_strs.push_back(str);
    }

    return parse_strs;
  }

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

  // Parse an .ini stream (lines with: "attribute = values [#xxx]", ignoring "#xxx" ones)
  bool load_initext(const std::string& text, std::vector<std::string>& attributes, std::vector<std::string>& values)
  {
    attributes.clear();
    values.clear();

    // Parse each line ...
    const std::vector<std::string> lines = split(text, "\n");
    for (std::vector<std::string>::const_iterator it = lines.begin(); it != lines.end(); ++ it)
    {
      // Check line (empty or comment)
      const std::string line = boost::trim_copy(*it);
      if (line.empty() || line[0] == '#')
      {
        continue;
      }

      // Parse line
      const std::vector<std::string> tokens = split(line, "=");
      if (tokens.size() != 2)
      {
        return false;
      }

      const std::string attribute = boost::trim_copy(split(tokens[0], "#")[0]);
      const std::string value = boost::trim_copy(split(tokens[1], "#")[0]);

      std::vector<std::string>::const_iterator it2 = std::find(attributes.begin(), attributes.end(), attribute);
      if (it2 != attributes.end())
      {
        values[it2 - attributes.begin()] = value;
      }
      else
      {
        attributes.push_back(attribute);			
        values.push_back(value);
      }
    }

    return true;
  }

  bool load_inifile(const std::string &path, std::vector<std::string>& attributes, std::vector<std::string>& values)
  {
    std::string text;
    if (load_file(path, text) == false)
    {
      return false;
    }

    return load_initext(text, attributes, values);
  }

  std::string attribute_value(const std::vector<std::string>& attributes, const std::vector<std::string>& values,
      const char* attribute)
  {
    for (std::vector<std::string>::const_iterator it_a = attributes.begin(), it_v = values.begin(); 
        it_a != attributes.end() && it_v != values.end(); ++ it_a, ++ it_v)
    {
      if (*it_a == attribute)
      {
        return *it_v;
      }
    }

    bob::core::error << "Missing value for the attribute <"
      << attribute << ">!" << std::endl;
    return "";
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

  // Extract the name of the file, with (filename) and without (basename) extension, from a path
  std::string filename(const std::string& path)
  {
    uint64_t pos = path.find_last_of("\\/");
    if (pos == std::string::npos)
    {
      return path;
    }
    else
    {
      pos = path.find_last_of("\\/", pos + 1);
      return path.substr(pos + 1, path.size() - pos);
    }
  }

  std::string basename(const std::string& path)
  {
    const std::string file = filename(path);

    const uint64_t pos = file.find_last_of(".");
    if (pos == std::string::npos)
    {
      return file;
    }
    else
    {
      return file.substr(0, pos);
    }
  }

  std::string extname(const std::string& path)
  {
    const std::string file = filename(path);

    const uint64_t pos = file.find_last_of(".");
    if (pos == std::string::npos)
    {
      return "";
    }
    else
    {
      return file.substr(pos, file.size());
    }
  }

  // Extract the name of the directory from a path
  std::string dirname(const std::string &path)
  {
    uint64_t pos = path.find_last_of("\\/");
    if (pos == std::string::npos)
    {
      return "./";
    }
    else
    {
      return path.substr(0, pos + 1);
    }
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
      const std::vector<std::string> tokens = split(buff, "#");
      if (tokens.size() == 1)
      {
        dir = dirname(path) + boost::trim_copy(tokens[0]);
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

    const std::vector<std::string> tokens = split(files, ":\n\t\r");
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

}}
