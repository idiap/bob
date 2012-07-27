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

#include <QObject>

#include "visioner/util/util.h"

namespace bob { namespace visioner {

  // Split some objects to process using multiple threads
  void thread_split(index_t n_objects, index_t* sbegins, index_t* sends)
  {
    const index_t n_objects_per_thread = n_objects / n_threads() + 1;
    for (index_t ith = 0, sbegin = 0; ith < n_threads(); ith ++)
    {
      sbegins[ith] = sbegin;
      sbegin = std::min(sbegin + n_objects_per_thread, n_objects);
      sends[ith] = sbegin;
    }
  }

  // Logging
  static std::ostream& log(const char* base, const char* class_name, const char* func_name)
  {
    std::cout << base << " <" << class_name << "::" << func_name << ">: ";
    return std::cout;
  }
  static std::ostream& log(const char* base, const char* func_name)
  {
    std::cout << base << " <" << func_name << ">: ";
    return std::cout;
  }

  std::ostream& log_warning(const char* class_name, const char* func_name)
  {
    return log("Warning", class_name, func_name);
  }
  std::ostream& log_warning(const char* func_name)
  {
    return log("Warning", func_name);
  }
  std::ostream& log_warning()
  {
    return std::cout;
  }

  std::ostream& log_error(const char* class_name, const char* func_name)
  {
    return log("Error", class_name, func_name);
  }
  std::ostream& log_error(const char* func_name)
  {
    return log("Error", func_name);
  }
  std::ostream& log_error()
  {
    return std::cout;
  }

  std::ostream& log_info(const char* class_name, const char* func_name)
  {
    return log("Info", class_name, func_name);
  }
  std::ostream& log_info(const char* func_name)
  {
    return log("Info", func_name);
  }	
  std::ostream& log_info()
  {
    return std::cout;
  }

  void log_finished()
  {
    std::cout << "Program finished correctly!\n";
  }

  // Trim a string
  string_t trim(const string_t& str, const char* trim_chars)
  {
    // Find the beginning of the trimmed string
    const index_t pos_beg = str.find_first_not_of(trim_chars);
    if (pos_beg == string_t::npos)
    {
      return "";
    }
    else
    {
      // Also the end of the trimmed string
      const index_t pos_end = str.find_last_not_of(trim_chars);
      return str.substr(pos_beg, pos_end - pos_beg + 1);
    }
  }

  // Split a string given some separators
  strings_t split(const string_t& str, const char* delim_chars)
  {
    strings_t parse_strs;

    // Find the beginning of the splitted strings_t ...
    index_t pos_beg = str.find_first_not_of(delim_chars);
    while (pos_beg != string_t::npos)
    {
      // Find the end of the splitted strings_t ...
      index_t pos_end = str.find_first_of(delim_chars, pos_beg + 1);
      if (pos_end == string_t::npos)
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
  string_t resize(const string_t& str, index_t size, char app_char)
  {
    string_t result = str;
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
  string_t round(scalar_t value, index_t digits)
  {
    return QObject::tr("%1").arg(value, 0, 'f', digits).toStdString();
  }        

  // Parse an .ini stream (lines with: "attribute = values [#xxx]", ignoring "#xxx" ones)
  bool load_initext(const string_t& text, strings_t& attributes, strings_t& values)
  {
    attributes.clear();
    values.clear();

    // Parse each line ...
    const strings_t lines = split(text, "\n");
    for (strings_t::const_iterator it = lines.begin(); it != lines.end(); ++ it)
    {
      // Check line (empty or comment)
      const string_t line = trim(*it);
      if (line.empty() || line[0] == '#')
      {
        continue;
      }

      // Parse line
      const strings_t tokens = split(line, "=");
      if (tokens.size() != 2)
      {
        return false;
      }

      const string_t attribute = trim(split(tokens[0], "#")[0]);
      const string_t value = trim(split(tokens[1], "#")[0]);

      strings_t::const_iterator it2 = std::find(attributes.begin(), attributes.end(), attribute);
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

  bool load_inifile(const string_t &path, strings_t& attributes, strings_t& values)
  {
    string_t text;
    if (load_file(path, text) == false)
    {
      return false;
    }

    return load_initext(text, attributes, values);
  }

  string_t attribute_value(const strings_t& attributes, const strings_t& values,
      const char* attribute)
  {
    for (strings_t::const_iterator it_a = attributes.begin(), it_v = values.begin(); 
        it_a != attributes.end() && it_v != values.end(); ++ it_a, ++ it_v)
    {
      if (*it_a == attribute)
      {
        return *it_v;
      }
    }

    log_error("attribute_value") << "Missing value for the attribute <" << attribute << ">!\n";
    return "";
  }

  // Load the content of a text file into a string
  bool load_file(const string_t& path, string_t& text)
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
  string_t filename(const string_t& path)
  {
    index_t pos = path.find_last_of("\\/");
    if (pos == string_t::npos)
    {
      return path;
    }
    else
    {
      pos = path.find_last_of("\\/", pos + 1);
      return path.substr(pos + 1, path.size() - pos);
    }
  }

  string_t basename(const string_t& path)
  {
    const string_t file = filename(path);

    const index_t pos = file.find_last_of(".");
    if (pos == string_t::npos)
    {
      return file;
    }
    else
    {
      return file.substr(0, pos);
    }
  }

  string_t extname(const string_t& path)
  {
    const string_t file = filename(path);

    const index_t pos = file.find_last_of(".");
    if (pos == string_t::npos)
    {
      return "";
    }
    else
    {
      return file.substr(pos, file.size());
    }
  }

  // Extract the name of the directory from a path
  string_t dirname(const string_t &path)
  {
    index_t pos = path.find_last_of("\\/");
    if (pos == string_t::npos)
    {
      return "./";
    }
    else
    {
      return path.substr(0, pos + 1);
    }
  }

  // Parse a list file
  static bool load_listfile(const string_t& path, strings_t& ifiles, strings_t& gfiles)
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
    string_t dir;
    while (is.getline(buff, buff_size))
    {
      const strings_t tokens = split(buff, "#");
      if (tokens.size() == 1)
      {
        dir = dirname(path) + trim(tokens[0]);
      }
      else if (tokens.size() == 2)
      {
        ifiles.push_back(dir + trim(tokens[0]));
        gfiles.push_back(dir + trim(tokens[1]));
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
  bool load_listfiles(const string_t& files, strings_t& ifiles, strings_t& gfiles)
  {
    ifiles.clear(), gfiles.clear();

    const strings_t tokens = split(files, ":\n\t\r");
    for (strings_t::const_iterator it = tokens.begin(); it != tokens.end(); ++ it)
    {
      strings_t ifiles_crt, gfiles_crt;
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
