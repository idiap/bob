/**
 * @file visioner/src/model.cc
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

#include <fstream>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filtering_stream.hpp>
#include <boost/filesystem.hpp>

#include "core/logging.h"

#include "visioner/model/model.h"
#include "visioner/model/mdecoder.h"

/**
 * Determines if the input filename ends in ".gz"
 *
 * @param filename The name of the file to be analyzed.
 */
inline static bool is_dot_gz(const std::string& filename) {
   return boost::filesystem::extension(filename) == ".gz" ||
     boost::filesystem::extension(filename) == ".vbgz";
}

inline static bool is_dot_vbin(const std::string& filename) {
  return boost::filesystem::extension(filename) == ".vbin" ||
    boost::filesystem::extension(filename) == ".vbgz";
}

namespace bob { namespace visioner {

  // Constructor
  Model::Model(const param_t& param)
    :       Parametrizable(param)
  {
    reset(param);
  }

  // Reset to new parameters
  void Model::reset(const param_t& param)
  {
    m_param = param;
    m_mluts.resize(make_tagger(param)->n_outputs());
    for (index_t o = 0; o < n_outputs(); o ++)                        
    {
      m_mluts[o].clear();
    }
  }

  // Reset to new LUTs (lut.size() == model.n_outputs()!)
  bool Model::set(const MultiLUTs& mluts)
  {
    if (mluts.size() != n_outputs())
    {
      return false;
    }

    for (index_t o = 0; o < n_outputs(); o ++)                        
    {
      m_mluts[o] = mluts[o];
    }
    return true;
  }

  // Save/load to/from file
  bool Model::save(const string_t& path) const
  {
    std::ios_base::openmode mode = std::ios_base::out | std::ios_base::trunc;
    if (is_dot_gz(path) || is_dot_vbin(path)) mode |= std::ios_base::binary;
    std::ofstream file(path.c_str(), mode);
    boost::iostreams::filtering_ostream ofs; ///< the output stream
    if (is_dot_gz(path)) 
      ofs.push(boost::iostreams::basic_gzip_compressor<>());
    ofs.push(file);

    if (ofs.good() == false)
    {
      bob::core::error << "Failed to save the model!" << std::endl;
      return false;
    }

    if (is_dot_vbin(path)) { //a binary file from visioner
      boost::archive::binary_oarchive oa(ofs);
      oa << m_param;
      oa << m_mluts;  
      save(oa);
    }
    else {
      boost::archive::text_oarchive oa(ofs);
      oa << m_param;
      oa << m_mluts;  
      save(oa);
    }

    return ofs.good();

  }

  bool Model::load(const string_t& path)
  {
    //AA: adds gzip decompression if necessary (depends on path)
    std::ios_base::openmode mode = std::ios_base::in;
    if (is_dot_gz(path) || is_dot_vbin(path)) mode |= std::ios_base::binary;
    std::ifstream file(path.c_str(), mode);
    boost::iostreams::filtering_istream ifs; ///< the input stream
    if (is_dot_gz(path)) 
      ifs.push(boost::iostreams::basic_gzip_decompressor<>());
    ifs.push(file);

    if (ifs.good() == false)
    {
      bob::core::error << "Failed to load the model!" << std::endl;
      return false;
    }

    if (is_dot_vbin(path)) { //a binary file from visioner
      boost::archive::binary_iarchive ia(ifs);
      ia >> m_param;
      ia >> m_mluts;
      load(ia);
    }
    else { //the default
      boost::archive::text_iarchive ia(ifs);
      ia >> m_param;
      ia >> m_mluts;  
      load(ia);
    }

    return ifs.good();
  }

  bool Model::load(const string_t& path, rmodel_t& model)
  {
    //AA: adds gzip decompression if necessary (depends on path)
    std::ios_base::openmode mode = std::ios_base::in;
    if (is_dot_gz(path) || is_dot_vbin(path)) mode |= std::ios_base::binary;
    std::ifstream file(path.c_str(), mode);
    boost::iostreams::filtering_istream ifs; ///< the input stream
    if (is_dot_gz(path)) 
      ifs.push(boost::iostreams::basic_gzip_decompressor<>());
    ifs.push(file);

    if (ifs.good() == false)
    {
      bob::core::error << "Failed to load the model!" << std::endl;
      return false;
    }

    param_t param;
    if (is_dot_vbin(path)) { //a binary file from visioner
      boost::archive::binary_iarchive ia(ifs);
      ia >> param;
    }
    else { //the default
      boost::archive::text_iarchive ia(ifs);
      ia >> param;
    }

    if (!ifs.good())
    {
      return false;
    }

    model = make_model(param);
    return model->load(path);
  }

  // Compute the model score at the (x, y) position for the output <o>
  scalar_t Model::score(index_t o, int x, int y) const
  {
    return score(o, 0, n_luts(o), x, y);
  }        
  scalar_t Model::score(index_t o, index_t rbegin, index_t rend, int x, int y) const
  {                
    scalar_t sum = 0.0;
    for (index_t r = rbegin; r < rend; r ++)
    {
      const LUT& lut = m_mluts[o][r];
      const index_t fv = get(lut.feature(), x, y);
      sum += lut[fv];
    }
    return sum;
  }

  // Return the selected features
  indices_t Model::features() const
  {
    indices_t result;
    for (index_t o = 0; o < n_outputs(); o ++)
    {
      for (index_t r = 0; r < n_luts(o); r ++)
      {
        const LUT& lut = m_mluts[o][r];
        result.push_back(lut.feature());
      }
    }

    unique(result);

    return result;
  }

}}
