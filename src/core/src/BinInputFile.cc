/**
 * @file src/core/src/BinInputFile.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief This class can be used to load multiarrays from files.
 */


#include "core/BinInputFile.h"

namespace Torch {
  namespace core {

    BinInputFile::BinInputFile(const std::string& filename): 
      m_current_array(0),
      m_in_stream( filename.c_str(), std::ios::in | std::ios::binary ) 
    {
      m_header.read(m_in_stream);
    }

    BinInputFile::~BinInputFile() {
      close();
    }

    void BinInputFile::close() {
      m_in_stream.close();
    } 

    void BinInputFile::load( Arrayset& arrayset) {
      //TODO: implementation
    }

    void BinInputFile::load( Array& array) {
      //TODO: implementation
    }

  }
}

