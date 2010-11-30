/**
 * @file src/core/src/DatasetXML.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the DatasetXML class.
 */

#include "core/DatasetXML.h"

namespace Torch {
  namespace core {

    DatasetXML::DatasetXML():
      m_doc(0) {
    }

    DatasetXML::DatasetXML(char *filename) {
      m_doc = xmlParseFile(filename);
    }

    DatasetXML::~DatasetXML() {
      if(m_doc) xmlFreeDoc(m_doc);
    }

  }
}

