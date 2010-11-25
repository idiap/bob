/**
 * @file src/core/src/DatasetXML.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements the DatasetXML class.
 */

#include "core/DatasetXML.h"

template<typename T, int dim> Torch::core::ArrayXML<T,dim>::ArrayXML():
  m_data(0) {

}

Torch::core::DatasetXML::DatasetXML():
  m_doc(0) {
}

Torch::core::DatasetXML::DatasetXML(char *filename) {
  loadDataset(filename);
}

Torch::core::DatasetXML::~DatasetXML() {
  if(m_doc) xmlFreeDoc(m_doc);
}

bool Torch::core::DatasetXML::loadDataset(char *filename) {
  m_doc = xmlParseFile(filename);
  return true;
}

