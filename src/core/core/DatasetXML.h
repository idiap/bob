/**
 * @file src/core/core/DatasetXML.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch abstract representation of a DatasetXML
 */

#ifndef TORCH5SPRO_CORE_DATASET_XML_H 
#define TORCH5SPRO_CORE_DATASET_XML_H

#include "core/Dataset.h"
#include <libxml/parser.h>
#include <libxml/tree.h>


namespace Torch {   
  namespace core {

    class ArrayXML: public Array { //pure virtual
      ArrayXML();
      //
      //load and save blitz::Array dumps, if data contained
      //call loader that knows how to read from file.
      //NULL pointer if no target!
      //Target == contained Array
      //template <typename T> load(const T&);
    };


    class PatternsetXML: public Patternset { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
    };


    class ClusterXML: public Cluster { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
    };


    class MappingXML: public Mapping {

    };


    class DatasetXML: public Dataset { //pure virtual
      public:
        DatasetXML();
        DatasetXML(char *filename);
        ~DatasetXML();
        //query/iterate over:
        //1. "Array"
        //2. "ArraySet"
        //3. "TargetSet"
        bool loadDataset(char *filename);

      private:
        xmlDocPtr m_doc;
    };




  }
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

