/**
 * @file src/core/core/Dataset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch abstract representation of a Dataset
 */

#ifndef TORCH5SPRO_CORE_DATASET_H 
#define TORCH5SPRO_CORE_DATASET_H


namespace Torch {   
  namespace core {

    class Array { //pure virtual
      //
      //load and save blitz::Array dumps, if data contained
      //call loader that knows how to read from file.
      //NULL pointer if no target!
      //Target == contained Array
      //template <typename T> load(const T&);
    };


    class Patternset { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
    };


    class Cluster { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
    };


    class Mapping {

    };


    class Dataset { //pure virtual
      //query/iterate over:
      //1. "Array"
      //2. "ArraySet"
      //3. "TargetSet"
      virtual bool loadDataset(char *filename) = 0;
    };




  }
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

