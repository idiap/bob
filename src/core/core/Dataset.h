/**
 * @file src/core/core/Dataset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch abstract representation of a Dataset
 */

#ifndef TORCH5SPRO_CORE_DATASET_H 
#define TORCH5SPRO_CORE_DATASET_H


namespace Torch {   
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

    /**
     * @brief The array class for a dataset
     */
    class Array { //pure virtual
      //
      //load and save blitz::Array dumps, if data contained
      //call loader that knows how to read from file.
      //NULL pointer if no target!
      //Target == contained Array
      //template <typename T> load(const T&);
    };


    /**
     * @brief The patternset class for a dataset
     */
    class Patternset { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
    };


    /**
     * @brief The cluster class for a dataset
     */
    class Cluster { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
    };

  
    /**
     * @brief The mapping class for a dataset
     */
    class Mapping {

    };


    /**
     * @brief The main dataset class
     */
    class Dataset { //pure virtual
      //query/iterate over:
      //1. "Array"
      //2. "ArraySet"
      //3. "TargetSet"
      virtual bool loadDataset(char *filename) = 0;
    };


  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

