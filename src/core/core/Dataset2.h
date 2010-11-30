/**
 * @file src/core/core/Dataset2.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
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
     * @brief The arrayset class for a dataset
     */
    class Arrayset { //pure virtual
      //
      //query/iterate over:
      //1. "Array"
      public:
/*        class const_iterator {
        };

        const_iterator begin() const =0;
        const_iterator end() const =0;
*/
/* A method cannot be both virtual and template: this method should
 * be implemented in the inherited class 
        virtual template<typename T, int D> at(size_t id, blitz::Array<T,D>& output) =0;*/
        // blitz::Array<float, 2> myarray;
        // arrayset->at(3, myarray);
        
    };


    /**
     * @brief The relation class for a dataset
     */
    class Relation { //pure virtual
    };

    /**
     * @brief The rule class for a dataset
     */
    class Rule { //pure virtual
    };

    /**
     * @brief The relationset class for a dataset
     */
    class Relationset { //pure virtual
    };

  
    /**
     * @brief The main dataset class
    */
    class Dataset { //pure virtual
      //query/iterate over:
      //1. "Patternset"
      //2. "Cluster"
      //3. "Mapping"
      public:
/*        class const_iterator {
          virtual void f() const =0;
        };
   
         
        virtual const_iterator begin() const =0;
        virtual const_iterator end() const =0;

        virtual const ArraySet& at (size_t id) const =0;*/
      private:
        
    };


  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

