/**
 * @file src/cxx/core/core/Dataset2.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief A torch representation of a Dataset
 */

#ifndef TORCH5SPRO_CORE_DATASET_H
#define TORCH5SPRO_CORE_DATASET_H

#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <blitz/array.h>
#include "core/logging.h"
#include "core/Exception.h"
#include "core/StaticComplexCast.h"
#include "core/dataset_common.h"

#include "core/Arrayset.h"
#include "core/Relationset.h"

#include <string>
#include <map>
#include <cstdlib> // required when using size_t type



namespace Torch {   
  /**
   * \ingroup libcore_api
   * @{
   *
   */
  namespace core {

    /**
     * @brief The main dataset class
    */
    class Dataset {
      public:
        /**
         * @brief Constructor
         */
        Dataset();
        /**
         * @brief Destructor
         */
        ~Dataset();

        /**
         * @brief Add an Arrayset to the Dataset
         */
        void append( boost::shared_ptr<Arrayset> arrayset);
        /**
         * @brief Remove a Relationset with a given name from the Dataset
         */
        void remove( const size_t id) {
          std::map<size_t, boost::shared_ptr<Arrayset> >::iterator it =
            m_arrayset.find(id);
          if(it!=m_arrayset.end())
            m_arrayset.erase(it);
          else
            throw NonExistingElement();
        }

        /**
         * @brief Get the name of the Dataset
         */
        const std::string& getName() const { return m_name; }

        /**
         * @brief Set the name of the Dataset
         */
        void setName( const std::string& name) { m_name = name; }

        /**
         * @brief Get the version of the Dataset
         */
        const size_t getVersion() const { return m_version; }

        /**
         * @brief Set the version of the Dataset
         */
        void setVersion( const size_t version) { m_version = version; }

        /**
         * @brief const_iterator over the Arraysets of the Dataset
         */
        typedef std::map<size_t, boost::shared_ptr<Arrayset> >::const_iterator
          const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Arrayset of 
         * the Dataset
         */
        const_iterator begin() const { return m_arrayset.begin(); }
        /**
         * @brief Return a const_iterator pointing at the last Arrayset of 
         * the Dataset
         */
        const_iterator end() const { return m_arrayset.end(); }

        /**
         * @brief iterator over the Arraysets of the Dataset
         */
        typedef std::map<size_t, boost::shared_ptr<Arrayset> >::iterator 
          iterator;
        /**
         * @brief Return an iterator pointing at the first Arrayset of 
         * the Dataset
         */
        iterator begin() { return m_arrayset.begin(); }
        /**
         * @brief Return an iterator pointing at the first Arrayset of 
         * the Dataset
         */
        iterator end() { return m_arrayset.end(); }
   
        /**
         * @brief Return the Arrayset of the given id 
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the arraysets.
         */
        const Arrayset& operator[]( const size_t id ) const;
        /**
         * @brief Return the arrayset of the given id
         */
        boost::shared_ptr<const Arrayset> getArrayset(const size_t id) const;
        boost::shared_ptr<Arrayset> getArrayset(const size_t id);

        /**
         * @brief Add a Relationset to the Dataset
         */
        void append( boost::shared_ptr<Relationset> relationset);
        /**
         * @brief Remove a Relationset with a given name from the Dataset
         */
        void remove( const std::string& name) {
          std::map<std::string, boost::shared_ptr<Relationset> >::iterator it=
            m_relationset.find(name);
          if(it!=m_relationset.end())
            m_relationset.erase(it);
          else
            throw NonExistingElement();
        }

        /**
         * @brief const_iterator over the Relationsets of the Dataset
         */
        typedef std::map<std::string, boost::shared_ptr<Relationset> >::
          const_iterator relationset_const_iterator;
        /**
         * @brief Return a const_iterator pointing at the first Relationset of
         * the Dataset
         */
        relationset_const_iterator relationset_begin() const { 
          return m_relationset.begin(); }
        /**
         * @brief Return a const_iterator pointing at the last Relationset of 
         * the Dataset
         */
        relationset_const_iterator relationset_end() const { 
          return m_relationset.end(); }

        /**
         * @brief iterator over the Relationsets of the Dataset
         */
        typedef std::map<std::string, boost::shared_ptr<Relationset> >::iterator
          relationset_iterator;
        /**
         * @brief Return an iterator pointing at the first Relationset of 
         * the Dataset
         */
        relationset_iterator relationset_begin() { return m_relationset.begin(); }
        /**
         * @brief Return an iterator pointing at the first Relationset of 
         * the Dataset
         */
        relationset_iterator relationset_end() { return m_relationset.end(); }
   
        /**
         * @brief Return the Relationset of the given name
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the relationsets.
         */
        const Relationset& operator[]( const std::string& name ) const;

        /**
         * @brief Return the arrayset of the given id
         */
        boost::shared_ptr<const Relationset> 
        getRelationset( const std::string& name ) const;
        boost::shared_ptr<Relationset> 
        getRelationset( const std::string& name );

      private:
        std::string m_name;
        size_t m_version;

        std::map<size_t, boost::shared_ptr<Arrayset> > m_arrayset;
        std::map<std::string, boost::shared_ptr<Relationset> > m_relationset;
    };



  }
  /**
   * @}
   */
}

#endif /* TORCH5SPRO_CORE_DATASET_H */

