/**
 * @file database/Dataset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a>
 *
 * @brief A torch representation of a Dataset
 */

#ifndef TORCH_DATABASE_DATASET_H
#define TORCH_DATABASE_DATASET_H

#include <string>
#include <map>
#include <cstdlib>
#include <boost/shared_ptr.hpp>

#include "database/Arrayset.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   *
   */
  namespace database {

    /**
     * The main dataset class.
     */
    class Dataset {

      public:

        /**
         * Default constructor, builds an otherwise empty Dataset
         */
        Dataset(const std::string& name, size_t version);

        /**
         * Reads the contents of this database from a given file
         */
        Dataset(const std::string& path);

        /**
         * Copy construct a Dataset by copying all members of the other Dataset
         */
        Dataset(const Dataset& other);

        /**
         * Destructor virtualization
         */
        virtual ~Dataset();

        /**
         * Assignment operator, copy all members of the other Dataset
         */
        Dataset& operator= (const Dataset& other);

        /**
         * @brief Get the name of the Dataset
         */
        inline const std::string& getName() const { return m_name; }

        /**
         * @brief Set the name of the Dataset
         */
        inline void setName(const std::string& name) { m_name = name; }

        /**
         * @brief Get the version of the Dataset
         */
        inline size_t getVersion() const { return m_version; }

        /**
         * @brief Set the version of the Dataset
         */
        inline void setVersion(const size_t version) { m_version = version; }

        /**
         * Appends a copy of an Arrayset into this Dataset. If you specify the
         * id of an Arrayset that already exists, we overwrites the existing
         * one and replace it with this one. 
         *
         * @return The id assigned to the arrayset.
         */
        size_t add(boost::shared_ptr<const Arrayset> arrayset);

        /**
         * Appends a copy of an Arrayset into this Dataset. If you specify the
         * id of an Arrayset that already exists, we overwrites the existing
         * one and replace it with this one. 
         *
         * @return The id assigned to the arrayset.
         */
        size_t add(const Arrayset& arrayset);

        /**
         * Removes an Arrayset with a given index from the Dataset. Please note
         * that this may also remove all relations that do depend on this
         * Arrayset.
         */
        void remove(size_t index);

        /**
         * Returns my internal arrayset index
         */
        inline const std::map<size_t, boost::shared_ptr<Arrayset> >& arraysetIndex() const { return m_id2arrayset; }

        /**
         * Returns my internal list of arraysets, by insertion order.
         */
        inline const std::list<boost::shared_ptr<Arrayset> >& arraysets () const { return m_arrayset; }

        /**
         * Returns the Arrayset given a certain, valid, arrayset-id
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the arraysets.
         */
        const Arrayset& operator[] (size_t id) const;
        Arrayset& operator[] (size_t id);

        /**
         * Returns the arrayset of the given id
         */
        boost::shared_ptr<const Arrayset> ptr(const size_t id) const;
        boost::shared_ptr<Arrayset> ptr(const size_t id);

        /**
         * @brief Add a Relationset to the Dataset. Please note that the
         * attribute "name" of the Relationset is used as acess key. If you
         * provide the name of an existing Relationset, it is replaced by this
         * one.
         */
        //void add(boost::shared_ptr<Relationset> relationset);

        /**
         * This method creates an internal copy of the given relationset and
         * store internally. If the relationset contains the name of an object
         * that already exists in this database, it is overwritten.
         */
        //void add(const Relationset& arrayset);

        /**
         * @brief Remove a Relationset with a given name from the Dataset
         */
        //void remove (const std::string& name);

        /**
         * @brief Return the Relationset of the given name
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the relationsets.
         */
        //const Relationset& operator[](const std::string& name) const;
        //Relationset& operator[](const std::string& name);

        /**
         * @brief Return the relationset of the given name
         */
        //boost::shared_ptr<const Relationset> getRelationset(const std::string& name) const;
        //boost::shared_ptr<Relationset> getRelationset(const std::string& name);

        /**
         * Returns my internal relationset index
         */
        //inline const std::map<std::string, boost::shared_ptr<Relationset> >& relationIndex() const { return m_name2relationset; }

        /**
         * Gets the next free id
         */
        size_t getNextFreeId() const;

        /**
         * Consolidates the arrayset ids by resetting the first arrayset to
         * have id = 1, the second id = 2 and so on.
         */
        void consolidateIds();

      private:
        std::string m_name;
        size_t m_version;
        std::list<boost::shared_ptr<Torch::database::Arrayset> > m_arrayset; ///< My arrayset list
        std::map<size_t, boost::shared_ptr<Arrayset> > m_id2arrayset;
        //std::map<std::string, boost::shared_ptr<Relationset> > m_name2relationset;
    };

  }
  /**
   * @}
   */
}

#endif /* TORCH_DATABASE_DATASET_H */
