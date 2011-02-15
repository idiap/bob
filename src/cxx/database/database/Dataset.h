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
#include "database/Relationset.h"

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
         * id of an Arrayset that already exists, we overwrite the existing
         * one and replace it with this one. 
         *
         * @return The id assigned to the arrayset.
         */
        size_t add(boost::shared_ptr<const Arrayset> arrayset);

        /**
         * Appends a copy of an Arrayset into this Dataset. If you specify the
         * id of an Arrayset that already exists, we overwrite the existing
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
         * Adds and removes Relationsets. If you add() a relationset with the
         * same name as of an existing relationset (within this dataset), that
         * relationset is overwritten. To remove a unexisting relationset is
         * not an error.
         */
        size_t add(boost::shared_ptr<const Relationset> relationset);
        size_t add(const Relationset& relationset);
        void remove(const std::string& name);

        /**
         * Returns my internal arrayset index
         */
        inline const std::map<size_t, boost::shared_ptr<Arrayset> >& arraysetIndex() const { return m_id2arrayset; }

        /**
         * Returns my internal list of arraysets, by insertion order.
         */
        inline const std::list<boost::shared_ptr<Arrayset> >& arraysets () const { return m_arrayset; }

        /**
         * Returns my internal relationset index
         */
        inline const std::map<std::string, boost::shared_ptr<Relationset> >& relationsetIndex() const { return m_name2relationset; }

        /**
         * Returns the Arrayset given a certain, valid, arrayset-id
         *
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
         * @brief Return the Relationset of the given name
         *
         * @warning Please note that if you use that method, scope matters,
         * because the dataset owns the relationsets.
         */
        const Relationset& operator[](const std::string& name) const;
        Relationset& operator[](const std::string& name);

        /**
         * Returns the relationset of the given name
         */
        boost::shared_ptr<const Relationset> ptr(const std::string& name) const;
        boost::shared_ptr<Relationset> ptr(const std::string& name);

        /**
         * Gets the next free id
         */
        size_t getNextFreeId() const;

        /**
         * Consolidates the arrayset ids by resetting the first arrayset to
         * have id = 1, the second id = 2 and so on.
         */
        void consolidateIds();

        /**
         * For a certain relationset, retrives 
         */

        /**
         * Saves the contents of this database at a given file. If the file
         * already exists, it is backed-up (filename + '~') and re-written. If
         * a backup already exists, it is erased before the process begins.
         */
        void save(const std::string& path) const;

      private: //some methods for internal usage.

        /**
         * Runs a full check on the rule consistency and the existance of
         * the respective arrays.
         */
        void checkRelationConsistency() const;

      private:
        std::string m_name;
        size_t m_version;
        std::list<boost::shared_ptr<Torch::database::Arrayset> > m_arrayset; ///< My arrayset list
        std::map<size_t, boost::shared_ptr<Arrayset> > m_id2arrayset;
        std::map<std::string, boost::shared_ptr<Relationset> > m_name2relationset;
    };

  }
  /**
   * @}
   */
}

#endif /* TORCH_DATABASE_DATASET_H */
