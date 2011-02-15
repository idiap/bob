/**
 * @file database/Arrayset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of an Arrayset for a Dataset.
 */

#ifndef TORCH_DATABASE_ARRAYSET_H
#define TORCH_DATABASE_ARRAYSET_H 1

#include <string>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <blitz/array.h>

#include "database/Array.h"
#include "database/InlinedArraysetImpl.h"
#include "database/ExternalArraysetImpl.h"

namespace Torch {   
  /**
   * \ingroup libdatabase_api
   * @{
   */
  namespace database {
    
    /**
     * The arrayset class for a dataset. It is responsible for holding and
     * allowing access to sets of arrays that share the same element type,
     * number of dimensions and shape.
     */
    class Arrayset {

      public:

        /**
         * Constructor. Start a new Arrayset with whatever suits the
         * InlinedArraysetImpl class
         */
        Arrayset (const detail::InlinedArraysetImpl& inlined);

        /**
         * Builds an Arrayset that contains data from a file.
         * You can optionally specify the name of a codec.
         */
        Arrayset(const std::string& filename, const std::string& codec="");

        /**
         * Copy construct an Arrayset
         */
        Arrayset(const Arrayset& other);

        /**
         * Destructor
         */
        virtual ~Arrayset();

        /**
         * Assign an arrayset
         */
        Arrayset& operator= (const Arrayset& other);

        /**
         * Adds a copy of the given Array to the Arrayset. If the array id is
         * zero, I'll assign a proper id to this array. Else, if the arrayset
         * is loaded in memory and if an array with existing id is already
         * inside, I'll overwrite it. If the arrayset is serialized in a file,
         * the array.id property will be ignored and the array will be appended
         * to that file. 
         */
        void add (boost::shared_ptr<const Array> array);
        void add (const Array& array);

        /**
         * Adds a blitz array to the Arrayset.
         */
        void add (const detail::InlinedArrayImpl& array);
        
        /**
         * A shortcut to add a blitz::Array<T,D>
         */
        template <typename T, int D> 
          inline void add(blitz::Array<T,D>& bz) {
          add(detail::InlinedArrayImpl(bz));
        }

        /**
         * Adds a blitz array to the Arrayset, specifying the id.
         * Please note that if this id is repeated, this may raise an
         * exception. If the arrayset is file-based, the id will be ignored.
         */
        void add (const detail::InlinedArrayImpl& array, size_t id);
        
        /**
         * Adds a new external array to the Arrayset
         */
        void add (const std::string& filename, const std::string& codec="", size_t id=0);

        /**
         * Removes an Array with a given id from the Arrayset. Please note that
         * if this arrayset is encoded in an external file, this will trigger
         * loading the whole arrayset into memory, deleting the required array
         * and re-saving the file, which can be time-consuming.
         */
        void remove (const size_t id);

        /**
         * Removes a given Array from the set.
         */
        void remove (const Array& array);
        void remove (boost::shared_ptr<const Array> array);

        /**
         * Returns some information from the current Arrayset
         */
        inline size_t getId () const { return m_id; }
        inline const std::string& getRole() const { return m_role; }
        inline bool isLoaded() const { return m_inlined; }
        
        Torch::core::array::ElementType getElementType() const;
        size_t getNDim() const;
        const size_t* getShape() const;
        size_t getNSamples() const;

        /**
         * Get the filename containing the data if any. An empty string
         * indicates that the data is stored inlined.
         */
        const std::string& getFilename() const;

        /**
         * Get the codec used to read the data from the external file 
         * if any. This will be non-empty only if the filename is non-empty.
         */
        boost::shared_ptr<const ArraysetCodec> getCodec() const; 

        /**
         * Sets the role of the Arrayset
         */
        inline void setRole (const std::string& role) { m_role = role; } 

        /**
         * Saves this arrayset in the given path using the codec indicated (or
         * by looking at the file extension if that is empty). If the arrayset
         * was already in a file it is moved/re-encoded as need to fulfill this
         * request. If the arrayset was in memory, it is serialized, from the
         * data I have in memory and subsequently erased. If the filename
         * specifies an existing file, this file is overwritten.
         */
        void save(const std::string& filename, const std::string&
            codecname="");

        /**
         * If the arrayset is in memory already, this is a noop. If it is in an
         * external file, the file data is read and I become an inlined
         * arrayset. The underlying file containing the data is <b>not</b>
         * erased, we just unlink it from this Arrayset. If you want to read
         * the arrayset data from the file without switching the internal
         * representation of this arrayset (from external to inlined), use the
         * operator[].
         */
        void load();

        /**
         * Inserts, in the given STL-conforming container (has to accept
         * push_back(size_t)), the identities of the Arrays I have in this
         * Arrayset.
         */
        template <typename T> void index(T& container) const;

        /**
         * This method tells if I have a certain array-id registered inside. It
         * avoids me loading files to verify that arrays with that id are
         * available.
         */
        bool exists (size_t id) const;

        /**
         * This set of methods allow you to access the data contained in this
         * Arrayset. Please note that, if this Arrayset is inlined, you will
         * get a reference to the pointed data. Changing it, will be reflected
         * in my internals (would you ever save me again!). If this Arrayset is
         * serialized in a file, you will get a copy of the data. In this last
         * case, changing this array will not affect my internals.
         */
        const Array operator[] (size_t index) const;
        Array operator[] (size_t index);

        template<typename T, int D> const blitz::Array<T,D> get (size_t index) const;
        template<typename T, int D> blitz::Array<T,D> cast (size_t index) const;

        //The next methods are sort of semi-private: Only to be used by the
        //Database loading system. You can adventure yourself, but really not
        //recommended to set the id or the parent of an array. Be sure to
        //understand the consequences.
        
        /**
         * Sets the id. No further checks are performed. This method is not
         * intended for normal programs, just for parsers that need to read
         * data from a file and fill it up.
         */
        inline void setId (size_t id) { m_id = id; }

      private:
        boost::shared_ptr<detail::InlinedArraysetImpl> m_inlined;
        boost::shared_ptr<detail::ExternalArraysetImpl> m_external;
        size_t m_id; ///< This is my id
        std::string m_role; ///< This is my role

    };

    template <typename T> void Arrayset::index(T& container) const {
      if (m_inlined) {
        for (std::list<boost::shared_ptr<Array> >::const_iterator it=m_inlined->arrays().begin(); it!=m_inlined->arrays().end(); ++it) container.push_back((*it)->getId());
      }
      else {
        for (size_t i=0; i<m_external->getNSamples(); ++i) container.push_back(i+1);
      }
    }
        
    template<typename T, int D> const blitz::Array<T,D> Arrayset::get (size_t index) const {
      return (*this)[index].get<T,D>();
    }

    template<typename T, int D> blitz::Array<T,D> Arrayset::cast (size_t index) const {
      return (*this)[index].cast<T,D>();
    }

  }
  /**
   * @}
   */
}

#endif /* TORCH_DATABASE_ARRAYSET_H */
