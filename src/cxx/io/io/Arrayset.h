/**
 * @file io/Arrayset.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief A torch representation of an Arrayset for a Dataset.
 */

#ifndef TORCH_IO_ARRAYSET_H
#define TORCH_IO_ARRAYSET_H 1

#include <string>
#include <boost/shared_ptr.hpp>
#include <blitz/array.h>

#include "io/Array.h"
#include "io/InlinedArraysetImpl.h"
#include "io/ExternalArraysetImpl.h"

namespace Torch {   
  /**
   * \ingroup libio_api
   * @{
   */
  namespace io {
    
    /**
     * The arrayset class for a dataset. It is responsible for holding and
     * allowing access to sets of arrays that share the same element type,
     * number of dimensions and shape.
     */
    class Arrayset {

      public:

        /**
         * Emtpy constructor. Start a new Arrayset with an empty inlined
         * arrayset
         */
        Arrayset ();

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
         * Adds a copy of the given Array to the Arrayset. This will
         * potentially trigger file re-writing in case the arrayset is
         * serialized in an external file.
         *
         * @return The assigned id for this array.
         */
        size_t add (boost::shared_ptr<const Array> array);
        size_t add (const Array& array);
        size_t add (const detail::InlinedArrayImpl& array);
        size_t add (const std::string& filename, const std::string& codec="");
        
        /**
         * A shortcut to add a blitz::Array<T,D>
         */
        template <typename T, int D> 
          inline size_t add(blitz::Array<T,D>& bz) {
            return add(detail::InlinedArrayImpl(bz));
        }

        /**
         * Sets a specific array to a new value. Note that if the id does not
         * exist, I'll raise an exception. You can check existing ids with
         * id < size().
         */
        void set (size_t id, boost::shared_ptr<const Array> array);
        void set (size_t id, const Array& array);
        void set (size_t id, const detail::InlinedArrayImpl& array);
        void set (size_t id, const std::string& filename, const std::string& codec="");

        /**
         * A shortcut to set a blitz::Array<T,D>
         */
        template <typename T, int D> 
          inline void set(size_t id, blitz::Array<T,D>& bz) {
            set(id, detail::InlinedArrayImpl(bz));
        }

        /**
         * Removes an Array with a given id from the Arrayset. Please note that
         * if this arrayset is encoded in an external file, this will trigger
         * loading the whole arrayset into memory, deleting the required array
         * and re-saving the file, which can be time-consuming.
         *
         * @return The current size of this arrayset in number of samples
         */
        void remove (const size_t id);

        /**
         * Returns some information from the current Arrayset
         */
        inline bool isLoaded() const { return m_inlined; }
        
        Torch::core::array::ElementType getElementType() const;
        size_t getNDim() const;
        const size_t* getShape() const;
        size_t size() const;

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

        /**
         * This is a non-templated version of the get() method that returns a
         * generic array, used for typeless manipulations. 
         *
         * @warning You do NOT want to use this!
         */
        detail::InlinedArraysetImpl get() const;

      private:
        boost::shared_ptr<detail::InlinedArraysetImpl> m_inlined;
        boost::shared_ptr<detail::ExternalArraysetImpl> m_external;

    };

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

#endif /* TORCH_IO_ARRAYSET_H */
