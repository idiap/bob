/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue 25 Oct 17:12:06 2011 CEST
 *
 * @brief A torch representation of an Arrayset for a Dataset.
 */

#ifndef TORCH_IO_ARRAYSET_H
#define TORCH_IO_ARRAYSET_H

#include <string>
#include <limits>
#include <boost/shared_ptr.hpp>
#include <blitz/array.h>

#include "io/Array.h"

namespace Torch { namespace io {
    
  /**
   * The arrayset class for a dataset. It is responsible for holding and
   * allowing access to sets of arrays that share the same data type and
   * shape.
   */
  class Arrayset {

    public:

      /**
       * Emtpy array set construction.
       */
      Arrayset ();

      /**
       * Start with all or some arrays in a given file. You can select the
       * start and/or the end. Numbers past the end of the given file are
       * ignored. For example, if a file contains 5 arrays, this constructor
       * will work ok if you leave 'end' on its default (maximum possible
       * unsigned integer).
       */
      Arrayset(boost::shared_ptr<File> file, size_t begin=0,
          size_t end=std::numeric_limits<size_t>::max());

      /**
       * Builds a new array set using all data available in the given file.
       * Please note this does not read the file itself, just create pointers
       * to the several arrays within such a file.
       *
       * @warning: This is a compatibility short cut to create a set object
       * internally to the Array. Don't use this on fresh new code! The
       * correct way to load an array is to use the (file, *) constructors
       * above, for which you have more flexibility.
       */
      Arrayset(const std::string& path);

      /**
       * Copy construct an Arrayset
       */
      Arrayset(const Arrayset& other);

      /**
       * A handle to start up from a std container of Arrays.
       */
      template <typename T> Arrayset (const T& container) {
        for (size_t i=0; i<container.size(); ++i) add(container[i]);
      }

      /**
       * Destructor
       */
      virtual ~Arrayset();

      /**
       * Assign an arrayset
       */
      Arrayset& operator= (const Arrayset& other);

      /**
       * Appends the given Array to the Arrayset - always by reference
       */
      void add (const Array& array);

      /**
       * A shortcut to add a blitz::Array<T,D> (const and non-const)
       */
      template <typename T, int D>
        inline void add(const blitz::Array<T,D>& bz) {
        add(Array(bz));
      }

      template <typename T, int D> inline void add(blitz::Array<T,D>& bz) {
        add(Array(bz));
      }

      /**
       * Sets a specific array to a new value. Note that if the id does not
       * exist, I'll raise an exception. You can check existing ids with
       * id < size().
       */
      void set (size_t id, const Array& array);

      /**
       * A shortcut to set a blitz::Array<T,D> (const and non-const)
       */
      template <typename T, int D> 
        inline void set(size_t id, const blitz::Array<T,D>& bz) {
          set(id, Array(bz));
        }

      template <typename T, int D> 
        inline void set(size_t id, blitz::Array<T,D>& bz) {
          set(id, Array(bz));
        }

      /**
       * Removes an Array with a given id from the Arrayset. If the Array count
       * reaches zero, the internal type information is reset.
       *
       * @warning: This does not remove the array from the originating file,
       * only from this Arrayset object.
       *
       * @warning: Internally, the array set is implemented as a
       * std::vector<Array> which is optimal for element access and, therefore,
       * erasing elements at the middle of the sequence can be slow.
       */
      void remove (size_t id);

      inline const Torch::core::array::typeinfo& type() const { return m_info; }

      inline size_t getNDim() const { return type().nd; }

      inline Torch::core::array::ElementType getElementType() const {
        return type().dtype; 
      }

      inline const size_t* getShape() const { return type().shape; }

      inline const size_t* getStride() const { return type().stride; }

      inline size_t size() const { return m_data.size(); }

      /**
       * Saves this arrayset to an external file, truncating it first in case
       * it exists. This will also unload all the data from memory if that is
       * the case and make all internal arrays point to their new position on
       * the file.
       */
      void save(const std::string& path);

      /**
       * Loads all data from all arrays in memory. Use this to make
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
      Array& operator[] (size_t index);
      const Array& operator[] (size_t index) const;

      template<typename T, int D> const blitz::Array<T,D> get (size_t index) const {
        return (*this)[index].get<T,D>();
      }

      template<typename T, int D> blitz::Array<T,D> cast (size_t index) const {
        return (*this)[index].cast<T,D>();
      }

    private:
      std::vector<Array> m_data; ///< data pointer
      Torch::core::array::typeinfo m_info; ///< information about arrays stored

  };

}}

#endif /* TORCH_IO_ARRAYSET_H */
