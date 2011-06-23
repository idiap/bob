/**
 * @file io/InlinedArraysetImpl.h>
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief A class that implements the polimorphic behaviour required when
 * reading and writing blitz arrays to disk or memory.
 */

#ifndef TORCH_IO_INLINEDARRAYSETIMPL_H
#define TORCH_IO_INLINEDARRAYSETIMPL_H

#include <map>
#include <vector>
#include <boost/shared_ptr.hpp>

#include "core/array_type.h"
#include "io/Array.h"

namespace Torch { namespace io { namespace detail {

  /**
   * An implementation of the Arrayset type that holds its contents in memory.
   */
  class InlinedArraysetImpl {

    public:
     
      /**
       * Starts an empty Arrayset. An empty set does not contain any typing
       * information so, you can add any Array to it. Once that is done for the
       * first time, we make sure all other additions will conform to that.
       */
      InlinedArraysetImpl();

      /**
       * Starts a new Arrayset with begin and end iterators to
       * boost::shared_ptr<Torch::io::Array>'s or anything else that
       * add() can accept.
       */
      template <typename T> InlinedArraysetImpl(T begin, T end): 
        m_elementtype(Torch::core::array::t_unknown),
        m_ndim(0),
        m_data()
      {
        m_data.reserve(end-begin);
        for (T it = begin; it != end; ++it) add(*it);
      }

      /**
       * Starts a new Arrayset with an STL conformant iterable container. This
       * can be for example std::vector<Array> or std::list<Array>, as you
       * wish.
       *
       * boost::shared_ptr<Torch::io::Array>'s or anything else that
       * add() can accept.
       */
      template <typename T> InlinedArraysetImpl(const T& iterable):
        m_elementtype(Torch::core::array::t_unknown),
        m_ndim(0),
        m_data()
      {
        m_data.reserve(iterable.end() - iterable.begin());
        for (typename T::const_iterator it = iterable.begin(); it != iterable.end(); ++it) add(*it);
      }

      /**
       * Copy construct by getting an extra reference to somebodies' arrays.
       */
      InlinedArraysetImpl(const InlinedArraysetImpl& other);

      /**
       * Destroyes me
       */
      virtual ~InlinedArraysetImpl();

      /**
       * Copies the content of the other array and gets a reference to the
       * other arrayset's data.
       */
      InlinedArraysetImpl& operator= (const InlinedArraysetImpl& other);

      /**
       * Gets the whole contents for the current arrayset in a vector.
       * The vector index represents the array id while the pointee are
       * boost::shared_ptr to the arrays.
       */
      inline const std::vector<boost::shared_ptr<Torch::io::Array> >& data() const { return m_data; }

      /**
       * Accesses a single array by their id
       */
      const Torch::io::Array& operator[] (size_t id) const;
      Torch::io::Array& operator[] (size_t id);

      /**
       * Accesses a single array by their id, but gets a shared_ptr<Array>
       * instead.
       */
      boost::shared_ptr<const Torch::io::Array> ptr (size_t id) const;
      boost::shared_ptr<Torch::io::Array> ptr (size_t id);

      /**
       * Adds a copy of an array to the list I have. 
       *
       * @return The id assigned to the array.
       */
      size_t add(boost::shared_ptr<const Torch::io::Array> array);
      size_t add(const Torch::io::Array& array);

      /**
       * This is a special version of the add() method that will
       * take a reference to the array you are manipulating instead of the
       * copying. This implies that I'll set up its parent and id myself.
       *
       * @return The id assigned to the array.
       */
      size_t adopt(boost::shared_ptr<Torch::io::Array> array);

      /**
       * Removes the array with a certain id. If the array does not exist, I'll
       * raise an exception. You can check that with exists() if you are not
       * sure before you try.
       */
      void remove(size_t id);

      /**
       * Some informative methods
       */
      inline Torch::core::array::ElementType getElementType() const 
      { return m_elementtype; }
      inline size_t getNDim() const { return m_ndim; }
      inline const size_t* getShape() const { return m_shape; }
      inline size_t size() const { return m_data.size(); }

    private: //checking and typing updating

      /**
       * Checks that the current Arrayset is compatible with the given Array.
       */ 
      void checkCompatibility (const Torch::io::Array& a) const;

      /**
       * Updates the internal typing information of this Arrayset, *iff* it is
       * uninitialized. Otherwise, this is a noop.
       */
      void updateTyping (const Torch::io::Array& a);

    private: //representation
      Torch::core::array::ElementType m_elementtype; ///< Elements' type
      size_t m_ndim; ///< The number of dimensions
      size_t m_shape[Torch::core::array::N_MAX_DIMENSIONS_ARRAY]; ///< The array shape
      std::vector<boost::shared_ptr<Torch::io::Array> > m_data; ///< My data

  };

}}}

#endif /* TORCH_IO_INLINEDARRAYSETIMPL_H */
