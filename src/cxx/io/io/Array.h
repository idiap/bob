/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Tue  4 Oct 23:13:45 2011 CEST
 *
 * The Array is the basic unit containing data in a Dataset 
 */

#ifndef TORCH_IO_ARRAY_H 
#define TORCH_IO_ARRAY_H

#include <cstdlib>
#include <stdint.h>
#include <boost/shared_ptr.hpp>
#include <boost/weak_ptr.hpp>
#include <blitz/array.h>

#include "io/File.h"
#include "io/carray.h"
#include "io/utils.h"

namespace Torch {

  namespace io {

    /**
     * The array class for a dataset. The Array class acts like a manager for
     * the underlying data (blitz::Array<> in memory or serialized in file).
     */
    class Array {

      public:

        /**
         * Starts a new array with in-memory content, copies the data.
         */
        Array(const buffer& data);

        /**
         * Starts a new array with in-memory content, refers to the data.
         */
        Array(boost::shared_ptr<buffer> data);

        /**
         * Reads all the data from the file into this Array.
         */
        Array(boost::shared_ptr<File> file);

        /**
         * Builds an Array that contains data from a file, specific data from
         * the file is loaded using this constructor.
         */
        Array(boost::shared_ptr<File> file, size_t index);

        /**
         * Refers to the Array data from another array.
         */
        Array(const Array& other);

        /**
         * Builds a new array using the first array available in the file path
         * given.
         *
         * @warning: This is a compatibility short cut to create a File object
         * internally to the Array. Don't use this on fresh new code! The
         * correct way to load an array is to use the (file, index) constructor
         * above.
         */
        Array(const std::string& path);

        /**
         * Destroys this array. 
         */
        virtual ~Array();

        /**
         * Copies data from another array.
         */
        Array& operator= (const Array& other);

        /**
         * If the array is in-memory nothing happens. If the array is in a
         * file, the file data is read and I become an inlined array. The
         * underlying file containing the data is <b>not</b> erased, we just
         * unlink it from this Array. 
         */
        void load();

        /**
         * This is a non-templated version of the get() method that returns a
         * generic array, used for typeless manipulations. 
         */
        boost::shared_ptr<buffer> get() const;

        /**
         * Sets the current data to the given array.
         */
        void set(boost::shared_ptr<buffer> data); //refer to data
        void set(const buffer& data); //copy data

        inline const typeinfo& type() const {
          return (m_inlined)?m_inlined->type(): external_type();
        }

        inline size_t getNDim() const { return type().nd; }
        
        inline Torch::core::array::ElementType getElementType() const {
          return type().dtype; 
        }

        inline const size_t* getShape() const { return type().shape; }
        
        inline const size_t* getStride() const { return type().stride; }

        inline size_t getIndex() const { return m_index; }

        inline void setIndex(size_t i) { m_index = i; }

        inline bool loadsAll() const { return m_loadsall; }

        inline void setLoadsAll() { m_loadsall = true; }
        inline void unsetLoadsAll() { m_loadsall = false; }

        /**
         * Get the filename containing the data if any. An empty string
         * indicates that the data is stored inlined.
         */
        const std::string& getFilename() const;

        /**
         * Get the codec used to read the data from the external file 
         * if any. This will be non-empty only if the filename is non-empty.
         */
        boost::shared_ptr<const File> getCodec() const;

        /**
         * Get the flag indicating if the array is loaded in memory
         */
        inline bool isLoaded() const { return m_inlined; }

        /**
         * Dumps the array contents on a file. The file is truncated if it
         * exists. No more data may possibly written to this file after calling
         * this method.
         */
        void save(const std::string& path);


        /******************************************************************
         * Blitz Array specific manipulations
         ******************************************************************/

        /**
         * Starts a new array with in-memory content, refers to the data
         */
        template <typename T, int N> Array(blitz::Array<T,N>& data):
          m_inlined(boost::make_shared<carray>(boost::make_shared<blitz::Array<T,N> >(data))) { }

        /**
         * Starts a new array with in-memory content, copies the data.
         */
        template <typename T, int N> Array(const blitz::Array<T,N>& data): 
          m_inlined(boost::make_shared<carray>(data)) { }

        /**
         * If the array is already in memory, we return a copy of it in the
         * type you wish (just have to get the number of dimensions right!). If
         * it is in a file, we load it and return a copy of the loaded data.
         */
        template <typename T, int N> blitz::Array<T,N> cast() const {
          if (!m_inlined) {
            const typeinfo& info = external_type();
            carray tmp(info);
            if (m_loadsall) m_external->array_read(tmp);
            else m_external->arrayset_read(tmp, m_index);
            return tmp.cast<T,N>();
          }
          else return Torch::io::cast<T,N>(*m_inlined);
        }

        /**
         * If the array is already in memory, we return a reference to it. If
         * it is in a file, we load it and return a reference to the loaded
         * data.
         */
        template <typename T, int N> blitz::Array<T,N> get() const {
          if (!m_inlined) {
            const typeinfo& info = external_type();
            carray tmp(info);
            if (m_loadsall) m_external->array_read(tmp);
            else m_external->arrayset_read(tmp, m_index);
            return tmp.get<T,N>();
          }
          else return Torch::io::wrap<T,N>(*m_inlined);
        }

        /**
         * A handle to simplify your life with blitz::Array<>'s
         */
        template <typename T, int N> 
          void set(blitz::Array<T,N>& bzarray) {
            boost::shared_ptr<blitz::Array<T,N> > 
              sbz(new blitz::Array<T,N>(bzarray));
            set(sbz);
        }

        /**
         * A handle to simplify your life with blitz::Array<>'s
         */
        template <typename T, int N> 
          void set(const blitz::Array<T,N>& bzarray) {
            //we copy the data only once!
            set(boost::make_shared<carray>(bzarray));
        }

        /**
         * A handle to simplify your life with blitz::Array<>'s
         */
        template <typename T, int N> 
          void set(boost::shared_ptr<blitz::Array<T,N> >& bzarray) {
            //no data copying...
            set(boost::make_shared<carray>(bzarray));
        }

      private: //useful methods

        inline const io::typeinfo& external_type() const {
          return m_loadsall? m_external->array_type() : m_external->arrayset_type();
        }

      private: //representation
        boost::shared_ptr<buffer> m_inlined;
        boost::shared_ptr<File> m_external;
        ptrdiff_t m_index; ///< position on a file.
        bool m_loadsall; ///< loads all data in file in one shot.
    };

  } //closes namespace io

} //closes namespace Torch

#endif /* TORCH_IO_ARRAY_H */
