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

#include "io/ArrayCodec.h"
#include "io/carray.h"
#include "io/utils.h"
#include "io/filearray.h"

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
         * Builds an Array that contains data from a file. You can optionally
         * specify the name of a codec.
         */
        Array(const std::string& filename, const std::string& codec="");

        /**
         * Refers to the Array data from another array.
         */
        Array(const Array& other);

        /**
         * Destroys this array. 
         */
        virtual ~Array();

        /**
         * Copies data from another array. 
         */
        Array& operator= (const Array& other);

        /**
         * Saves this array in the given path using the codec indicated (or by
         * looking at the file extension if that is empty). If the array was
         * already in a file it is moved/re-encoded as need to fulfill this
         * request. If the array was in memory, it is serialized, from the data
         * I have in memory and subsequently erased. If the filename specifies
         * an existing file, this file is overwritten.
         */
        void save(const std::string& filename, const std::string& codecname="");

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
         * Sets the current data to the given array
         */
        void set(boost::shared_ptr<buffer> data); //refer to data
        void set(const buffer& data); //copy data

        inline const typeinfo& type() const {
          return (m_inlined)?m_inlined->type(): m_external->type(); 
        }

        inline size_t getNDim() const { return type().nd; }
        
        inline Torch::core::array::ElementType getElementType() const { 
          return type().dtype; 
        }

        inline const size_t* getShape() const { return type().shape; }
        
        inline const size_t* getStride() const { return type().stride; }

        /**
         * Get the filename containing the data if any. An empty string
         * indicates that the data is stored inlined.
         */
        const std::string& getFilename() const;

        /**
         * Get the codec used to read the data from the external file 
         * if any. This will be non-empty only if the filename is non-empty.
         */
        boost::shared_ptr<const ArrayCodec> getCodec() const;

        /**
         * Get the flag indicating if the array is loaded in memory
         */
        inline bool isLoaded() const { return m_inlined; }


        /******************************************************************
         * Blitz Array specific manipulations
         ******************************************************************/

        /**
         * Starts a new array with in-memory content, refers to the data
         */
        template <typename T, int N> Array(blitz::Array<T,N>& data):
          m_inlined(new carray(boost::make_shared<blitz::Array<T,N> >(data))) {}

        /**
         * Starts a new array with in-memory content, copies the data.
         */
        template <typename T, int N> Array(const blitz::Array<T,N>& data):
          m_inlined(new carray(data)) {}

        /**
         * If the array is already in memory, we return a copy of it in the
         * type you wish (just have to get the number of dimensions right!). If
         * it is in a file, we load it and return a copy of the loaded data.
         */
        template <typename T, int N> blitz::Array<T,N> cast() const {
          if (!m_inlined) {
            const typeinfo& info = m_external->type();
            carray tmp(info);
            m_external->load(tmp);
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
            const typeinfo& info = m_external->type();
            carray tmp(info);
            m_external->load(tmp);
            return tmp.get<T,N>();
          }
          else return Torch::io::wrap<T,N>(*m_inlined);
        }

        /**
         * A handle to simplify your life with blitz::Array<>'s
         */
        template <typename T, int N> 
          inline void set(const blitz::Array<T,N>& bzarray) {
            if (!m_inlined) {
              carray tmp(boost::make_shared<blitz::Array<T,N> >(bzarray));
              m_external->save(tmp);
            }
            else {
              //we copy the data only once!
              set(boost::make_shared<carray>(bzarray));
            }
        }

        /**
         * A handle to simplify your life with blitz::Array<>'s
         */
        template <typename T, int N> 
          inline void set(boost::shared_ptr<blitz::Array<T,N> >& bzarray) {
            if (!m_inlined) {
              carray tmp(bzarray);
              m_external->save(tmp);
            }
            else {
              //no data copying...
              set(carray(bzarray));
            }
        }

      private: //representation
        boost::shared_ptr<buffer> m_inlined;
        boost::shared_ptr<filearray> m_external;
    };

  } //closes namespace io

} //closes namespace Torch

#endif /* TORCH_IO_ARRAY_H */
