#ifndef BOB_VISIONER_ITERATORS_H
#define BOB_VISIONER_ITERATORS_H

#include <vector>

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // 1D fixed increment iterator
  /////////////////////////////////////////////////////////////////////////////////////////

  template <typename T>
    class DeltaIterator
    {
      public:

        // Constructor
        DeltaIterator(T* pdata, std::size_t delta)
          :       m_pdata(pdata), m_delta(delta)
        {
        }

        // Next element
        void operator++()
        {
          m_pdata += m_delta;
        }

        // Test convergence
        bool operator==(const DeltaIterator<T>& other) const
        {
          return m_pdata == other.m_pdata;
        }
        bool operator!=(const DeltaIterator<T>& other) const
        {
          return m_pdata != other.m_pdata;
        }

        // Access functions
        T& operator*() { return *m_pdata; }                        

      private:

        // Attributes
        T*              m_pdata;
        std::size_t         m_delta;
    };

  template <typename T>
    class ConstDeltaIterator
    {
      public:

        // Constructor
        ConstDeltaIterator(const T* pdata, std::size_t delta)
          :       m_pdata(pdata), m_delta(delta)
        {
        }

        // Next element
        void operator++()
        {
          m_pdata += m_delta;
        }

        // Test convergence
        bool operator==(const DeltaIterator<T>& other) const
        {
          return m_pdata == other.m_pdata;
        }
        bool operator!=(const DeltaIterator<T>& other) const
        {
          return m_pdata != other.m_pdata;
        }

        // Access functions
        const T& operator*() const { return *m_pdata; }          

      private:

        // Attributes
        const T*        m_pdata;
        std::size_t         m_delta;
    };

}}

#endif // BOB_VISIONER_ITERATORS_H
