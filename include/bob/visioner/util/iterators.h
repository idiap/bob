/**
 * @file bob/visioner/util/iterators.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_VISIONER_ITERATORS_H
#define BOB_VISIONER_ITERATORS_H

#include <vector>

namespace bob { namespace visioner {

  /** 
   * 1D fixed increment iterator
   */
  template <typename T> class DeltaIterator {

    public:

      // Constructor
      DeltaIterator(T* pdata, size_t delta)
        : m_pdata(pdata), m_delta(delta) {
      }

      // Next element
      void operator++() {
        m_pdata += m_delta;
      }

      // Test convergence
      bool operator==(const DeltaIterator<T>& other) const {
        return m_pdata == other.m_pdata;
      }

      bool operator!=(const DeltaIterator<T>& other) const {
        return m_pdata != other.m_pdata;
      }

      // Access functions
      T& operator*() { return *m_pdata; }                        

    private: //representation

      T* m_pdata;
      size_t m_delta;

  };

  template <typename T> class ConstDeltaIterator {

    public:

      // Constructor
      ConstDeltaIterator(const T* pdata, size_t delta)
        : m_pdata(pdata), m_delta(delta) {
      }

      // Next element
      void operator++() {
        m_pdata += m_delta;
      }

      // Test convergence
      bool operator==(const DeltaIterator<T>& other) const {
        return m_pdata == other.m_pdata;
      }

      bool operator!=(const DeltaIterator<T>& other) const {
        return m_pdata != other.m_pdata;
      }

      // Access functions
      const T& operator*() const { return *m_pdata; }

    private: // representation
      const T* m_pdata;
      size_t m_delta;
  };

}}

#endif // BOB_VISIONER_ITERATORS_H
