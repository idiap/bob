/**
 * @file visioner/visioner/util/timer.h
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

#ifndef BOB_VISIONER_TIMER_H
#define BOB_VISIONER_TIMER_H

#include <boost/date_time/posix_time/posix_time.hpp>

namespace bob { namespace visioner {

  /////////////////////////////////////////////////////////////////////////////////////////
  // Timer (Boost based, using the date_time library)
  /////////////////////////////////////////////////////////////////////////////////////////
  class Timer
  {
    public:

      Timer() : m_start(boost::posix_time::microsec_clock::local_time()) 
    {
    }

      void restart()
      {
        m_start = boost::posix_time::microsec_clock::local_time();
      }

      double elapsed() const
      {
        const boost::posix_time::time_duration dt = 
          boost::posix_time::microsec_clock::local_time() - m_start;
        return 0.001 * dt.total_milliseconds();
      }

    private:

      boost::posix_time::ptime	m_start;
  };

}}

#endif // BOB_VISIONER_TIMER_H
