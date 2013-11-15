/**
 * @file bob/visioner/util/timer.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_VISIONER_TIMER_H
#define BOB_VISIONER_TIMER_H

#include <boost/date_time/posix_time/posix_time.hpp>

namespace bob { namespace visioner {

  /**
   * Timer (Boost based, using the date_time library)
   */
  class Timer {

    public: //api

      /**
       * Default constructor, sets the start time as the current time.
       */
      Timer() : m_start(boost::posix_time::microsec_clock::local_time()) {
      }

      /**
       * Resets the start time.
       */
      void restart() {
        m_start = boost::posix_time::microsec_clock::local_time();
      }

      /**
       * Returns the total time in seconds
       */
      double elapsed() const {
        const boost::posix_time::time_duration dt = 
          boost::posix_time::microsec_clock::local_time() - m_start;
        return 0.001 * dt.total_milliseconds();
      }

    private:

      boost::posix_time::ptime	m_start;
  };

}}

#endif // BOB_VISIONER_TIMER_H
