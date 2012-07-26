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
