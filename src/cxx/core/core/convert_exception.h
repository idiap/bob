/**
 * @file src/cxx/core/core/convert_exception.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch>Laurent El Shafey</a> 
 *
 * @brief Exception for the convert functions
 */

#ifndef TORCH5SPRO_CORE_CONVERT_EXCEPTION_H 
#define TORCH5SPRO_CORE_CONVERT_EXCEPTION_H

#include "core/Exception.h"

namespace Torch { 
  
  namespace core {

    /**
      * A ConvertZeroInputRange is thrown when the specified input range is
      * empty
      */
    class ConvertZeroInputRange: public Exception {

      public:
        ConvertZeroInputRange() throw();
        virtual ~ConvertZeroInputRange() throw();
        virtual const char* what() const throw();

      private:
        mutable std::string m_message;
    };

    /**
      * A ConvertInputAboveMaxRange is thrown when an input value is above 
      * the maximum of the given input range.
      */
    class ConvertInputAboveMaxRange: public Exception {

      public:
        ConvertInputAboveMaxRange(const double v, const double m) throw();
        virtual ~ConvertInputAboveMaxRange() throw();
        virtual const char* what() const throw();

      private:
        double m_val;
        double m_max;
        mutable std::string m_message;
    };

    /**
      * A ConvertInputBelowMinRange is thrown when an input value is below 
      * the minimum of the given input range.
      */
    class ConvertInputBelowMinRange: public Exception {

      public:
        ConvertInputBelowMinRange(const double v, const double m) throw();
        virtual ~ConvertInputBelowMinRange() throw();
        virtual const char* what() const throw();

      private:
        double m_val;
        double m_min;
        mutable std::string m_message;
    };

  }

}

#endif /* TORCH5SPRO_CORE_CONVERT_EXCEPTION_H */
