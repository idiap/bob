/**
 * @file core/logging.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This file contains contructions for logging and its configuration
 * within torch. All streams and filters are heavily based on the boost
 * iostreams framework. Manual here:
 *
 * http://www.boost.org/doc/libs/release/libs/iostreams/doc/index.html
 */

#ifndef TORCH_CORE_LOGGING_H
#define TORCH_CORE_LOGGING_H

#include <string>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

namespace Torch {

  namespace core {

    /**
     * The device is what tells the sink where to actually send the messages
     * to. If the Sink does not have a device, the messages are discarded.
     */
    struct Device {
      /**
       * Virtual destructor.
       */
      virtual ~Device();

      /**
       * Writes n bytes of data into this device
       */
      virtual std::streamsize write(const char* s, std::streamsize n) =0;

      /**
       * Closes this device
       */
      virtual void close() {}
    };

    /**
     * Use this sink always in Torch C++ programs. You can configure it to send
     * messages to stdout, stderr, to a file or discard them. 
     */
    class Sink: public boost::iostreams::sink { 

      public:

        /**
         * C'tor, empty, make it "Nullified" sink. All messages are discarded.
         */
        Sink();

        /**
         * Creates a new sink using one of the built-in strategies.
         * - null: discards all messages
         * - stdout: send all messages to stdout
         * - stderr: send all messages to stderr
         * - filename: send all messages to the file named "filename"
         * - filename.gz: send all messagses to the file named "filename.gz",
         *   in compressed format.
         *
         * @param configuration The configuration string to use for this sink
         * as declared above
         */
        Sink(const std::string& configuration);

        /**
         * Copies the configuration from the other Sink
         */
        Sink(const Sink& other);

        /**
         * Intializes with a device.
         */
        Sink(const boost::shared_ptr<Device>& device);

        /**
         * Initializes with an allocated device. Please note that this method
         * will take the ownership of the device and delete it when done.
         */
        Sink(Device* device);

        /**
         * D'tor
         */
        virtual ~Sink();

        /**
         * Resets the current sink and use a new strategy according to the
         * possible settings in `Sink(const std::string& configuration)`.
         */
        void reset(const std::string& configuration);
        void reset(const boost::shared_ptr<Device>& device);

        /**
         * This method will reset the current device by taking the ownership of
         * the input pointer and will delete the currently pointed device.
         */
        void reset(Device* device);

        /**
         * Discards all data input
         */
        virtual std::streamsize write(const char* s, std::streamsize n);

        /**
         * Closes this base sink
         */
        virtual void close();

      private:

        boost::shared_ptr<Device> m_device; ///< Who does the real job.

    };

    /**
     * Usage example: Re-setting the output error stream
     *
     * Torch::core::error->reset("null");
     */
    struct Stream: public boost::iostreams::stream<Sink> {

      /**
       * Constructs the current stream 
       */
      template <typename T> Stream(const T& value)
        : boost::iostreams::stream<Sink>(value) {}

      virtual ~Stream();

      /**
       * Resets the current sink and use a new strategy according to the
       * possible settings in `Sink()`.
       */
      template <typename T> void reset(const T& value) {
        (*this)->reset(value);
      }

    };

    extern Stream debug; ///< default debug stream
    extern Stream info; ///< default info stream
    extern Stream warn; ///< default warning stream
    extern Stream error; ///< default error stream

    /**
     * This method is used by our TDEBUGX macros to define if the current debug
     * level set in the environment is enough to print the current debug
     * message.
     *
     * If TORCH_DEBUG is defined and has an integer value of 1, 2 or 3, this
     * method will return 'true', if the value of 'i' is smaller or equal to
     * the value collected from the environment. Otherwise, returns false.
     */
    bool debug_level(unsigned int i);

  }

}

//returns the current location where the message is being printed
#ifndef TLOCATION
#define TLOCATION __FILE__ << "+" << __LINE__
#endif

//returns the current date and time
#ifndef TNOW
#define TNOW boost::posix_time::second_clock::local_time()
#endif

//an unified marker for the location, date and time
#ifndef TMARKER
#define TMARKER TLOCATION << ", " << TNOW << ": "
#endif

#ifdef TORCH_DEBUG
#define TDEBUG1(v) if (Torch::core::debug_level(1)) { Torch::core::debug << "DEBUG1@" << TMARKER << v << std::endl; }
#define TDEBUG2(v) if (Torch::core::debug_level(2)) { Torch::core::debug << "DEBUG2@" << TMARKER << v << std::endl; }
#define TDEBUG3(v) if (Torch::core::debug_level(3)) { Torch::core::debug << "DEBUG3@" << TMARKER << v << std::endl; }
#else
#define TDEBUG1(v)
#define TDEBUG2(v)
#define TDEBUG3(v)
#endif

#endif /* TORCH_CORE_LOGGING_H */
