/**
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file contains contructions for logging and its configuration
 * within bob. All streams and filters are heavily based on the boost
 * iostreams framework. Manual here:
 * http://www.boost.org/doc/libs/release/libs/iostreams/doc/index.html
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_CORE_LOGGING_H
#define BOB_CORE_LOGGING_H

#include <string>
#include <boost/iostreams/stream.hpp>
#include <boost/iostreams/concepts.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/date_time/posix_time/posix_time.hpp>
#include <boost/date_time/posix_time/posix_time_io.hpp>

/**
 * @addtogroup CORE core
 * @brief Core module API
 */
namespace bob {

  /**
   * @ingroup CORE
   */
  namespace core {
    /**
     * @ingroup CORE
     * @{
     */

    /**
     * @brief The device is what tells the sink where to actually send the
     * messages to. If the AutoOutputDevice does not have a device, the
     * messages are discarded.
     */
    struct OutputDevice {
      /**
       * @brief Virtual destructor.
       */
      virtual ~OutputDevice();

      /**
       * @brief Writes n bytes of data into this device
       */
      virtual std::streamsize write(const char* s, std::streamsize n) =0;

      /**
       * @brief Closes this device
       */
      virtual void close() {}
    };

    /**
     * @brief Use this sink always in bob C++ programs. You can configure it
     * to send messages to stdout, stderr, to a file or discard them.
     */
    class AutoOutputDevice: public boost::iostreams::sink {

      public:

        /**
         * @brief C'tor, empty, discards all input.
         */
        AutoOutputDevice();

        /**
         * @brief Creates a new sink using one of the built-in strategies.
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
        AutoOutputDevice(const std::string& configuration);

        /**
         * @brief Intializes with a device.
         */
        AutoOutputDevice(boost::shared_ptr<OutputDevice> device);

        /**
         * @brief D'tor
         */
        virtual ~AutoOutputDevice();

        /**
         * @brief Forwards call to underlying OutputDevice
         */
        virtual std::streamsize write(const char* s, std::streamsize n);

        /**
         * @brief Closes this base sink
         */
        virtual void close();

      private:

        boost::shared_ptr<OutputDevice> m_device; ///< Who does the real job.

    };

    // standard streams
    extern boost::iostreams::stream<AutoOutputDevice> debug;
    extern boost::iostreams::stream<AutoOutputDevice> info;
    extern boost::iostreams::stream<AutoOutputDevice> warn;
    extern boost::iostreams::stream<AutoOutputDevice> error;

    /**
     * @brief This method is used by our TDEBUGX macros to define if the
     * current debug level set in the environment is enough to print the
     * current debug message.
     *
     * If BOB_DEBUG is defined and has an integer value of 1, 2 or 3, this
     * method will return 'true', if the value of 'i' is smaller or equal to
     * the value collected from the environment. Otherwise, returns false.
     */
    bool debug_level(unsigned int i);

    /**
     * @brief Chooses the correct temporary directory to use, like this:
     *
     * - The environment variable TMPDIR, if it is defined. For security reasons
     *   this only happens if the program is not SUID or SGID enabled.
     * - The directory /tmp.
     */
    std::string tmpdir();

    /**
     * @brief Returns the full path of a temporary file in tmpdir().
     *
     * @param extension The desired extension for the file
     */
    std::string tmpfile(const std::string& extension=".hdf5");

    /**
     * @}
     */
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

#ifdef BOB_DEBUG
#define TDEBUG1(v) if (bob::core::debug_level(1)) { bob::core::debug << "DEBUG1@" << TMARKER << v << std::endl; }
#define TDEBUG2(v) if (bob::core::debug_level(2)) { bob::core::debug << "DEBUG2@" << TMARKER << v << std::endl; }
#define TDEBUG3(v) if (bob::core::debug_level(3)) { bob::core::debug << "DEBUG3@" << TMARKER << v << std::endl; }
#else
#define TDEBUG1(v)
#define TDEBUG2(v)
#define TDEBUG3(v)
#endif

#endif /* BOB_CORE_LOGGING_H */
