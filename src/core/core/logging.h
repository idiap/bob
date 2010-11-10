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
     * to. If the AutoOutputDevice does not have a device, the messages are discarded.
     */
    struct OutputDevice {
      /**
       * Virtual destructor.
       */
      virtual ~OutputDevice();

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
     * The device is what tells the source where to actually read the messages
     * from. If the AutoInputDevice does not have a device, the messages are 
     * discarded.
     */
    struct InputDevice {
      /**
       * Virtual destructor.
       */
      virtual ~InputDevice();

      /**
       * Reads n bytes of data from this device
       */
      virtual std::streamsize read(char* s, std::streamsize n) =0;

      /**
       * Closes this device
       */
      virtual void close() {}
    };

    /**
     * Use this sink always in Torch C++ programs. You can configure it to send
     * messages to stdout, stderr, to a file or discard them. 
     */
    class AutoOutputDevice: public boost::iostreams::sink { 

      public:

        /**
         * C'tor, empty, discards all input.
         */
        AutoOutputDevice();

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
        AutoOutputDevice(const std::string& configuration);

        /**
         * Copies the configuration from the other AutoOutputDevice
         */
        AutoOutputDevice(const AutoOutputDevice& other);

        /**
         * Intializes with a device.
         */
        AutoOutputDevice(const boost::shared_ptr<OutputDevice>& device);

        /**
         * D'tor
         */
        virtual ~AutoOutputDevice();

        /**
         * Resets the current sink and use a new strategy according to the
         * possible settings in 
         * `AutoOutputDevice(const std::string& configuration)`.
         */
        void reset(const std::string& configuration);
        void reset(const boost::shared_ptr<OutputDevice>& device);

        /**
         * Forwards call to underlying OutputDevice
         */
        virtual std::streamsize write(const char* s, std::streamsize n);

        /**
         * Closes this base sink
         */
        virtual void close();

      private:

        boost::shared_ptr<OutputDevice> m_device; ///< Who does the real job.

    };

    /**
     * Use this source always in Torch C++ programs. You can configure it to
     * read messages from stdin or a file.
     */
    class AutoInputDevice: public boost::iostreams::source { 

      public:

        /**
         * C'tor, empty, reads from stdin.
         */
        AutoInputDevice();

        /**
         * Creates a new source using one of the built-in strategies.
         * - stdin: reads from the standard input
         * - filename: read all messages from the file named "filename"
         * - filename.gz: read all messagses from the file named "filename.gz",
         *   in compressed format.
         *
         * @param configuration The configuration string to use for this source
         * as declared above
         */
        AutoInputDevice(const std::string& configuration);

        /**
         * Copies the configuration from the other AutoInputDevice
         */
        AutoInputDevice(const AutoInputDevice& other);

        /**
         * Intializes with a device.
         */
        AutoInputDevice(const boost::shared_ptr<InputDevice>& device);

        /**
         * D'tor
         */
        virtual ~AutoInputDevice();

        /**
         * Resets the current source and use a new strategy according to the
         * possible settings in 
         * `AutoInputDevice(const std::string& configuration)`.
         */
        void reset(const std::string& configuration);
        void reset(const boost::shared_ptr<InputDevice>& device);

        /**
         * Forwards call to underlying InputDevice
         */
        virtual std::streamsize read(char* s, std::streamsize n);

        /**
         * Closes this base source
         */
        virtual void close();

      private:

        boost::shared_ptr<InputDevice> m_device; ///< Who does the real job.

    };

    /**
     * Usage example: Re-setting the output error stream
     *
     * Torch::core::error->reset("null");
     */
    struct OutputStream: public boost::iostreams::stream<AutoOutputDevice> {

      /**
       * Constructs an empty version of the stream, uses NullOutputDevice.
       */
      OutputStream()
        : boost::iostreams::stream<AutoOutputDevice>() {}

      /**
       * Copy construct the current stream.
       */
      OutputStream(const OutputStream& other)
        : boost::iostreams::stream<AutoOutputDevice>(*const_cast<OutputStream&>(other)) {}

      /**
       * Constructs the current stream 
       */
      template <typename T> OutputStream(const T& value)
        : boost::iostreams::stream<AutoOutputDevice>(value) {}

      virtual ~OutputStream();

      /**
       * Resets the current sink and use a new strategy according to the
       * possible settings in `AutoOutputDevice()`.
       */
      template <typename T> void reset(const T& value) {
        (*this)->reset(value);
      }

    };

    /**
     * Create streams of this type to input data into Torch5spro
     */
    struct InputStream: public boost::iostreams::stream<AutoInputDevice> {

      /**
       * Constructs an empty version of the stream, uses NullInputDevice.
       */
      InputStream()
        : boost::iostreams::stream<AutoInputDevice>() {}

      /**
       * Copy construct the current stream.
       */
      InputStream(const InputStream& other)
        : boost::iostreams::stream<AutoInputDevice>(*const_cast<InputStream&>(other)) {}

      /**
       * Constructs the current stream 
       */
      template <typename T> InputStream(const T& value)
        : boost::iostreams::stream<AutoInputDevice>(value) {}

      virtual ~InputStream();

      /**
       * Resets the current sink and use a new strategy according to the
       * possible settings in `AutoInputDevice()`.
       */
      template <typename T> void reset(const T& value) {
        (*this)->reset(value);
      }

    };

    extern OutputStream debug; ///< default debug stream
    extern OutputStream info; ///< default info stream
    extern OutputStream warn; ///< default warning stream
    extern OutputStream error; ///< default error stream

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

    /**
     * Chooses the correct temporary directory to use, like this:
     *
     * - The environment variable TMPDIR, if it is defined. For security reasons
     *   this only happens if the program is not SUID or SGID enabled.
     * - The directory /tmp.
     */
    std::string tmpdir();

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
