/**
 * @file core/python/logging.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * @brief Bindings to re-inject C++ messages into the Python logging module
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <pthread.h>
#include <boost/python.hpp>
#include <boost/shared_array.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

#include <bob/python/ndarray.h>
#include <bob/python/gil.h>
#include <bob/core/logging.h>

/**
 * MT "lock" support was only introduced in Boost 1.35. Before copying this
 * very ugly hack, make sure we are still using Boost 1.34. This will no longer
 * be the case starting January 2011.
 */
#include <boost/version.hpp>
#include <boost/thread/mutex.hpp>
#if ((BOOST_VERSION / 100) % 1000) > 34
#include <boost/thread/locks.hpp>
#else
#warning Disabling MT locks because Boost < 1.35!
#endif

using namespace boost::python;

#define PYTHON_LOGGING_DEBUG 0

static boost::iostreams::stream<bob::core::AutoOutputDevice> static_log("stdout");

/**
 * Objects of this class are able to redirect the data injected into a
 * bob::core::OutputStream to be re-injected in a given python callable object,
 * that is given upon construction. The key idea is that you feed in something
 * like logging.debug to the constructor, for the debug stream, logging.info
 * for the info stream and so on.
 */
struct PythonLoggingOutputDevice: public bob::core::OutputDevice {
  public:
    /**
     * Builds a new OutputDevice from a given callable
     *
     * @param callable A python callable object. Can be a function or an object
     * that implements the __call__() slot.
     */
    PythonLoggingOutputDevice(object callable):
      m_callable(callable), m_mutex(new boost::mutex) {
#if   PYTHON_LOGGING_DEBUG != 0
        pthread_t thread_id = pthread_self();
        static_log << "(0x" << std::hex << thread_id << std::dec
          << ") Constructing new PythonLoggingOutputDevice from callable"
          << std::endl;
#endif
      }

    PythonLoggingOutputDevice(const PythonLoggingOutputDevice& other):
      m_callable(other.m_callable), m_mutex(other.m_mutex) {
#if   PYTHON_LOGGING_DEBUG != 0
        pthread_t thread_id = pthread_self();
        static_log << "(0x" << std::hex << thread_id << std::dec
          << ") Copy-constructing PythonLoggingOutputDevice"
          << std::endl;
#endif
      }

    /**
     * D'tor
     */
    virtual ~PythonLoggingOutputDevice() {
      close();
    }

    /**
     * Closes this stream for good
     */
    virtual void close() {
      m_callable = object(); //set to None
    }

    /**
     * Writes a message to the callable.
     */
    virtual inline std::streamsize write(const char* s, std::streamsize n) {
#if   ((BOOST_VERSION / 100) % 1000) > 35
      boost::lock_guard<boost::mutex> lock(*m_mutex);
#endif
      if (TPY_ISNONE(m_callable)) return 0;
      std::string value(s, n);
      if (std::isspace(value[n-1])) { //remove accidental newlines in the end
        value = value.substr(0, n-1);
      }
      bob::python::gil gil;
#if   PYTHON_LOGGING_DEBUG != 0
      pthread_t thread_id = pthread_self();
      static_log << "(0x" << std::hex << thread_id << std::dec
        << ") Processing message `" << value << "' (size = " << n << ")" << std::endl;
#endif
      m_callable(value);
#if   PYTHON_LOGGING_DEBUG != 0
      m_callable.attr("flush")();
#endif
      return n;
    }

  virtual boost::shared_ptr<bob::core::OutputDevice> clone() const {
    return boost::make_shared<PythonLoggingOutputDevice>(*this);
  }

  private:
    object m_callable; ///< the callable we use to stream the data out.
    boost::shared_ptr<boost::mutex> m_mutex; ///< multi-threading guardian
};

struct message_info_t {
  boost::iostreams::stream<bob::core::AutoOutputDevice>* s;
  std::string message;
  bool exit;
  unsigned int ntimes;
  unsigned int thread_id;
};

static void* log_message_inner(void* cookie) {
  message_info_t* mi = (message_info_t*)cookie;
  if (PyEval_ThreadsInitialized()) {
    static_log << "(thread " << mi->thread_id << ") Python threads initialized correctly for this thread" << std::endl;
  }
  else {
    static_log << "(thread " << mi->thread_id << ") Python threads NOT INITIALIZED correctly for this thread" << std::endl;
  }
  for (unsigned int i=0; i<(mi->ntimes); ++i) {
    static_log << "(thread " << mi->thread_id << ") Injecting message `" << mi->message << " (thread " << mi->thread_id << "; iteration " << i << ")'" << std::endl;
    *(mi->s) << mi->message << " (thread " << mi->thread_id << "; iteration " << i << ")" << std::endl;
    mi->s->flush();
  }
  if (mi->exit) {
    static_log << "(thread " << mi->thread_id << ") Exiting this thread" << std::endl;
    pthread_exit(0);
  }
  if (mi->exit) {
    static_log << "(thread " << mi->thread_id << ") Returning 0" << std::endl;
  }
  return 0;
}

/**
 * A test function for your python bindings
 */
static void log_message(unsigned int ntimes, boost::iostreams::stream<bob::core::AutoOutputDevice>& s, const char* message) {
  bob::python::no_gil unlock;
  message_info_t mi = {&s, message, false, ntimes, 0};
  log_message_inner((void*)&mi);
  static_log << "(thread 0) Returning to caller" << std::endl;
}

/**
 * Logs a number of messages from a separate thread
 */
static void log_message_mt(unsigned int nthreads, unsigned int ntimes,
    boost::iostreams::stream<bob::core::AutoOutputDevice>& s,
    const char* message) {
  bob::python::no_gil unlock;

  boost::shared_array<pthread_t> threads(new pthread_t[nthreads]);
  boost::shared_array<message_info_t> infos(new message_info_t[nthreads]);
  for (unsigned int i=0; i<nthreads; ++i) {
    message_info_t mi = {&s, message, true, ntimes, i+1};
    infos[i] = mi;
  }

  static_log << "(thread 0) Launching " << nthreads << " thread(s)" << std::endl;

  for (unsigned int i=0; i<nthreads; ++i) {
    static_log << "(thread 0) Launch thread " << (i+1) << ": `" << message << "'" << std::endl;
    pthread_create(&threads[i], NULL, &log_message_inner, (void*)&infos[i]);
    static_log << "(thread 0) thread " << (i+1)
      << " == 0x" << std::hex << threads[i] << std::dec
      << " launched" << std::endl;
  }

  void* status;
  static_log << "(thread 0) Waiting " << nthreads << " thread(s)" << std::endl;
  for (unsigned int i=0; i<nthreads; ++i) {
    pthread_join(threads[i], &status);
    static_log << "(thread 0) Waiting on thread " << (i+1) << std::endl;
  }
  static_log << "(thread 0) Returning to caller" << std::endl;
}

static void ostream_open_1(boost::iostreams::stream<bob::core::AutoOutputDevice>& s, const std::string& config) {
  s.open(config);
}

static void ostream_open_2(boost::iostreams::stream<bob::core::AutoOutputDevice>& s, boost::shared_ptr<bob::core::OutputDevice> device) {
  s.open(device);
}

static void ostream_reset_1(boost::iostreams::stream<bob::core::AutoOutputDevice>& s, const std::string& config) {
  s.close();
  s.open(config);
}

static void ostream_reset_2(boost::iostreams::stream<bob::core::AutoOutputDevice>& s, boost::shared_ptr<bob::core::OutputDevice> device) {
  s.close();
  s.open(device);
}

void bind_core_logging() {
  class_<bob::core::OutputDevice, boost::noncopyable>("OutputDevice", "OutputDevices act like sinks for the messages emitted from within C++", no_init);

  class_<PythonLoggingOutputDevice, boost::shared_ptr<PythonLoggingOutputDevice>, bases<bob::core::OutputDevice> >("PythonLoggingOutputDevice", "The PythonLoggingOutputDevice is the default logging class for bob.core.OutputStream objects to be used in python. It diverges the output of logged messages in C++ into the pythonic logging module.", init<object>("Initializes the PythonLoggingOutputDevice with a new callable that will be used to emit messages."))
    .def("__del__", &PythonLoggingOutputDevice::close, (arg("self")), "Resets this stream before calling the C++ destructor")
    ;

  class_<boost::iostreams::stream<bob::core::AutoOutputDevice>, boost::noncopyable>("OutputStream", "The OutputStream object represents a normal C++ stream and is used as the basis for configuring the message output re-direction inside bob.", init<>("Initializes a stream w/o a device (it should be closed by default)"))
    .def(init<const std::string&>((arg("config")), "Initializes this stream with one of the default C++ methods available: ``stdout``, ``stderr``, ``null`` or a filename (if the filename ends in ``.gz``, it will be compressed on the fly)."))
    .def(init<boost::shared_ptr<bob::core::OutputDevice>>((arg("device")), "Initializes this stream with an existing device"))
    .def("open", &ostream_open_1, (arg("self"), arg("config")), "Opens a connection to a new output device as defined by the parameter ``config``.\n\nCommon values for the parameter ``config`` are ``stdout``, ``stderr``, ``null`` or a filename (possibly ending in ``.gz`` for on-the-fly compression).")
    .def("open", &ostream_open_2, (arg("self"), arg("device")), "Connects to an existing device by copying the device information.")
    .def("reset", &ostream_reset_1, (arg("self"), arg("config")), "Closes the current device and then opens a connection to a new output device as defined by the parameter ``config``.\n\nCommon values for the parameter ``config`` are ``stdout``, ``stderr``, ``null`` or a filename (possibly ending in ``.gz`` for on-the-fly compression).")
    .def("reset", &ostream_reset_2, (arg("self"), arg("device")), "Closes the current device and then connects to an existing device by copying the device information.")
    .def("close", &boost::iostreams::stream<bob::core::AutoOutputDevice>::close, (arg("self")), "Closes the this output stream")
    .def("is_open", &boost::iostreams::stream<bob::core::AutoOutputDevice>::is_open, (arg("self")), "Tells if this stream is attached to a device and ready to output data.")
    ;

  //binds the standard C++ streams for logging output to python
  scope().attr("debug") = object(ptr(&bob::core::debug));
  scope().attr("info") = object(ptr(&bob::core::info));
  scope().attr("warn") = object(ptr(&bob::core::warn));
  scope().attr("error") = object(ptr(&bob::core::error));

  //a test function
  def("__log_message__", &log_message, (arg("ntimes"), arg("stream"), arg("message")),
      "Logs a message into Bob's logging system from C++.\n" \
      "\n" \
      "This method is included for testing purposes only and should not be considered part of the Python API for Bob.\n" \
      "\n" \
      "Keyword parameters:\n" \
      "\n" \
      "ntimes\n" \
      "  The number of times to print the given message\n" \
      "\n" \
      "stream\n" \
      "  The stream to use for logging the message. Choose from the streams available at this module.\n" \
      "\n" \
      "message\n" \
      "  A string containing the message to be logged.\n" \
      "\n"
     );

  //a test function
  def("__log_message_mt__", &log_message_mt, (arg("nthreads"), arg("ntimes"), arg("stream"), arg("message")),
      "Logs a message into Bob's logging system from C++ in a separate thread.\n" \
      "\n" \
      "This method is included for testing purposes only and should not be considered part of the Python API for Bob.\n" \
      "\n" \
      "Keyword parameters:\n" \
      "\n" \
      "nthreads\n" \
      "  The total number of threads from which to write messages to the logging system using the C++->Python API.\n" \
      "\n" \
      "ntimes\n" \
      "  The number of times to print the given message, in the same thread\n" \
      "\n" \
      "stream\n" \
      "  The stream to use for logging the message. Choose from the streams available at this module.\n" \
      "\n" \
      "message\n" \
      "  A string containing the message to be logged.\n" \
      "\n"
     );
}
