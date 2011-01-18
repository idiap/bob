/**
 * @file src/core/python/src/logging.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Bindings to re-inject C++ messages into the Python logging module
 */

#include <boost/python.hpp>

#include "core/logging.h"

using namespace boost::python;

/**
 * Objects of this class are able to redirect the data injected into a
 * Torch::core::OutputStream to be re-injected in a given python callable object,
 * that is given upon construction. The key idea is that you feed in something
 * like logging.debug to the constructor, for the debug stream, logging.info
 * for the info stream and so on.
 */
struct PythonLoggingOutputDevice: public Torch::core::OutputDevice {
  public:
    /**
     * Builds a new OutputDevice from a given callable
     *
     * @param callable A python callable object. Can be a function or an object
     * that implements the __call__() slot.
     */
    PythonLoggingOutputDevice(object callable): m_callable(callable) {}

    /**
     * D'tor
     */
    virtual ~PythonLoggingOutputDevice() {}

    /**
     * Writes a message to the callable.
     */
    virtual inline std::streamsize write(const char* s, std::streamsize n) {
      std::string value(s);
      if (std::isspace(value[n-1])) { //remove accidental newlines in the end
        value = value.substr(0, n-1);
      }
      m_callable(value); 
      return n;
    }

  private:
    object m_callable; ///< the callable we use to stream the data out.
};

/**
 * A test function for your python bindings 
 */
static void log_message(Torch::core::OutputStream& s, const std::string& message) {
  s << message << std::endl;
}

void bind_core_logging() {
  class_<Torch::core::OutputDevice, boost::shared_ptr<Torch::core::OutputDevice>, boost::noncopyable>("OutputDevice", "OutputDevices act like sinks for the messages emitted from within C++", no_init); 

  class_<PythonLoggingOutputDevice, boost::shared_ptr<PythonLoggingOutputDevice>, bases<Torch::core::OutputDevice> >("PythonLoggingOutputDevice", "The PythonLoggingOutputDevice is the default logging class for torch.core.OutputStream objects to be used in python. It diverges the output of logged messages in C++ into the pythonic logging module.", init<object>("Initializes the PythonLoggingOutputDevice with a new callable that will be used to emit messages."));

  class_<Torch::core::OutputStream, boost::shared_ptr<Torch::core::OutputStream> >("OutputStream", "The OutputStream object represents a normal C++ stream and is used as the basis for configuring the message output re-direction inside Torch.", init<>("Constructs a new OutputStream using no parameters. Ignores any input received."))
    .def(init<const std::string&>((arg("configuration")), "Initializes this stream with one of the default C++ methods available: stdout, stderr, null or a filename (if the filename ends in '.gz', it will be compressed on the fly)."))
    .def(init<boost::shared_ptr<Torch::core::OutputDevice> >((arg("device")), "Constructs a new OutputStream using the given existing OutputDevice."))
    .def("reset", &Torch::core::OutputStream::reset<const std::string>, (arg("self"), arg("configuration")), "Resets the current stream to use a new method for output instead of the currently configured.")
    .def("reset", &Torch::core::OutputStream::reset<boost::shared_ptr<Torch::core::OutputDevice> >, (arg("self"), arg("device")), "Resets the current stream to use a new method for output instead of the currently configured. This version of the API allows you to pass an existing OutputDevice to be used for output data.")
    .def("log", &log_message, (arg("self"), arg("message")), "This method logs an arbitrary message to the current log stream")
    ;

  //binds the standard C++ streams for logging output to python
  scope().attr("debug") = &Torch::core::debug;
  scope().attr("info") = &Torch::core::info;
  scope().attr("warn") = &Torch::core::warn;
  scope().attr("error") = &Torch::core::error;

  //a test function
}
