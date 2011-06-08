===============
 Configuration
===============

This section explains you how to deal with |project| configurable components in C++
or Python code. It is mainly about loading options starting from a command-line
and going through configuring the components in detail.

Commandline Option parsing
--------------------------

While in C++, Boost offers a framework called "program_options" that contains
what you need plus all sorts of additions that include reading options from
files or having more complex input (`more info here`_). The Boost
"program_option" library are linked against ``torch_core``, so
if you are using it directly or indirectly, you don't have to worry about
anything else.

Basic C++ Option Readout
------------------------

Sometimes one needs to read a bunch of information out of a few text or binary
files and configure one or more |project| modules based on that information.
For this kind of job, you can use the ``Torch::config::Configuration`` class.
It allows one to read arbitrary information in text or binary format into a C++
executable.

Example:
 
|project| components that are configurable have special constructors that allow
creation with special settings or ``get``/``set`` methods to adjust and verify
the current settings. Each type designer should decide which of the two
mechanisms is the most interesting for their application. Here is an example
C++ class we are going to use:

.. code-block:: c++

  class ImageResizer {

    //can construct with a width and a height
    ImageResizer(size_t width, size_t height);

    ...

    //consult current values set
    size_t getWidth() const { return m_width; }
    size_t getHeight() const { return m_height; }

  };

This class requires 2 unsigned integers for initialization. 

Reading out options from Text Files
-----------------------------------

|project| provides a type called ``Torch::config::Configuration`` that makes it
easy to load values from a simple text file. A valid |project| configuration
text file with data for the configuration of an object of type ``ImageResizer``
could look like this:

.. code-block:: python

  #comments are started with a hash mark like in this line
  width = 240
  height = 320

And this is how you would lookup the option information from within your C++
application:

.. code-block:: c++

  #include "core/logging.h"
  #include "config/Configuration.h"

  try {
    Torch::config::Configuration cnf("myoptions.txt");
    ImageResizer resize(cnf.get<size_t>("width"), cnf.get<size_t>("height"));
    //use "resize" object...
  }
  catch (std::except& e) {
    Torch::core::error << e.what() << std::endl;
  }
  catch (...) {
    Torch::core::error << "Unknown exception thrown" << std::endl;
  }

You can extend the example above to introduce as many parameters you need in
the file "myoptions.txt". For example, it is conceivable that you may have two
resizers:

.. code-block:: python

  # parameters for resize1
  width1 = 240
  height1 = 320

  # parameters for resize2
  width2 = 480
  height2 = 640

Your C++ program would not change much w.r.t. the ``Configuration`` object
usage:

.. code-block:: c++ 

  #include "core/logging.h"
  #include "config/Configuration.h"

  try {
    Torch::config::Configuration cnf("myoptions.txt");
    ImageResizer resize_small(cnf.get<size_t>("width1"), cnf.get<size_t>("height1"));
    ImageResizer resize_big(cnf.get<size_t>("width2"), cnf.get<size_t>("height2"));
    //use "resize" object...
  }
  catch (std::except& e) {
    Torch::core::error << e.what() << std::endl;
  }
  catch (...) {
    Torch::core::error << "Unknown exception thrown" << std::endl;
  }

The |project| Configuration system also allows you to add more complex
expressions into the game. The following configuration file example will have
the exact same effect as the previous one:

.. code-block:: python
  
  # we only choose one width, the rest is calculated accordingly
  width1 = 240
  width2 = 2*width1
  height1 = 4*width1/3
  height2 = 2*height1

As a matter of fact, you can use any allowable Python expression on it. Let's
suppose I would like my ``Resizer`` object to receive a ``blitz::Array`` for
configuration. In this case, I have to instantiate one in the configuration
file. I use the |project| standard python bindings for such an operation:

.. code-block:: python

  import torch
  resize_dimensions = torch.core.array.uint32_1((240, 320), (2,))

In C++, you retrieve the objects naturally:

.. code-block:: c++

  #include "core/logging.h"
  #include "config/Configuration.h"
  #include <blitz/array.h>

  try {
    Torch::config::Configuration cnf("myoptions.txt");
    blitz::Array<uint32_t,1>& values = cnf.get<blitz::Array<uint32_t,1>&>("resize_dimensions");
    ImageResizer resize(values(0), values(1));
    //use "resize" object...
  }
  catch (std::except& e) {
    Torch::core::error << e.what() << std::endl;
  }
  catch (...) {
    Torch::core::error << "Unknown exception thrown" << std::endl;
  }

This mechanism opens the door for re-using any already existing binding to C++
objects built in |project| or elsewhere. Here is how to load a whole
``Torch::database::Arrayset`` and a ``Torch::database::BinFile`` from an
external file:

.. code-block:: python

  import torch
  mydata = torch.database.BinFile("data-I-produced-before.bin")
  arrayset = torch.database.Arrayset("some-other-data-in-matlab-format.mat")

And you access it just following the patterns explained above:

.. code-block:: c++

  #include "core/logging.h"
  #include "config/Configuration.h"
  #include "database/BinFile.h"
  #include "database/Arrayset.h"

  try {
    Torch::config::Configuration cnf("myoptions.txt");
    Torch::database::BinFile& data = cnf.get<Torch::database::BinFile&>("mydata");
    Torch::database::Arrayset& extras = cnf.get<Torch::database::Arrayset&>("mydata");
    //...
  }
  catch (std::except& e) {
    Torch::core::error << e.what() << std::endl;
  }
  catch (...) {
    Torch::core::error << "Unknown exception thrown" << std::endl;
  }

A question that may be popping on your mind is: but how do I produce data on
those formats? Well, just use the C++ or Python API of (in this case)
``Torch::database::BinFile`` or ``Torch::database::Arrayset`` to save the data.
You choose whatever fits your code best!

You can combine the C++ code snippets above with command-line options as you
see fit. (For C++ commandline option parsing, we recommend the use of
`Boost Program-Options`_.) Suppose, for example, one would like to load 2
configuration files. In this case, the command line to your program may look
like this:

.. code-block:: sh

  # input_data.py: contains a python script enumerating my inputs
  # trained_gmm.py: contains a python script enumeration my trained GMM model parameters
  $ myprogram --input="input_data.py" --model="trained_gmm.py"

Then, within your application, you read it normally:

.. code-block:: c++

  try {
    std::string input_file = process_cmdline_and_get("input");
    Torch::config::Configuration input(input_file);
    Torch::database::Dataset db(input_file.get<std::string>("dataset_name"));

    std::string gmm_file = process_cmdline_and_get("model");
    Torch::config::Configuration gmm_parameters(gmm_file);
    //...
  }
  catch (std::except& e) {
    Torch::core::error << e.what() << std::endl;
  }
  catch (...) {
    Torch::core::error << "Unknown exception thrown" << std::endl;
  }

Actually, you can go beyond just reading out parameters. If you consider the
example above, you could have instantiated the Dataset directly from within the
configuration file! Your choice.

.. Place your links here:
.. _more info here: http://www.boost.org/doc/libs/1_40_0/doc/html/program_options.html
.. _boost program-options: http://www.boost.org/doc/libs/1_40_0/doc/html/program_options.html
