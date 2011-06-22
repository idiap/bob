.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Mon  4 Apr 22:35:13 2011 

=============================
 Error reporting and logging
=============================

If you are programming a |project| extension or just doing code maintenance,
read this documentation to get familiarized with the way C++ extensions in
|project| are supposed to report problems to callers.

Use cases
---------

In this guide, we cover use cases for error reporting and logging tools in the
following domains:

* From within libraries: Applicable when you are developing code that will be
  called by somebody else. You don't control the ``main()`` execution loop of
  the application - i.e. 99% of the time
* At the application level: You are creating a top-level script or application
  and control the main loop of execution

In each of the situations, these may occur:

* You want to report something that is not a problem
* You want to report a non-fatal problem
* You want to report a fatal problem

Before reading this manual, be sure to understand in which of the two-levels
you are developing or modifying code. If unsure, please read :ref:`case-study`.

Error reporting
---------------

.. note::

  The Boost documentation includes `some guidelines on error reporting and
  handling`_ which are probably clearer than this guide. Please read that
  before proceeding.

Errors can be classified in two main groups: fatal or non-fatal. Non-fatal
errors should be reported through the object's API and documented so callers
can detect problems and take actions. Fatal errors should be reported with
exception throwing. If you are unsure on how to classify your problems, please
read :ref:`case-study` bellow. 

Exceptions in Torch are arranged in an exception tree, all inheriting from
``Torch::core::Exception``, which in turn inherits from ``std::exception``. You
should not throw a ``Torch::core::Exception`` ever. They are there just to
provide a convenient way to catch any non-covered Torch exceptions in a
try/catch clause. If you want to report a specific problem in your code, that
you think may be solvable upstream, you can inherit from
``Torch::core::Exception`` to make your problem specific:

.. code-block:: c++

  class MySpecialProblem: public Torch::core::Exception {
    
    MySpecialProblem(type1 param1, type2 param2)  throw(): m_param1(param1), m_param2(param2) { }

    MySpecialProblem(const MySpecialProblem& other) throw(): m_param1(other.m_param1), m_param2(other.m_param2) { }

    virtual ~MySpecialProblem() throw() { }

    //any other getters and setters you may want in this part  

    virtual const char* what() const throw() {
      //possibly format error string.
    } 
  }

And throw it when appropriate within your code. Please make sure to document
your code stating you may throw exceptions, but do '''not''' use exception
specification (`here is some lengthier explanation of why`_). Another important
tip is to **avoid formatting an error message string before ``what()`` is
called**. The reason is related to memory allocation: formatting a string is
very often a memory intensive operatio. If the exception was throw because
memory was lacking, allocating even more memory will not help. To be on the
safe side, just defer the formatting to until ``what()`` is called - at this
point, hopefully, more system resources will be freed and there is less risk
for problems. Take your parameters and compose a printable message that helps
users understand what was the '''cause''' of the problem. Protect the message
formatting with try/catch clauses to avoid surprises.

If your inherited exception type does not provide a ``what()`` method, the
default message will be printed.

You can find examples in `io/Exception.h`_ and
`io/src/Exception.cc`_ examples on how to extend the Exception system to
become more specific and format messages.

Logging
-------

Error logging is a two-sided problem. On one side, libraries want to say things
about events that occurred inside them. On the other side, there is the
application main loop that needs to be able to choose what is allowed to come
out and eventually reach the human seating in front the screen or through a web
page where results are displayed. Normally, these sides have contradicting
requirements: libraries want to report most they can so it is easy to spot
problems if they ever occur. Application main loops want to shut-up all the
libraries because so much is written to the screen it becomes difficult to see
what is really important.

Here are some advices:

* If you are a library developer avoid to inject messages into the Torch
  logging system. Messages are computationally expensive. They require strings
  to be built, possibly formatted and memory to be allocated. If you do that
  often, it will slow-down your processing;
* If you want to inject a message for debugging purposes, use one of the
  available ``TDEBUG*`` macro variants. They will help you inject strings that
  will be **compiled out** in optimized (release) builds which avoids wasting
  time with memory allocation and string formatting;
* If you absolutely need to inject a message, use the report streams choosing
  the appropriate level: info, error or warning.

How to inject TDEBUG messages
-----------------------------

.. code-block:: c++

  #include "core/logging.h"
  ...

  TDEBUG1("This is a debug message, level 1");
  TDEBUG2("This is an info message, level 2");
  TDEBUG3("This is an info message, level 3");

  ...
  // Another example with a variable
  double evil_value = 666;
  TDEBUG1("Evil value is: " << evil_value);

These messages will be compiled out in release builds. If you want to display
them, you have to compile your code in debug mode or use the debug build from
one of the installed releases. After setting up, make sure that the
``TORCH_DEBUG`` environment variable is set on your environment with one of the
3 values:

* **1**: In this case only messages with level 1 will be displayed;
* **2**: In this case only messages with level 1 and 2 will be displayed;
* **3**: In this case, all debug messages will be displayed.

Again: If you need debugging aid, consider using a real debugger. Debug
messages like the ones in this section are meant for other people to make sure
your code is executing as expected. Not for you.

How to inject more important messages
-------------------------------------

Report messages are the ones that will be injected in the report system
irrespectively on the type of compilation. There are 3 levels of messages you
can choose from:

* ``info``: This is the most basic level, it informs the user of things which
  happen inside a method;
* ``warning``: Messages of this type inform users of potential problems or
  problems that have been **correctly** treated by your code and should not
  be there in the normal code flow;
* ``error``: Messages of this type inform users of errors that could not be
  treated by your code and should not exist in the normal code flow. 

Here is some sample C++ code:

.. code-block:: c++

  #include "core/logging.h"
  ...

  Torch::core::info << "This is an INFO message" << std::endl;
  Torch::core::warn << "This is a WARNING message" << std::endl;
  Torch::core::error << "This is an ERROR message" << std::endl;

We also provide marker macros that help you marking the output so you know
where it comes from. We use those consistently when you use our ``TDEBUG*``
macros. You can also make use of those for your messages:

.. code-block:: c++

  #include "core/logging.h"
  ...

  Torch::core::warn << TMARKER << "This message will be marked" << std::endl;

This should printout something like:

.. code-block:: sh

  /path/to/my/file.cc+27, 2010-Nov-08 15:08:10: This message will be marked

Please note that these messages streamed directly into one of the report
streams will **never** be compiled out, so be careful not to penalize the
execution speed more than needed. Also understand that it is the application
developer that ultimately configures the final destination of report messages.
It is possible, for instance, that somebody decides to throw away "info"-style
messages within their application. **Do not rely on messages to have problems
fixed!** If you think something is wrong and should never happen, it may be
more appropriate to throw an exception. Please read the section named "Case
Study" bellow to understand when to make use of exceptions as an error
reporting mechanism for your code.

How to configure streams
------------------------

If you are an application developer, it maybe upon you to decide how to stream
information from the code you are calling into the appropriate stream. The
Torch defaults are:

* debug-style messages and info message go to ``stdout``;
* warning and error messages are re-directed to ``stderr``.

You can change that behavior by adjusting the output sinks in the following
way:

.. code-block:: c++

  #include "core/logging.h"
  ...
  //diverges, globally, debug messages to go to stderr
  Torch::core::debug.reset("stderr");

  //suppresses, globally, all info messages
  Torch::core::info.reset("null");

It is illegal to use these calls in library code, only ``main()`` loops should
be able to configure how to diverge the streams as its developer is the
ultimate responsible on deciding how to display the messages.

.. _case-study:

Case Study
----------

Library developer
=================

When you are coding for |project|, it is more likely you are adding
functionality to it in the form of new classes or functions that can be used in
somebody's applications. There are a few things you should keep in your mind at
this time:

* You don't know what is the application execution context, so don't use any
  constructions that assume standard inputs, output or error streams are
  present;
* You don't know how much capable of solving problems is your caller. Don't
  assume that problems like for example memory exhaustion are unsolvable and
  you should call exit if a call to ``malloc()`` or ``new`` fails.

As a library developer you should **only** report the best way you can and let
the caller take action. There are two main mechanisms to report **problems**
in a C/C++ or Python routine:

* Exception throwing
* Status codes

The use of each is very specific to each situation and which to use should be
chosen carefully. To make a decision, you should analyze how the code you are
writing is supposed to be called and which kinds of problems should lead to a
fatal (exception throwing path) or non-fatal (status return) actions. The main
concern here is execution speed. When you throw an exception, a gigantic
machinery for stack unrolling is activated which will slow-down the code
execution. The advantage of exception throwing is that you can contextual
information back from the callee that you don't get with a normal status
return. So, trading the execution speed for information is not a problem if the
situation is truly exceptional - i.e. happens only when attention is required
by developer to fix code problems. At this time you **do** want to have more
information.

Exception throwing is **not** recommended to cope with normal (say "legal")
errors that are allowed to occur during the execution of your routine though.
For example, suppose a routine that receives a vector of integers and counts
how many of them are prime numbers.

.. code-block:: c++

  /***
   * This method returns the number of primes within a std::vector.
   * Note: is_prime() is a non-declared predicate...
   *
   * @param input The input vector from where to count primes
   *
   * Please note a SegmentationFaultException may be thrown on NULL input.
   ***/
  int count_primes(const std::vector<unsigned int>* input) {
    if (!input) throw SegmentationFaultException();
    return std::count_if(input->begin(), input->end(), is_prime);
  }

Needless to say, it is possible that the input vector is empty in which case
the return value would be zero. What would happen if the input vector is
``NULL``? In this case, a segmentation fault would occur and an exception is
raised to indicate that is a fatal condition for this function call. If not
caught at higher execution levels, this exception will cause the program
ultimately to terminate. By looking at the code and the API one notices the
developer has decided that passing a NULL input is a fatal problem and requires
the developer of the bracketing code to take action to fix the input.

Later on the development of the project and by inspecting the situation and
**understanding how people finally use this function** we may decide
otherwise and assume that it is legal to specify a NULL input, in which case we
return ``-1`` to indicate the problem. Here is the modified call:

.. code-block:: c++

  /***
   * This method returns the number of primes within a std::vector.
   * Note: is_prime() is a non-declared predicate...
   *
   * @param input The input vector from where to count primes
   *
   * @return The number of primes in input -or- `-1`, if input is NULL
   ***/
  int count_primes(const std::vector<unsigned int>* input) {
    if (!input) return -1;
    return std::count_if(input->begin(), input->end(), is_prime);
  }

Now, if the input is NULL the function will return ``-1`` to indicate a
problem.  Even if we have not changed the API of the method, any bracketing
code should now be aware of the newly introduced convention (i.e. if returns
``-1``, there was a ``NULL`` input) and take action if that is required. There
is no right or wrong. Every situation needs to be analyzed and a design
decision taken.

Bracketing exceptions
=====================

If you decide you can fix a fatal error that went wrong with one of your
callees, you can bracket the code with ``try/catch`` clauses. Taking the
example above, using the version of ``count_primes()`` that throws exceptions:

.. code-block:: c++

  try {
    value = count_primes(my_input_vector_pointer);
  }
  catch (SegmentationFaultException& e) {
    action(e);
  }
  //continue doing some other stuff.

There is a big difference between the two situations we are studying
(library/application developer). If you are inside yet another library that is
making use of ``count_primes()``, you have to fix the problem or re-throw
another (or even the same) exception. This is what ``action()`` is supposed to
do. If you are at the application main loop, you can decide to report the
exception to the standard error stream and exit, for example.

At most instances you don't want to do anything at all and just let the
exception through, in which case you would not need to bracket the call with
the ``try/catch`` clauses. Only use ``try/catch`` if you need to take an action
on the problem.

Application developer
=====================

The application developer is normally the last resource layer in the stack and
controls what needs to be done if an exception is received. Many times, no
action is also a good action! If you don't bracket your code with ``try/catch``
clauses, exception throwing by one of your callees will call ``terminate()``
and get you a core dump you can debug problems from, with the precise stack
trace that lead you to this problem.

The application developer is also responsible for determining what to do with
messages that may be logged by its callees into the Torch logging system. As
the application master, you can decide to suppress all messages or let them be
printed to screen (the default), if you can afford them. Be sure to familiarize
yourself with our logging API for the language you are programming at.

.. Place your references here:
.. _some guidelines on error reporting and handling: http://www.boost.org/community/error_handling.html
.. _here is some lengthier explanation of why: http://www.gotw.ca/publications/mill22.htm
.. _io/Exception.h: http://www.idiap.ch/software/torch5spro/browser/src/cxx/io/io/Exception.h
.. _io/src/Exception.cc: http://www.idiap.ch/software/torch5spro/browser/src/cxx/io/src/Exception.cc
