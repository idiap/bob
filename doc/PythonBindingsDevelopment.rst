.. vim: set fileencoding=utf-8 :
.. Roy Wallace
.. 27 Mar 2012
.. 
.. Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
.. 
.. This program is free software: you can redistribute it and/or modify
.. it under the terms of the GNU General Public License as published by
.. the Free Software Foundation, version 3 of the License.
.. 
.. This program is distributed in the hope that it will be useful,
.. but WITHOUT ANY WARRANTY; without even the implied warranty of
.. MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
.. GNU General Public License for more details.
.. 
.. You should have received a copy of the GNU General Public License
.. along with this program.  If not, see <http://www.gnu.org/licenses/>.

=============================
 Python bindings to C++ code
=============================

Since there are two programming languages (with major differences in their concepts)
there need to be functionality to access classes and functions from on language in
the other one. To access the C++ functions and classes in python code, they need to 
be **bound**.


Binding functions
~~~~~~~~~~~~~~~~~

For many practical reasons we decided to use `Boost.Python`_ to do the bindings. 
`Boost.Python`_ allows to selectively expose C++ functionality to python. Imagine,
you have a C++ function:

.. code-block:: c++

  int foo ( int bar );
 
and you want to export this function to python, you simply write:

.. code-block:: c++

  #import <boost/python.hpp>
  
  void bind_foo(){
  
    // define the function that should be callable in python
    boost::python::def(
      // function name
      "foo", 
      // the function pointer to the function you want to bind
      & foo,
      // parameter list
      (boost::python::arg("bar")), 
      "Description of the function, which will appear as the __doc__ string"
    );
  }
  

The bind_foo function needs to be called during compilation. To 
automatically bind the foo function into <module>, the most simple
way is to add the bind_foo() function call in

.. code-block:: sh

  <bob_src_root>/python/<module>/src/main.cc
  
and make sure that it is compiled by adding the according file into the

.. code-block:: sh

  <bob_src_root>/python/<module>/CMakeLists.txt

There are two more things to note:

* The parameter list of the bound function is given in parentheses. This is 
  especially required, when there is more than one parameter (see example below). 
  Defining default parameters is possible, but this is not shown in this example.

* The above definition of the function does neither specify the type of the 
  parameter nor the return type of the function. The types will be determined during
  runtime, and a python exception will be risen if the conversion does not work.
  For the simple types, `Boost.Python`_ provides automatic python/C++ type conversion.
  For some more complicated types like `NumPy`_ ndarray's, |project| provides functions
  to bind them to `blitz`_ Array's:


.. code-block:: c++

  int bar ( const blitz::Array<double, 2>& image );
  ...
  boost::python::def(
    "bar",
    & bar,
    (boost::python::arg("image")),
    "Description of the bar function."
  );
  
Using this function in python is straightforward. Simply create an ndarray and call it.

.. code-block:: py

  >>> input = numpy.array ( [[1, 2, 3], [4, 5, 6]], dtype = numpy.float64 )
  >>> result = bar ( input )
  
Note that both the data type and the dimensionality of the ndarray must fit to the ones
specified by the C++ foo() function, otherwise python will throw an exception.

Usually, C++ functions operate on given input data and return output data. One important
issue is the memory allocation. To avoid memory leaks, it is most save to allocate the 
memory in the python side and give input and output data as parameters to the function:

.. code-block:: c++

  void foobar ( const blitz::Array<double, 2>& input, blitz::Array<double, 2>& output );

.. code-block:: python
 
  >>> output = numpy.ndarray( (3, 2), dtype = numpy.float64 )
  >>> foobar ( input, output )


Hence, when you write C++ functions that should be bound to python, design the functions
not to allocate and return objects (other than simple types), but rather to operate on
given data.


Binding classes
~~~~~~~~~~~~~~~

To bind a class is a little more complicated. First, you have to specify the class itself, 
the possible constructors, the functions you want to make available, and maybe data members.

The following code should be defined all together, like:

.. code-block:: c++
  
  boost::python::class_<...>(
    ...
  )
  // this definition belongs to the class 
  .def(
    ...
  ); //< This semicolon finalizes the class definition.

Suppose you have the following C++ class:

.. code-block:: c++

  class Foo {
   public:
    // constructor taking one int argument; including default argument
    Foo (int bar = 0);
    
    // exec function which will be bound
    void bound_function(const blitz::Array<double, 2>& input, blitz::Array<double, 2>& output);
    
    // getter and setter for bar_
    int get_bar();
    void set_bar(int new_bar);
    
    // function that will not be available in the python binding
    void other_function_without_binding(...);
    
   private:
    // private variable
    int bar_;
  };
  
So, you start writing the class definition:

.. code-block:: c++

  boost::python::class_<Foo, boost::shared_ptr<Foo> >(
    // The name of the python class; should be identical to the C++ one
    "Foo",
    // The description of the class
    "The Foo class is just used as an example.",
    // and finally the constructor
    boost::python::init < boost::python::optional <int> >(
      // the list of parameters, encapsulated in parentheses
      (boost::python::arg("bar")),
      // the description of the constructor
      "This is the default constructor of Foo with an optional integral bar argument."
    )
  ) // no semicolon here since we want to extend this class

Now, you add the bound_function. Since this will become a python method, the first argument is
always **self** (i.e., **this** in C++). Often, C++ functions are bound to the special __call__
python function, which mimics the **operator ()** behaviour in C++:

.. code-block:: c++

  .def(
    "__call__",
    & Foo::bound_function,
    (boost::python::arg("self"), boost::python::arg("input"), boost::python::arg("output")),
    "Executes the Foo class on the given input and writes the given output"
  ) // No semicolon here either

Also, data members can be bound to the python class. One way to do this is to use the 
getter and setter functions that are provided by the class:

.. code-block:: c++

  .add_property(
    // name of the property
    "bar",
    // pointer to the getter function
    & Foo::get_bar,
    // pointer to the setter function
    & Foo::set_bar,
    // and the description of the variable
    "The bar member defines, how the __call__ function is executed"
  ); // This semicolon ends the class definition

On the python side, this class can now be easily be used:

.. code-block:: py

  >>> # create new Foo object
  >>> foo = Foo ( 20 )
  >>> # set a new bar value
  >>> foo.bar = 10
  >>> # execute the __call__ function
  >>> foo ( input, output )

Finally, to get information about the class, you can use python's help function:

.. code-block:: py

  >>> help ( foo )
  
which will include (amongst some documentation added by `Boost.Python`_) the 
descriptions that you were adding in the bindings.


.. include:: links.rst
.. _`Boost.Python`: http://www.boost.org/libs/python/doc
.. _blitz: http://www.oonumerics.org/blitz/

