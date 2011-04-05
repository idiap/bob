.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.dos.anjos@gmail.com>
.. Mon  4 Apr 22:15:19 2011 

=================================
 Using Blitz Arrays in |project|
=================================

You can use `Blitz`_ arrays as tensor primitives. For a complete description on
how to use `Blitz`_ arrays, please refer to the `Blitz Manual`_. In this
guide we just give a brief introduction to some of Blitz array properties for a
quick start.

.. _supported-element-types:

A note on supported element types
---------------------------------

Blitz arrays a flexible containers that support a different range of base
element types and up to 11 dimensions. For most of |project|, we limit the
support of provided code to manipulate arrays *up to 4 dimensions* and using
the following element types. These element types and ranks should cover all of
the functionality needed by machine learning algorithms and signal processing
utilities.

+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| Common name  |   C++                       | `NumPy`_     | Meaning                                                      |
+==============+=============================+==============+==============================================================+
| bool         | bool                        | bool         | a boolean                                                    |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int8         | int8_t                      | int8         | a signed integer, with 8 bits                                |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int16        | int16_t                     | int16        | a signed integer, with 16 bits                               |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int32        | int32_t                     | int32        | a signed integer, with 32 bits                               |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| int64        | int64_t                     | int64        | a signed integer, with 64 bits                               |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint8        | uint8_t                     | uint8        | a unsigned integer, with 8 bits                              |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint16       | uint16_t                    | uint16       | a unsigned integer, with 16 bits                             |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint32       | uint32_t                    | uint32       | a unsigned integer, with 32 bits                             |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| uint64       | uint64_t                    | uint64       | a unsigned integer, with 64 bits                             | 
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| float32      | float                       | float32      | a real number with single precision, with 32 bits            |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| float64      | double                      | float64      | a real number with single precision, with 64 bits            |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| complex64    | std::complex<float>         | complex64    | a complex number with each part using each a 32-bit float    |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+
| complex128   | std::complex<double>        | complex128   | a complex number with each part using each a 64-bit float    |
+--------------+-----------------------------+--------------+--------------------------------------------------------------+

.. dropped support:
  +--------------+-----------------------------+--------------+--------------------------------------------------------------+
  | ~~float128~~ | ~~long double~~             | ~~float128~~ | ~~a real number with quadruple precision, with 128 bits~~    |
  +--------------+-----------------------------+--------------+--------------------------------------------------------------+
  | -complex256- | -std::complex<long double>- | -complex256- | -a complex number with each part using each a 128-bit float- |
  +--------------+-----------------------------+--------------+--------------------------------------------------------------+

We have decided not to support ``long double`` and ``std::complex<long double>``. An
explanation might be found `here`_.

Construction
------------

Blitz arrays are templates, so you must the element types and the number of
dimensions are part of the type. Here is how to declare arrays:

.. code-block:: c++
    
   #include <blitz/array.h>
   blitz::Array<short, 1> empty();
   blitz::Array<short, 3> my(3, 3, 3); //uninitialized, 3x3x3
   blitz::Array<short, 2> other(2, 2); //uninitialized, 2x2

   //Blitz also allows powerful initializations:
   other = {1, 2, 3, 4};

   //Blitz and the use of expressions:
   blitz::Array<short, 2> other_squared(other*other);

Element access
--------------

To access elements in a Blitz array, one uses the parenthesis operator:

.. code-block:: c++

  blitz::Array<short, 2> my();
  my = 0; //a complete fill of all positions
  short val = my(0, 0); 
  const short& v3 = my(1, 0); //gets a reference
  my(0, 1) = v3; //sets the value internally

Selection, transposition and narrowing
--------------------------------------

Blitz arrays use an underlying memory system to map array positions into
contiguously allocated regions. Two or more arrays can share the underlying
memory. To create a modified version of an array, pointing to the same memory
and therefore having access to the same data, use the following:

.. code-block:: c++

  blitz::Array<short, 2> my(100, 100);
  blitz::Array<short, 1> selected(my(blitz::Range::all(), 50));
  blitz::Array<short, 2> narrowed(my(blitz::Range::all(), blitz::Range(50, 80));
  blitz::Array<short, 2> transposed(my);
  transposed.transposeSelf(1, 0); //transposes dimensions 1 <-> 0

Please note that Blitz arrays are not limited to these operations, you can
select any number of transposition dimensions in a single call or narrow in
multiple dimensions. Please refer to the `Blitz manual`_ for more details.

Loading and saving Blitz arrays in text mode with the Blitz adapter
-------------------------------------------------------------------

The default implementation of Blitz++ overloads the streaming operators ``>>``
and ``<<``. However, this mechanism does not provide information such as the
number of dimensions or the type of the multiarray, which might cause tricky
problems. For this purpose, a Blitz adapter class was created.

An example, on how to use this adapter to save and read a array from a stream
(in this case a file), is shown below:

.. code-block:: c++

  typedef blitz::Array<double,2> BAd2;

  // Save the blitz array to a file
  BAd2 bl1(2,2); // create a multiarray
  bl1 = 1.1, 0.5, 
        0.3, 1.4; // initialize the multiarray
  Torch::core::BlitzAdapter<BAd2> X(bl1); // create and initialize the BlitzAdapter
  std::ofstream out_d("multiarray.blitz"); // An output file stream
  out_d << X; // save the blitzarray
  out_d.close(); // close the output stream

  // Read the blitz array from a file
  BAd2 bl2(2,2); // create a multiarray
  std::ifstream in_d("multiarray.blitz"); // An input file stream
  Torch::core::BlitzAdapter<BAd2> Y(bl2); // create and initialize the BlitzAdapter
  in_d >> Y; // read the blitz array and put it in bl2
  in_d.close(); // close the input stream

The type stored in an output file stream corresponds to the result of the C++
``typeid()`` function. The resulting string might depend on the environment
(compiler). If the type needs to be ignored when reading a blitz array from a
file, the creation of a Blitz adapter should be done as follow: 

.. code-block:: c++
  
  typedef blitz::Array<double,2> BAd2; BAd2 bl(2,2); // creates a array
  Torch::core::BlitzAdapter<BAd2> Y(bl, false); // creates a Blitz adapter which will not perform type checking 
                                                // (second argument of the constructor set to false, whereas default value is true)

You can make use of our ``Torch::core::In/OutputStream`` to input and output
data in compressed format (as with gzip), which might save you some space.
Please read InputOutput for details.

Loading and saving Blitz arrays in binary mode with the BinFile class
---------------------------------------------------------------------

The following example shows how to save and load blitz arrays into and from a
binary file:

.. code-block:: c++

  #include "database/BinFile.h"

  int main() {
    // Create and initialize two blitz arrays
    blitz::Array<uint64_t,1> bl1(5);
    blitz::Array<uint64_t,1> bl2(5);
    bl1 = 1, 2, 3, 4, 5;
    bl2 = 6, 7, 8, 9, 10; 

    // save them to a binary file
    Torch::database::BinFile out("blitz.bin", Torch::core::BinFile::out);
    out.write( bl1);
    out.write( bl2);
    // close the output stream
    out.close();

    // Read the uint64_t data in the binary file and 
    // cast it into a double blitz array
    blitz::Array<double,1> bl3;
    Torch::core::BinFile in("blitz.bin", Torch::core::BinFile::in);
    // 1st array
    in.read<double,1>( bl3);
    std::cout << bl3 << std::endl;

    // 2nd array
    in.read<double,1>( bl3);
    std::cout << bl3 << std::endl;

    // End of file is now reached.
    // Alternative: Direct access to a particular array in the file
    in.read<double,1>( 0, bl3); // 0 is the index of the first array
    std::cout << bl3 << std::endl;

    in.read<double,1>( 1, bl3); // 1 is the index of the second array
    std::cout << bl3 << std::endl;

    // close the input stream
    in.close();

    return 0;
  }

.. Place your references down here
.. _blitz: http://www.oonumerics.org/blitz/
.. _blitz manual: http://www.oonumerics.org/blitz/docs/blitz.html
.. _numpy: http://numpy.scipy.org/
.. _here: http://www.idiap.ch/software/torch5spro/wiki/TorchDatabaseBindata#Binaryfileformatheader
