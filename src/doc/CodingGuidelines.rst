===================
 Coding Guidelines
===================

If you plan to write code that should be incorporated in |project|, here are a
few websites to get you started. Don't take everything to the letter, but
observe good practices from the software that already exists and derive from
there.

C++ Coding Guidelines:

* `C++ FAQ`_: A must if you want to write reliable code
* `C++ Coding Standard`_
* `The Google C++ Style Guide`_
* `The CERN C++ Style`_
* Book: *C++ Coding Standards by Herb Sutter* and Andrei Alexandrescu

Here are some C++ tips we have found useful:

* Optimization: **Don't**. If you still want to do it, please make sure you
  fully read :doc:`OptimizingTorch`.
* DRY: means `Don't repeat yourself`_. You should apply this principle at
  **all** moments while you are coding to avoid repeating bugs or only
  partially fixing them
* Let it crash: Don't use (ever) ``exit()`` outside the scope of ``main()``
* Prefer stack allocation: Dynamic memory allocation and high-performance
  computing are two things that don't go well together. That is for a good
  reason: dynamic memory allocation can be quite slow in many circumstances.
  You should avoid it when possible
* You are not better than the compiler: Do not replace the compiler! Every time
  we do a runtime check that could have been executed by a compiler at
  compilation time, we are just wasting resources.
* Avoid breaking encapsulation: If you have designed your code right, you
  should not need to find out hidden information about objects.
* Avoid global variables: Global variables are evil
* Make methods concise: This one is obvious and most would agree that small,
  readable code is easier to understand. The problem is that it takes a bit
  more of time to write it down and people end up not doing it
* Use references instead of pointers: The rule is simple: if you can use
  references, don't use pointers.
* Use ``const``: The use of the ``const`` keyword says a lot about an API. It
  determines what is the fate of objects that are passed in or out through a
  call and give the programmer some assurance on the variable's fate. It allows
  compilers to more aggressively optimize underlying code.
* Document your code: It does not matter how many times you say it, it is
  always better to say it again: document, document, document! Undocumented
  code is useless.

Python Coding Guidelines:

* `The Google Python Style Guide`_

.. Place your references here:

.. _`c++ faq`: http://www.parashift.com/c++-faq-lite/
.. _`c++ coding standard`: http://www.possibility.com/Cpp/CppCodingStandard.html
.. _`the google c++ style guide`: http://google-styleguide.googlecode.com/svn/trunk/cppguide.xml
.. _`the cern c++ style`: http://pst.cern.ch/HandBookWorkBook/Handbook/Programming/CodingStandard/c++standard.pdf
.. _`the google python style guide`: http://google-styleguide.googlecode.com/svn/trunk/pyguide.html
.. _`don't repeat yourself`: http://en.wikipedia.org/wiki/Don't_repeat_yourself
