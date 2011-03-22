====================
How to install Torch
====================

To install Torch you need first to set your mind on what to install. You can choose between a released stable version from TorchDistribution or checkout and build yourself following the TorchCompilation wiki. 

**WARNING**: *Make sure to read  and install all requirements defined in* :doc:`TorchDependencies`, *prior to running Torch applications.*

Grab a tarball and change into the directory of your choice, let's say `WORKDIR`:

.. code-block:: sh

  $ cd WORKDIR
  $ wget http://cicatrix01.idiap.ch:8000/torch/chrome/site/main/torch-nightly-latest.tar.gz
  $ tar xvfz torch-nightly-latest.tar.gz

To setup a working environment at your present shell, do:

.. code-block:: sh

  $ torch-x.y/bin/shell.py

These instructions will create a clone of your current shell with the Torch environment *appended* so your applications can find our libraries and executables. Would you need to use the debug version of the code, add a `-d` (or `--debug`) to the source command line:

.. code-block:: sh

  $ torch-x.y/bin/shell.py -d

You can use scripts like these to keep programs and setup together. Or else, to simplify batch job submission.

Debug environments can be useful if you need to run a debugger session or to send us a core dump with embedded debugging symbols.

-----------------------------------------------
Starting programs in Torch-enabled environments
-----------------------------------------------

Sometimes you just want to execute one particular program in a Torch-enabled environment and, when you leave it, you have your previous environment back. You can use the setup program `shell.py` for that purpose as well. For example, if you want to start the python interpreter within a Torch-enabled environment just do:

.. code-block:: sh

  $ torch-x.y/bin/shell.py -- python

When you leave the python prompt, your environment will be back to the previous state.

----------------------------------------
Creating complete self-contained scripts
----------------------------------------

You can also create scripts that can run standalone and require no configuration. For those, `/usr/bin/env` is your dearest friend. Here is an example of a python script that executes in a Torch-enabled environment:

.. code-block:: python

  #/usr/bin/env /WORKDIR/torch-x.y/bin/shell.py -- python
  import torch
  print torch.core.array.int16_2()

Here is another one that is just a shell script using `bash`:

.. code-block:: sh

  #/usr/bin/env /WORKDIR/torch-x.y/bin/shell.py -- bash
  echo $TORCH_VERSION

