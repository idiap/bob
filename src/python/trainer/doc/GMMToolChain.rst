.. vim: set fileencoding=utf-8 :
.. Francois Moulin <francois.moulin@idiap.ch>

===============
 GMM Toolchain
===============

In this section we present a small example how to train GMM with |project|.

Generate features
-----------------
We assume that you have 2 directories as database: the first one contains pgm
files and the other one contains pos files. We set these two paths in two
variables (in this example we use banca english):

.. code-block:: bash

   $ PGM_PATH=/path/to/pgm/files
   $ POS_PATH=/path/to/pos/files
   $ #$PGM_PATH contains pgm files:
   $ ls $PGM_PATH
   1001_f_g1_s01_1001_en_1.pgm
   1001_f_g1_s01_1001_en_2.pgm
   1001_f_g1_s01_1001_en_3.pgm
   ...
   $ #$POS_PATH contains pos files:
   $ ls $POS_PATH
   1001_f_g1_s01_1001_en_1.pos
   1001_f_g1_s01_1001_en_2.pos
   1001_f_g1_s01_1001_en_3.pos
   ...

We also need a list of input files. If you don't have such a list, you can
generate the list of all database files using the following command:

.. code-block:: bash

   $ find $PGM_PATH -type f -exec basename {} ".pgm" \; > filelist.list
   $ #filelist.list contains the list of database files without extension:
   $ cat filelist.list
   1001_f_g1_s01_1001_en_1
   1001_f_g1_s01_1001_en_2
   1001_f_g1_s01_1001_en_3
   ...

The geometric normalization script needs the list of images and the list of
corresponding pos files. We use the script *filelist.py* to generate these
files from *filelist.list*:

.. code-block:: bash

   $ filelist.py -d $PGM_PATH -e .pgm -c filelist.list > pgm.list
   $ filelist.py -d $POS_PATH -e .pos -c filelist.list > pos.list
   $ #pgm.list contains the full path to images:
   $ cat pgm.list
   /path/to/pgm/files/1001_f_g1_s01_1001_en_1.pgm
   /path/to/pgm/files/1001_f_g1_s01_1001_en_2.pgm
   /path/to/pgm/files/1001_f_g1_s01_1001_en_3.pgm
   ...

Now we are able to generate the features:

.. code-block:: bash

   $ facenorm.py pgm.list pos.list | tantriggs.py | blockDCT.py

This command do the following things:

* Executes *facenorm.py* and puts the result in ./facenorm
* Executes *tantriggs.py* with output of *facenorm.py* as input and puts the result in ./tantriggs
* Executes *blockDCT.py* with output of *tantriggs.py* as input and puts the result in ./blockDCT

The generated features are now in ./blockDCT

Train GMM models
----------------

We train a wold model with all the generated features:

.. code-block:: bash

   $ filelist.py -d blockDCT -e .hdf5 filelist.list | gmm_train.py -o wm.hdf5

We train a model for client 1001 (i.e. using only files beginning with 1001):

.. code-block:: bash

   $ cat filelist.list | grep "^1001" | filelist.py -d blockDCT -e .hdf5 | gmm_adapt.py -p wm.hdf5 -o client1001.hdf5

Finally we test this client model against all our features:

.. code-block:: bash

   $ filelist.py -d blockDCT -e .hdf5 filelist.list | gmm_test.py -m client1001.hdf5 -w wm.hdf5
   blockDCT/1001_f_g1_s01_1001_en_1.hdf5 0.676082991806
   blockDCT/1001_f_g1_s01_1001_en_2.hdf5 1.34133196882
   blockDCT/1001_f_g1_s01_1001_en_3.hdf5 3.35919831582

