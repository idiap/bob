.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri Aug 12 13:36:45 2011 +0200
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

.. Index file for the Python bob::visioner bindings

==========
 Visioner
==========

The Visioner is a library that implements face detection and key point
localization in still images. The Visioner is compiled as an external project
to |project| and we only provide a limited set of interfaces allowing
detection and localization. You can incorporate a call to the Visioner
detection system in 3-ways on your script:

1. Use simple (single) face detection with
   :py:class:`bob.visioner.MaxDetector`:

   In this mode, the Visioner will only detect the most likely face object in
   a given image. It returns a tuple containing the detection bounding box
   (top-left x, top-left y, width, height, score). Here is an usage example:

   .. code-block:: python

      detect_max = bob.visioner.MaxDetector()
      image = bob.io.load(...)
      bbox = detect_max(image)

   With this technique you can control: 
    
   * the number of scanning levels;
   * the scale variation in pixels.
        
   Look at the user manual using :py:func:`help()` for operational details.

2. Use simple face detection with :py:class:`bob.visioner.Detector`:

   In this mode, the Visioner will return all bounding boxes above a given
   threshold in the image.  It returns a tuple of tuples (descending threshold
   ordered) containing the detection bounding boxes (top-left x, top-left y,
   width, height, score). Here is an usage example:

   .. code-block:: python

      detect = bob.visioner.Detector()
      image = bob.io.load(...)
      bboxes = detect(image) #note this is a tuple of tuples

   With this technique you can control: 
    
   * the minimum detection threshold;
   * the number of scanning levels;
   * the scale variation in pixels;
   * the NMS clustering overlapping threshold.
        
   Look at the user manual using :py:func:`help()` for operational details.

3. Use key-point localization with :py:class:`bob.visioner.Localizer`:

   In this mode, the Visioner will return a single bounding box and the x and y
   coordinates of every detected land mark in the image. The number of
   landmarks following the bounding box is determined by the loaded model. In
   |project|, we ship with two basic models:

   * :py:const:`bob.visioner.DEFAULT_LMODEL_EC`: this is the default model
     used for keypoint localization if you don't provide anything to the
     :py:const:`bob.visioner.Localizer` constructor. A call to the function
     operator (:py:meth:`__call__()`) will return the bounding box followed by
     the coordinates of the left and right eyes respectively. The format is
     (top-left b.box x, top-left b.box y, b.box width, b.box height, left-eye
     x, left-eye y, right-eye x, right-eye y).

   * :py:const:`bob.visioner.DEFAULT_LMODEL_MP`: this is an alternative model
     that can be used for keypoint localization. A call to the function
     operator with a Localizer equipped with this model will return the
     bounding box followed by the coordinates of the eye centers, eye corners,
     nose tip, nostrils and mouth corners (always left and then right
     coordinates, with the x value coming first followed by the y value of the
     keypoint).

   .. note::

     No scores are returned in this mode.
   
   Example usage:

     .. code-block:: python

        locate = bob.visioner.Localizer()
        image = bob.io.load(...)
        bbx_points = locate(image) #note (x, y, width, height, x1, y1, x2, y2...)

   With this technique you can control:
    
   * the number of scanning levels;
   * the scale variation in pixels;
        
   Look at the user manual using :py:func:`help()` for operational details.

Applications
------------

We provide 3 applications that are shipped with |project|:

* visioner_facebox.py: This application takes as input either a video or image
  file and can output bounding boxes for faces detected on those files. It uses
  :py:class:`bob.visioner.MaxDetector` for this purpose. You can configure,
  via command-line parameters, the number of scanning levels or the use of a
  user-provided classification model for face localization;

* visioner_fecepoints.py: Is similar to the facebox script, but detects both
  the face and keypoints on the given video or image. You can configure the
  number of scanning levels, or provide external classification and
  localization models. By default, this program will use the default
  localization model provide by |project| which can detect eye-centers;

* visioner_transcode.py: This program can convert text model files to
  alternative formats that can be more compact.

The face detection and keypoint localization programs can, optionally, create
an output video or image with the face bounding box and localized keypoints
drawned, for debugging purposes.

Reference Manual
----------------

.. autodata:: bob.visioner.DEFAULT_DETECTION_MODEL
.. autodata:: bob.visioner.DEFAULT_LOCALIZATION_MODEL
.. autoclass:: bob.visioner.MaxDetector
:members:
:undoc-members:
.. autoclass:: bob.visioner.Detector
:members:
:undoc-members:
.. autoclass:: bob.visioner.Localizer
:members:
:undoc-members:
