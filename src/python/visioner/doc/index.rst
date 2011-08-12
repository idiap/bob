.. vim: set fileencoding=utf-8 :
.. Andre Anjos <andre.anjos@idiap.ch>
.. Fri 12 Aug 2011 11:47:43 CEST

.. Index file for the Python Torch::visioner bindings

==========
 Visioner
==========

The Visioner is a library that implements face detection and key point
localization in still images. The Visioner is compiled as an external project
to |project| and we only provide a limited set of interfaces allowing
detection and localization. You can incorporate a call to the Visioner
detection system in 3-ways on your script:

1. Use simple (single) face detection with ``torch.visioner.MaxDetector``:

   In this mode, the Visioner will only detect the most likely face object in
   a given image. It returns a tuple containing the detection bounding box
   (top-left x, top-left y, width, height, score). Here is an usage example:

   .. code-block:: python

      detect_max = torch.visioner.MaxDetector()
      image = torch.core.array.load(...)
      bbox = detect_max(image)

   With this technique you can control: 
    
   * the number of scanning levels;
   * the scale variation in pixels.
        
   Look at the user manual using ``help()`` for operational details.

2. Use simple face detection with ``torch.visioner.Detector``:

   In this mode, the Visioner will return all bounding boxes above a given
   threshold in the image.  It returns a tuple of tuples (descending threshold
   ordered) containing the detection bounding boxes (top-left x, top-left y,
   width, height, score). Here is an usage example:

   .. code-block:: python

      detect = torch.visioner.Detector()
      image = torch.core.array.load(...)
      bboxes = detect(image) #note this is a tuple of tuples

   With this technique you can control: 
    
   * the minimum detection threshold;
   * the number of scanning levels;
   * the scale variation in pixels;
   * the NMS clustering overlapping threshold.
        
   Look at the user manual using ``help()`` for operational details.

3. Use key-point localization with ``torch.visioner.Localizer``:

   In this mode, the Visioner will return a single bounding box and the x and y
   coordinates of every detected land mark in the image. The number of
   landmarks following the bounding box is determined by the loaded model. In
   |project|, we ship with two basic models:

   * ``torch.visioner.DEFAULT_LMODE_EC``: this is the default model used for
     keypoint localization if you don't provide anything to the
     ``torch.visioner.Localizer`` constructor. A call to the function
     operator (``__call__()``) will return the bounding box followed by the
     coordinates of the left and right eyes respectively. The format is
     (top-left b.box x, top-left b.box y, b.box width, b.box height, left-eye
     x, left-eye y, right-eye x, right-eye y).

   * ``torch.visioner.DEFAULT_LMODE_MP``: this is an alternative model that
     can be used for keypoint localization. A call to the function operator
     with a Localizer equipped with this model will return the bounding box
     followed by the coordinates of the eye centers, eye corners, nose tip,
     nostrils and mouth corners (always left and then right coordinates, with
     the x value coming first followed by the y value of the keypoint).

   .. note::

     No scores are returned in this mode.
   
   Example usage:

     .. code-block:: python

        locate = torch.visioner.Localizer()
        image = torch.core.array.load(...)
        bbx_points = locate(image) #note (x, y, width, height, x1, y1, x2, y2...)

   With this technique you can control:
    
   * the number of scanning levels;
   * the scale variation in pixels;
        
   Look at the user manual using ``help()`` for operational details.

Reference Manual
----------------

.. automodule:: torch.visioner
   :members:
