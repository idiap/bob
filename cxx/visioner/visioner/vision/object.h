#ifndef BOB_VISIONER_OBJECT_H
#define BOB_VISIONER_OBJECT_H

#include <string>
#include <vector>

#include "visioner/vision/vision.h"

namespace bob { namespace visioner {

  class Object;
  typedef std::vector<Object>	objects_t;

  // Check if a label is known
  inline bool is_known(const string_t& label)
  {
    return label != "unknown";
  }

  /////////////////////////////////////////////////////////////////////////////////////////
  // Keypoints for object detection & recognition
  /////////////////////////////////////////////////////////////////////////////////////////

  struct Keypoint
  {
    // Constructors
    Keypoint(const string_t& id = string_t(), float x = 0.0, float y = 0.0)
      :   m_id(id), m_point(x, y)
    {
    }
    Keypoint(const string_t& id, const point_t& point)
      :   m_id(id), m_point(point)
    {
    }

    // Attributes
    string_t	m_id;
    point_t		m_point;
  };

  typedef std::vector<Keypoint>	keypoints_t;

  // Compute the maximum overlap of a rectangle with a collection of objects
  scalar_t overlap(const rect_t& reg, const objects_t& objects, int* pwhich = 0);

  // Filter objects by type, pose or ID
  objects_t filter_by_type(const objects_t& objects, const string_t& type);
  objects_t filter_by_pose(const objects_t& objects, const string_t& pose);
  objects_t filter_by_id(const objects_t& objects, const string_t& id);

  /////////////////////////////////////////////////////////////////////////////////////////
  // Object:
  //	- type + pose + ID
  //	- bounding box
  //	- keypoints (if any)
  /////////////////////////////////////////////////////////////////////////////////////////

  class Object
  {
    public:	

      // Constructors
      Object(const string_t& type = string_t(), 
          const string_t& pose = string_t(),
          const string_t& id = string_t(), 
          float bbx_x = 0.0f, float bbx_y = 0.0f, float bbx_w = 0.0f, float bbx_h = 0.0f);
      Object(const string_t& type, 
          const string_t& pose,
          const string_t& id, 
          const rect_t& bbx);

      // Keypoint setup
      void clear();
      void add(const Keypoint& feat);

      // Change geometry
      void move(const rect_t& bbx);		
      void scale(scalar_t factor);
      void translate(scalar_t dx, scalar_t dy);

      // Find some keypoint (if any)
      bool find(const string_t& id, Keypoint& keypoint) const;

      // Load/Save a list of ground truth object positions from/to some file
      static bool load(const string_t& filename, objects_t& objects);
      bool save(const string_t& filename) const;
      static bool save(const string_t& filename, const objects_t& objects);

      // Access functions
      const rect_t& bbx() const { return m_bbx; }
      const keypoints_t& keypoints() const { return m_keypoints; }
      const string_t& type() const { return m_type; }
      const string_t& pose() const { return m_pose; }
      const string_t& id() const { return m_id; }

    private:

      // Attributes		
      string_t	m_type, m_pose, m_id;	// Type + pose + ID
      rect_t		m_bbx;                  // Bounding box
      keypoints_t	m_keypoints;            // Keypoints
  };

}}

#endif // BOB_VISIONER_OBJECT_H
