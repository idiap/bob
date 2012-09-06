/**
 * @file bob/visioner/vision/object.h
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef BOB_VISIONER_OBJECT_H
#define BOB_VISIONER_OBJECT_H

#include <string>
#include <vector>

#include "bob/visioner/vision/vision.h"

namespace bob { namespace visioner {

  // Check if a label is known
  inline bool is_known(const std::string& label) {
    return label != "unknown";
  }

  /**
   * Keypoints for object detection & recognition
   */
  struct Keypoint {
    // Constructors
    Keypoint(const std::string& id = std::string(), float x = 0.0,
        float y = 0.0) 
      : m_id(id), m_point(x, y) { }

    Keypoint(const std::string& id, const QPointF& point)
      : m_id(id), m_point(point) { }

    // Attributes
    std::string	m_id;
    QPointF		m_point;
  };

  /**
   * Object:
   * - type + pose + ID
   * - bounding box
   * - keypoints (if any)
   */
  class Object {

    public:	

      // Constructors
      Object(const std::string& type = std::string(), 
          const std::string& pose = std::string(),
          const std::string& id = std::string(), 
          float bbx_x = 0.0f, float bbx_y = 0.0f, float bbx_w = 0.0f, float bbx_h = 0.0f);
      Object(const std::string& type, 
          const std::string& pose,
          const std::string& id, 
          const QRectF& bbx);

      // Keypoint setup
      void clear();
      void add(const Keypoint& feat);

      // Change geometry
      void move(const QRectF& bbx);		
      void scale(double factor);
      void translate(double dx, double dy);

      // Find some keypoint (if any)
      bool find(const std::string& id, Keypoint& keypoint) const;

      // Load/Save a list of ground truth object positions from/to some file
      static bool load(const std::string& filename, std::vector<Object>& objects);
      bool save(const std::string& filename) const;
      static bool save(const std::string& filename, const std::vector<Object>& objects);

      // Access functions
      const QRectF& bbx() const { return m_bbx; }
      const std::vector<Keypoint>& keypoints() const { return m_keypoints; }
      const std::string& type() const { return m_type; }
      const std::string& pose() const { return m_pose; }
      const std::string& id() const { return m_id; }

    private:

      // Attributes		
      std::string	m_type, m_pose, m_id;	// Type + pose + ID
      QRectF		m_bbx;                  // Bounding box
      std::vector<Keypoint>	m_keypoints;            // Keypoints
  };

  // Compute the maximum overlap of a rectangle with a collection of objects
  double overlap(const QRectF& reg, 
      const std::vector<Object>& objects, int* pwhich = 0);

  // Filter objects by type, pose or ID
  std::vector<Object> filter_by_type(const std::vector<Object>& objects, const std::string& type);
  std::vector<Object> filter_by_pose(const std::vector<Object>& objects, const std::string& pose);
  std::vector<Object> filter_by_id(const std::vector<Object>& objects, const std::string& id);

}}

#endif // BOB_VISIONER_OBJECT_H
