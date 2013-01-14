/**
 * @file visioner/cxx/object.cc
 * @date Fri 27 Jul 13:58:57 2012 CEST
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * @brief This file was part of Visioner and originally authored by "Cosmin
 * Atanasoaei <cosmin.atanasoaei@idiap.ch>". It was only modified to conform to
 * Bob coding standards and structure.
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <fstream>
#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string/trim.hpp>

#include "bob/visioner/vision/object.h"

namespace bob { namespace visioner {

  Object::Object(const std::string& type, const std::string& pose, const std::string& id, 
      float bbx_x, float bbx_y, float bbx_w, float bbx_h)
    :	m_type(type), m_pose(pose), m_id(id), m_bbx(bbx_x, bbx_y, bbx_w, bbx_h)
  {		
  }

  Object::Object(const std::string& type, const std::string& pose, const std::string& id, 
      const QRectF& bbx)
    :	m_type(type), m_pose(pose), m_id(id), m_bbx(bbx)
  {
  }

  void Object::clear()
  {
    m_keypoints.clear();
  }

  void Object::add(const Keypoint& keypoint)
  {
    m_keypoints.push_back(keypoint);
  }

  void Object::move(const QRectF& bbx)
  {
    m_bbx = bbx;
  }

  void Object::scale(double factor)
  {
    // Scale the object bounding box
    const double bbx_w = factor * m_bbx.width();
    const double bbx_h = factor * m_bbx.height();
    const double top = factor * m_bbx.top();
    const double left = factor * m_bbx.left();

    m_bbx = QRectF(left, top, bbx_w, bbx_h);

    // Scale each keypoint
    for (std::vector<Keypoint>::iterator it = m_keypoints.begin(); it != m_keypoints.end(); ++ it)
    {
      it->m_point = QPointF(factor * it->m_point.x(), factor * it->m_point.y());
    }
  }

  void Object::translate(double dx, double dy)
  {
    m_bbx.translate(dx, dy);
    for (std::vector<Keypoint>::iterator it = m_keypoints.begin(); it != m_keypoints.end(); ++ it)
    {
      it->m_point.setX(it->m_point.x() + dx);
      it->m_point.setY(it->m_point.y() + dy);
    }
  }

  bool Object::find(const std::string& id, Keypoint& keypoint) const
  {
    for (std::vector<Keypoint>::const_iterator it = m_keypoints.begin(); it != m_keypoints.end(); ++ it)
      if (it->m_id == id)
      {
        keypoint = *it;
        return true;
      }
    return false;
  }

  bool Object::load(const std::string& filename, std::vector<Object>& objects)
  {
    objects.clear();

    // Open the file, test
    std::ifstream is(filename.c_str());
    if (is.is_open() == false) {
      return false;
    }

    // Parse the file
    static const int buff_size = 4096;
    char buff[buff_size];
    int n_objects = -1;
    while (is.getline(buff, buff_size))
    {
      std::string sbuff(buff);
      boost::trim(sbuff);
      std::vector<std::string> tokens;
      boost::split(tokens, sbuff, boost::is_any_of(" "));

      // Number of objects
      if (n_objects < 0) {
        if (tokens.size() != 1 ||
            (n_objects = boost::lexical_cast<int>(tokens[0].c_str())) < 0 || n_objects > 1000) {
          n_objects = -1;
          break;
        }
      }

      // Bounding box + keypoints [id x y]
      else {
        if (tokens.size() < 7) {
          n_objects = -1;
          break;
        }

        Object object(
            tokens[0], tokens[1], tokens[2],
            boost::lexical_cast<float>(tokens[3].c_str()),
            boost::lexical_cast<float>(tokens[4].c_str()),
            boost::lexical_cast<float>(tokens[5].c_str()),
            boost::lexical_cast<float>(tokens[6].c_str()));

        const int n2read = tokens.size() - 7;
        if (n2read % 3 != 0)
        {
          n_objects = -1;
          break;
        }

        const int n_keypoints = n2read / 3;
        for (int i = 0, ii = 7; i < n_keypoints; i ++, ii += 3)
        {
          object.add(Keypoint(  
                tokens[ii],
                boost::lexical_cast<float>(tokens[ii + 1]),
                boost::lexical_cast<float>(tokens[ii + 2])));
        }

        objects.push_back(object);
      }
    }

    // Check if the objects were read correctly
    if (n_objects < 0 || n_objects != (int)objects.size())
    {
      objects.clear();
      is.close();
      return false;
    }

    // OK
    is.close();
    return true;
  }

  bool Object::save(const std::string& filename, const std::vector<Object>& objects)
  {
    // Open the file
    std::ofstream os(filename.c_str());
    if (os.is_open() == false)
    {
      return false;
    }

    // Save the objects
    os << objects.size() << "\n";
    for (std::vector<Object>::const_iterator it = objects.begin(); it != objects.end(); ++ it)
    {
      const Object& object = *it;
      os << object.type() << " " << object.pose() << " " << object.id() << " "
        << object.bbx().left() << " " << object.bbx().top() << " " 
        << object.bbx().width() << " " << object.bbx().height() << " ";

      const std::vector<Keypoint>& keypoints = object.keypoints();
      for (std::vector<Keypoint>::const_iterator itf = keypoints.begin(); itf != keypoints.end(); ++ itf)
      {
        os << itf->m_id << " " << itf->m_point.x() << " " << itf->m_point.y() << " ";
      }

      os << "\n";
    }

    // OK
    os.close();
    return true;
  }

  bool Object::save(const std::string& filename) const
  {
    std::vector<Object> objects;
    objects.push_back(*this);
    return save(filename, objects);
  }

  double overlap(const QRectF& reg, const std::vector<Object>& objects, int* pwhich)
  {
    if (pwhich != 0)
      *pwhich = 0;

    double max_overlap = 0.0;		
    for (std::vector<Object>::const_iterator it = objects.begin(); it != objects.end(); ++ it)
    {
      const double o = overlap(reg, it->bbx());
      if (o > max_overlap)
      {
        max_overlap = o;
        if (pwhich != 0)
        {
          *pwhich = it - objects.begin();
        }
      }
    }
    return max_overlap;
  }

  std::vector<Object> filter_by_type(const std::vector<Object>& objects, const std::string& type)
  {
    std::vector<Object> result;
    for (std::vector<Object>::const_iterator it = objects.begin(); it != objects.end(); ++ it)
      if (it->type() == type)
      {
        result.push_back(*it);
      }
    return result;
  }

  std::vector<Object> filter_by_pose(const std::vector<Object>& objects, const std::string& pose)
  {
    std::vector<Object> result;
    for (std::vector<Object>::const_iterator it = objects.begin(); it != objects.end(); ++ it)
      if (it->pose() == pose)
      {
        result.push_back(*it);
      }
    return result;
  }

  std::vector<Object> filter_by_id(const std::vector<Object>& objects, const std::string& id)
  {
    std::vector<Object> result;
    for (std::vector<Object>::const_iterator it = objects.begin(); it != objects.end(); ++ it)
      if (it->id() == id)
      {
        result.push_back(*it);
      }
    return result;

  }

}}
