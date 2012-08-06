/**
 * @file visioner/src/ipyramid.cc
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

#include "visioner/model/ipyramid.h"
#include "visioner/vision/image.h"
#include "visioner/vision/integral.h"

namespace bob { namespace visioner {

  // Compute the scalling factors for a given image size,
  //	model size and a difference in pixels <ds0> between detections at adjacent scales.
  static std::vector<double> scan_scales(	uint64_t model_rows, uint64_t model_cols,
      uint64_t image_rows, uint64_t image_cols,
      uint64_t ds0)
  {
    std::vector<double> scales;

    // First scale (original image)
    int prv_s = 0;
    if (image_rows >= model_rows && image_cols >= model_cols)
    {
      prv_s = 100;
      scales.push_back(1.0);
    }

    // Next scales (downscaled image)
    const double alpha = inverse(ds0) * model_cols;
    const int n_scales = inverse(ds0) *
      std::max(0, std::min((int)image_cols - (int)model_cols,
            (int)image_rows - (int)model_rows));

    for (int i = 0; i < n_scales; i ++)
    {
      const double scale = alpha * inverse(alpha + i);
      const int s = (int)(0.5 + 100.0 * scale);

      if (s == prv_s)
        continue;// Make sure the same scale is not added multiple times!

      prv_s = s;
      scales.push_back(scale);
    }

    return scales;
  }

  // Compute the displacement factors at the given <scale> scale
  //	given the desired <dx0/dy0> displacement at the original image size.
  static uint64_t scan_dx(uint64_t dx0, double scale)
  {
    return std::max((uint64_t)1, (uint64_t)(0.5 + scale * dx0));
  }
  static uint64_t scan_dy(uint64_t dy0, double scale)
  {
    return std::max((uint64_t)1, (uint64_t)(0.5 + scale * dy0));
  }

  // Update the internals of a scaled image
  static void update_ipscale(ipscale_t& ipscale, const param_t& param)
  {
    // Compute the scanning parameters
    ipscale.m_scan_dx = scan_dx(param.m_ds, ipscale.m_scale);
    ipscale.m_scan_dy = scan_dy(param.m_ds, ipscale.m_scale);
    ipscale.m_scan_min_x = param.min_col(ipscale.rows(), ipscale.cols());
    ipscale.m_scan_min_y = param.min_row(ipscale.rows(), ipscale.cols());
    ipscale.m_scan_max_x = param.max_col(ipscale.rows(), ipscale.cols());
    ipscale.m_scan_max_y = param.max_row(ipscale.rows(), ipscale.cols());
    ipscale.m_scan_w = param.m_cols;
    ipscale.m_scan_h = param.m_rows;
    ipscale.m_scan_o_w = (uint64_t)(0.5 + inverse(ipscale.m_scale) * param.m_cols);
    ipscale.m_scan_o_h = (uint64_t)(0.5 + inverse(ipscale.m_scale) * param.m_rows);
  }

  // Scale an image and its ground truth
  void ipscale_t::scale(double sfactor, ipscale_t& dst) const
  {
    dst.m_scale = range(sfactor, 0.0, 1.0);
    dst.m_inv_scale = inverse(dst.m_scale);		

    dst.m_objects = m_objects;
    for (std::vector<Object>::iterator it = dst.m_objects.begin(); it != dst.m_objects.end(); ++ it)
    {
      it->scale(dst.m_scale);
    }

    visioner::scale(m_image, dst.m_scale, dst.m_image);
  }

  // Constructor
  ipyramid_t::ipyramid_t(const param_t& param)
    :       Parametrizable(param)
  {                
  }

  // Reset to new parameters
  void ipyramid_t::reset(const param_t& param)
  {
    m_param = param;
  }

  // Loads scaled versions of an image and its ground truth
  bool ipyramid_t::load(const std::string& ifile, const std::string& gfile)
  {
    Matrix<uint8_t> tmp_image;
    if (visioner::load(ifile, tmp_image) == false)
    {
      return false;
    }

    // Compute the scalling factors
    const std::vector<double> scales = scan_scales(
        m_param.m_rows, m_param.m_cols, tmp_image.rows(), tmp_image.cols(), m_param.m_ds);
    if (scales.empty())
    {
      return false;
    }

    m_ipscales.resize(scales.size());

    // Load the ground truth and the image
    m_ipscales[0].m_scale = 1.0;
    m_ipscales[0].m_inv_scale = 1.0;
    if (	visioner::Object::load(gfile, m_ipscales[0].m_objects) == false)
      //visioner::load(ifile, m_ipscales[0].m_image) == false)
    {
      return false;
    }
    m_ipscales[0].m_image = tmp_image;
    update_ipscale(m_ipscales[0], m_param);

    // Build the scaled versions of the original image
    const ipscale_t& src = m_ipscales[0];
    for (uint64_t i = 1; i < scales.size(); i ++)
    {
      ipscale_t& dst = m_ipscales[i];
      src.scale(scales[i], dst);
      update_ipscale(dst, m_param);

      if (	dst.m_scan_min_x >= dst.m_scan_max_x ||
          dst.m_scan_min_y >= dst.m_scan_max_y)
      {
        m_ipscales.erase(m_ipscales.begin() + i, m_ipscales.end());
        break;
      }
    }

    // OK
    return true;
  }

  // Loads scaled versions of an image and its ground truth
  bool ipyramid_t::load(const ipscale_t& ipscale)
  {
    m_ipscales.clear();

    // Compute the scalling factors
    const std::vector<double> scales = scan_scales(
        m_param.m_rows, m_param.m_cols, ipscale.rows(), ipscale.cols(), m_param.m_ds);
    if (scales.empty())
    {
      return false;                        
    }

    m_ipscales.resize(scales.size());

    // Load the ground truth and the image
    m_ipscales[0] = ipscale;
    m_ipscales[0].m_scale = 1.0;
    m_ipscales[0].m_inv_scale = 1.0;
    update_ipscale(m_ipscales[0], m_param);

    // Build the scaled versions of the original image
    const ipscale_t& src = m_ipscales[0];
    for (uint64_t i = 1; i < scales.size(); i ++)
    {
      ipscale_t& dst = m_ipscales[i];
      src.scale(scales[i], dst);
      update_ipscale(dst, m_param);

      if (	dst.m_scan_min_x >= dst.m_scan_max_x ||
          dst.m_scan_min_y >= dst.m_scan_max_y)
      {
        m_ipscales.erase(m_ipscales.begin() + i, m_ipscales.end());
        break;
      }
    }

    // OK
    return true;
  }

  // Loads scaled versions of an image and its ground truth
  bool ipyramid_t::load(const uint8_t* image, uint64_t rows, uint64_t cols)
  {
    //AA: no simple way not to have a copy here...
    Matrix<uint8_t> tmp_image(rows, cols, image);

    // Compute the scalling factors
    const std::vector<double> scales = scan_scales(
        m_param.m_rows, m_param.m_cols, tmp_image.rows(), tmp_image.cols(), m_param.m_ds);
    if (scales.empty())
    {
      return false;
    }

    m_ipscales.resize(scales.size());

    // Load the ground truth and the image
    m_ipscales[0].m_scale = 1.0;
    m_ipscales[0].m_inv_scale = 1.0;
    m_ipscales[0].m_image = tmp_image;
    update_ipscale(m_ipscales[0], m_param);

    // Build the scaled versions of the original image
    const ipscale_t& src = m_ipscales[0];
    for (uint64_t i = 1; i < scales.size(); i ++)
    {
      ipscale_t& dst = m_ipscales[i];
      src.scale(scales[i], dst);
      update_ipscale(dst, m_param);

      if (	dst.m_scan_min_x >= dst.m_scan_max_x ||
          dst.m_scan_min_y >= dst.m_scan_max_y)
      {
        m_ipscales.erase(m_ipscales.begin() + i, m_ipscales.end());
        break;
      }
    }

    // OK
    return true;
  }

  // Map regions (at the original scale) to sub-windows
  subwindow_t ipyramid_t::map(const QRectF& reg, const param_t& param) const
  {
    subwindow_t sw;

    // Estimate the scale of the region
    const double scale = (double)param.m_cols * inverse(reg.width());
    double min_dist = std::numeric_limits<double>::max();
    sw.m_s = 0;
    for (uint64_t s = 0; s < size(); s ++)
    {
      const double dist = my_abs(scale - m_ipscales[s].m_scale);
      if (dist < min_dist)
      {
        min_dist = dist;
        sw.m_s = s;
      }
    }

    // ... and the coordinates for that scale                
    const ipscale_t& ip = m_ipscales[sw.m_s];
    sw.m_x = range((int)(0.5 + reg.center().x() * ip.m_scale - param.m_cols / 2), 
        0, (int)ip.cols() - (int)param.m_cols - 1);
    sw.m_y = range((int)(0.5 + reg.center().y() * ip.m_scale - param.m_rows / 2), 
        0, (int)ip.rows() - (int)param.m_rows - 1);

    return sw;
  }
  QRectF ipyramid_t::map(const subwindow_t& sw) const
  {
    const ipscale_t& ip = m_ipscales[sw.m_s];
    return QRectF(	ip.m_inv_scale * sw.m_x, 
        ip.m_inv_scale * sw.m_y,
        ip.m_scan_o_w, ip.m_scan_o_h);
  }

  // Return the neighbours (location + scale) of the given sub-window
  // NB: The sub-windows are returned for each scale independently
  std::vector<std::vector<subwindow_t> > ipyramid_t::neighbours(
      const subwindow_t& sw, int n_ds, 
      int n_dx, int dx, int n_dy, int dy, const param_t& param) const
  {
    std::vector<std::vector<subwindow_t> > sws;

    // Vary the scale
    for (int ds = -n_ds; ds <= n_ds; ds ++)
    {                
      const int s = sw.m_s + ds;
      if (s < 0 && s >= (int)size())
      {
        continue;
      }

      // Map the sub-window at this scale
      const subwindow_t seed = map(sw, s, param);
      if (check(seed, param) == false)
      {
        continue;
      }

      // Vary the location (at this scale)
      sws.push_back(neighbours(seed, n_dx, dx, n_dy, dy, param));
    }

    return sws;
  }

  // Check sub-windows
  bool ipyramid_t::check(const subwindow_t& sw) const
  {                
    return	check(sw, param());
  }
  bool ipyramid_t::check(const subwindow_t& sw, const param_t& param) const
  {
    return	(uint64_t)sw.m_s >= 0 && (uint64_t)sw.m_s < m_ipscales.size() &&
      sw.m_x >= 0 && sw.m_y >= 0 &&	
      (uint64_t)sw.m_x + param.m_cols < (uint64_t)m_ipscales[sw.m_s].cols() &&
      (uint64_t)sw.m_y + param.m_rows < (uint64_t)m_ipscales[sw.m_s].rows();
  }

  // Project a sub-window to another scale
  subwindow_t ipyramid_t::map(const subwindow_t& sw, int s, const param_t& param) const
  {
    const ipscale_t& ref_ip = m_ipscales[sw.m_s];
    const ipscale_t& ip = m_ipscales[s];

    // Project the center of the original detection to this scale
    const double cx = ip.m_scale / ref_ip.m_scale * (sw.m_x + 0.5 * param.m_cols);
    const double cy = ip.m_scale / ref_ip.m_scale * (sw.m_y + 0.5 * param.m_rows);

    return  subwindow_t(   (int)(0.5 + cx - 0.5 * param.m_cols),
        (int)(0.5 + cy - 0.5 * param.m_rows),
        s);
  }  

  // Return the neighbours (location only) of the given sub-window
  std::vector<subwindow_t> ipyramid_t::neighbours(
      const subwindow_t& sw, int n_dx, int dx, int n_dy, int dy, const param_t& param) const
  {
    std::vector<subwindow_t> sws;		

    for (int idx = -(int)n_dx; idx <= (int)n_dx; idx ++)
    {
      for (int idy = -(int)n_dy; idy <= (int)n_dy; idy ++)
      {
        const int x = sw.m_x + idx * dx, y = sw.m_y + idy * dy;
        if (check(subwindow_t(x, y, sw.m_s), param) == true)
        {
          sws.push_back(subwindow_t(x, y, sw.m_s));
        }
      }
    }

    return sws;
  }

}}
