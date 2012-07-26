#include "visioner/model/ipyramid.h"
#include "visioner/vision/image.h"
#include "visioner/vision/integral.h"

namespace bob { namespace visioner {

  // Compute the scalling factors for a given image size,
  //	model size and a difference in pixels <ds0> between detections at adjacent scales.
  static scalars_t scan_scales(	index_t model_rows, index_t model_cols,
      index_t image_rows, index_t image_cols,
      index_t ds0)
  {
    scalars_t scales;

    // First scale (original image)
    int prv_s = 0;
    if (image_rows >= model_rows && image_cols >= model_cols)
    {
      prv_s = 100;
      scales.push_back(1.0);
    }

    // Next scales (downscaled image)
    const scalar_t alpha = inverse(ds0) * model_cols;
    const int n_scales = inverse(ds0) *
      std::max(0, std::min((int)image_cols - (int)model_cols,
            (int)image_rows - (int)model_rows));

    for (int i = 0; i < n_scales; i ++)
    {
      const scalar_t scale = alpha * inverse(alpha + i);
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
  static index_t scan_dx(index_t dx0, scalar_t scale)
  {
    return std::max((index_t)1, (index_t)(0.5 + scale * dx0));
  }
  static index_t scan_dy(index_t dy0, scalar_t scale)
  {
    return std::max((index_t)1, (index_t)(0.5 + scale * dy0));
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
    ipscale.m_scan_o_w = (index_t)(0.5 + inverse(ipscale.m_scale) * param.m_cols);
    ipscale.m_scan_o_h = (index_t)(0.5 + inverse(ipscale.m_scale) * param.m_rows);
  }

  // Scale an image and its ground truth
  void ipscale_t::scale(scalar_t sfactor, ipscale_t& dst) const
  {
    dst.m_scale = range(sfactor, 0.0, 1.0);
    dst.m_inv_scale = inverse(dst.m_scale);		

    dst.m_objects = m_objects;
    for (objects_t::iterator it = dst.m_objects.begin(); it != dst.m_objects.end(); ++ it)
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
  bool ipyramid_t::load(const string_t& ifile, const string_t& gfile)
  {
    greyimage_t tmp_image;
    if (visioner::load(ifile, tmp_image) == false)
    {
      return false;
    }

    // Compute the scalling factors
    const scalars_t scales = scan_scales(
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
    for (index_t i = 1; i < scales.size(); i ++)
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
    const scalars_t scales = scan_scales(
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
    for (index_t i = 1; i < scales.size(); i ++)
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
  bool ipyramid_t::load(const grey_t* image, index_t rows, index_t cols)
  {
    //AA: no simple way not to have a copy here...
    greyimage_t tmp_image(rows, cols, image);

    // Compute the scalling factors
    const scalars_t scales = scan_scales(
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
    for (index_t i = 1; i < scales.size(); i ++)
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
  sw_t ipyramid_t::map(const rect_t& reg, const param_t& param) const
  {
    sw_t sw;

    // Estimate the scale of the region
    const scalar_t scale = (scalar_t)param.m_cols * inverse(reg.width());
    scalar_t min_dist = std::numeric_limits<scalar_t>::max();
    sw.m_s = 0;
    for (index_t s = 0; s < size(); s ++)
    {
      const scalar_t dist = my_abs(scale - m_ipscales[s].m_scale);
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
  rect_t ipyramid_t::map(const sw_t& sw) const
  {
    const ipscale_t& ip = m_ipscales[sw.m_s];
    return rect_t(	ip.m_inv_scale * sw.m_x, 
        ip.m_inv_scale * sw.m_y,
        ip.m_scan_o_w, ip.m_scan_o_h);
  }

  // Return the neighbours (location + scale) of the given sub-window
  // NB: The sub-windows are returned for each scale independently
  std::vector<sws_t> ipyramid_t::neighbours(
      const sw_t& sw, int n_ds, 
      int n_dx, int dx, int n_dy, int dy, const param_t& param) const
  {
    std::vector<sws_t> sws;

    // Vary the scale
    for (int ds = -n_ds; ds <= n_ds; ds ++)
    {                
      const int s = sw.m_s + ds;
      if (s < 0 && s >= (int)size())
      {
        continue;
      }

      // Map the sub-window at this scale
      const sw_t seed = map(sw, s, param);
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
  bool ipyramid_t::check(const sw_t& sw) const
  {                
    return	check(sw, param());
  }
  bool ipyramid_t::check(const sw_t& sw, const param_t& param) const
  {
    return	(index_t)sw.m_s >= 0 && (index_t)sw.m_s < m_ipscales.size() &&
      sw.m_x >= 0 && sw.m_y >= 0 &&	
      (index_t)sw.m_x + param.m_cols < (index_t)m_ipscales[sw.m_s].cols() &&
      (index_t)sw.m_y + param.m_rows < (index_t)m_ipscales[sw.m_s].rows();
  }

  // Project a sub-window to another scale
  sw_t ipyramid_t::map(const sw_t& sw, int s, const param_t& param) const
  {
    const ipscale_t& ref_ip = m_ipscales[sw.m_s];
    const ipscale_t& ip = m_ipscales[s];

    // Project the center of the original detection to this scale
    const scalar_t cx = ip.m_scale / ref_ip.m_scale * (sw.m_x + 0.5 * param.m_cols);
    const scalar_t cy = ip.m_scale / ref_ip.m_scale * (sw.m_y + 0.5 * param.m_rows);

    return  sw_t(   (int)(0.5 + cx - 0.5 * param.m_cols),
        (int)(0.5 + cy - 0.5 * param.m_rows),
        s);
  }  

  // Return the neighbours (location only) of the given sub-window
  sws_t ipyramid_t::neighbours(
      const sw_t& sw, int n_dx, int dx, int n_dy, int dy, const param_t& param) const
  {
    sws_t sws;		

    for (int idx = -(int)n_dx; idx <= (int)n_dx; idx ++)
    {
      for (int idy = -(int)n_dy; idy <= (int)n_dy; idy ++)
      {
        const int x = sw.m_x + idx * dx, y = sw.m_y + idy * dy;
        if (check(sw_t(x, y, sw.m_s), param) == true)
        {
          sws.push_back(sw_t(x, y, sw.m_s));
        }
      }
    }

    return sws;
  }

}}
