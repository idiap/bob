/**
 * @file visioner/visioner/model/ipyramid.h
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

#ifndef BOB_VISIONER_IPYRAMID_H
#define BOB_VISIONER_IPYRAMID_H

#include "visioner/vision/object.h"
#include "visioner/vision/image.h"
#include "visioner/model/param.h"

namespace bob { namespace visioner {

  struct ipscale_t;
  typedef std::vector<ipscale_t>  ipscales_t;

  /////////////////////////////////////////////////////////////////////////////////////////
  // Pyramid of scaled images.
  /////////////////////////////////////////////////////////////////////////////////////////

  struct ipscale_t
  {
    public:

      // Constructor
      ipscale_t()
        :	m_scale(1.0), m_inv_scale(1.0),
        m_scan_dx(1), m_scan_dy(1),
        m_scan_min_x(0), m_scan_max_x(0),
        m_scan_min_y(0), m_scan_max_y(0),
        m_scan_w(0), m_scan_h(0),
        m_scan_o_w(0), m_scan_o_h(0)
    {			
    }

      // Access functions
      index_t rows() const { return m_image.rows(); }
      index_t cols() const { return m_image.cols(); }

      // Scale an image and its ground truth
      void scale(scalar_t sfactor, ipscale_t& dst) const;

    public:

      // Attributes
      greyimage_t	m_image;	// Grayscale image
      objects_t	m_objects;	// Ground truth data

      scalar_t	m_scale;	// Scale factor relative to the original image size		
      scalar_t	m_inv_scale;
      // Scanning:
      int		m_scan_dx, m_scan_dy;		// Ox/Oy displacement at this scale
      int		m_scan_min_x, m_scan_max_x;	// Ox extremes
      int		m_scan_min_y, m_scan_max_y;	// Oy extremes
      index_t		m_scan_w, m_scan_h;		// Scanning window at this scale
      index_t		m_scan_o_w, m_scan_o_h;		//	and at the original scale
  };

  struct ipyramid_t : public Parametrizable
  {
    public:

      // Constructor
      ipyramid_t(const param_t& param = param_t());

      // Destructor
      virtual ~ipyramid_t() {}

      // Reset to new parameters
      virtual void reset(const param_t& param);

      // Load scaled versions of an image and its ground truth	
      bool load(const string_t& ifile, const string_t& gfile);		
      bool load(const ipscale_t& ipscale);
      bool load(const grey_t* image, index_t rows, index_t cols);

      // Map regions (at the original scale) to sub-windows
      sw_t map(const rect_t& reg, const param_t& param) const;
      rect_t map(const sw_t& sw) const;

      // Return the neighbours (location + scale) of the given sub-window
      // NB: The sub-windows are returned for each scale independently
      std::vector<sws_t> neighbours(
          const sw_t& sw, int n_ds, 
          int n_dx, int dx, int n_dy, int dy, const param_t& param) const;

      // Check sub-windows
      bool check(const sw_t& sw) const;
      bool check(const sw_t& sw, const param_t& param) const;

      // Access functions
      bool empty() const { return m_ipscales.empty(); }
      index_t size() const { return m_ipscales.size(); }
      const ipscale_t& operator[](index_t i) const { return m_ipscales[i]; }

    private:

      // Project a sub-window to another scale
      sw_t map(const sw_t& sw, int s, const param_t& param) const;

      // Return the neighbours (location only) of the given sub-window
      sws_t neighbours(
          const sw_t& sw, int n_dx, int dx, int n_dy, int dy, const param_t& param) const;

    private:

      // Attributes
      std::vector<ipscale_t>  m_ipscales;             // Images at different scales        
  };

}}

#endif // BOB_VISIONER_IPYRAMID_H
