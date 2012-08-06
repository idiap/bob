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

  /**
   * Pyramid of scaled images.
   */
  struct ipscale_t {

    public: //api

      // Constructor
      ipscale_t() :	m_scale(1.0), m_inv_scale(1.0),
        m_scan_dx(1), m_scan_dy(1), m_scan_min_x(0), m_scan_max_x(0),
        m_scan_min_y(0), m_scan_max_y(0), m_scan_w(0), m_scan_h(0),
        m_scan_o_w(0), m_scan_o_h(0) {			}

      // Access functions
      uint64_t rows() const { return m_image.rows(); }
      uint64_t cols() const { return m_image.cols(); }

      // Scale an image and its ground truth
      void scale(double sfactor, ipscale_t& dst) const;

    public: //attributes

      Matrix<uint8_t>	m_image;	// Grayscale image
      std::vector<Object>	m_objects;	// Ground truth data

      double	m_scale;	// Scale factor relative to the original image size		
      double	m_inv_scale;

      // Scanning:
      int		m_scan_dx, m_scan_dy;		// Ox/Oy displacement at this scale
      int		m_scan_min_x, m_scan_max_x;	// Ox extremes
      int		m_scan_min_y, m_scan_max_y;	// Oy extremes
      uint64_t		m_scan_w, m_scan_h;		// Scanning window at this scale
      uint64_t		m_scan_o_w, m_scan_o_h;		//	and at the original scale
  };

  struct ipyramid_t : public Parametrizable {

    public:

      // Constructor
      ipyramid_t(const param_t& param = param_t());

      // Destructor
      virtual ~ipyramid_t() {}

      // Reset to new parameters
      virtual void reset(const param_t& param);

      // Load scaled versions of an image and its ground truth	
      bool load(const std::string& ifile, const std::string& gfile);		
      bool load(const ipscale_t& ipscale);
      bool load(const uint8_t* image, uint64_t rows, uint64_t cols);

      // Map regions (at the original scale) to sub-windows
      subwindow_t map(const QRectF& reg, const param_t& param) const;
      QRectF map(const subwindow_t& sw) const;

      // Return the neighbours (location + scale) of the given sub-window
      // NB: The sub-windows are returned for each scale independently
      std::vector<std::vector<subwindow_t> > neighbours(
          const subwindow_t& sw, int n_ds, 
          int n_dx, int dx, int n_dy, int dy, const param_t& param) const;

      // Check sub-windows
      bool check(const subwindow_t& sw) const;
      bool check(const subwindow_t& sw, const param_t& param) const;

      // Access functions
      bool empty() const { return m_ipscales.empty(); }
      uint64_t size() const { return m_ipscales.size(); }
      const ipscale_t& operator[](uint64_t i) const { return m_ipscales[i]; }

    private:

      // Project a sub-window to another scale
      subwindow_t map(const subwindow_t& sw, int s, const param_t& param) const;

      // Return the neighbours (location only) of the given sub-window
      std::vector<subwindow_t> neighbours(const subwindow_t& sw, int n_dx, int dx, int n_dy, int dy, const param_t& param) const;

    private: // representation

      std::vector<ipscale_t>  m_ipscales; // Images at different scales        
  };

}}

#endif // BOB_VISIONER_IPYRAMID_H
