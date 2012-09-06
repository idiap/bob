/**
 * @file bob/ip/VLSIFT.h
 * @date Mon Dec 19 16:35:13 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file defines classes to compute SIFT features using VLFeat
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

#ifndef BOB_IP_VLSIFT_H
#define BOB_IP_VLSIFT_H

#include <blitz/array.h>
#include <stdint.h> // uint16_t declaration
#include <vector>
#include <boost/shared_ptr.hpp>
#include <vl/generic.h>
#include <vl/sift.h>

namespace bob {
  /**
    * \ingroup libip_api
    * @{
    *
    */
  namespace ip {

    /**
      * @brief This class allows the computation of SIFT features
      *   For more information, please refer to the following article:
      *     "Distinctive Image Features from Scale-Invariant Keypoints", 
      *     from D.G. Lowe,
      *     International Journal of Computer Vision, 60, 2, pp. 91-110, 2004
      */
    class VLSIFT
    {
      public:
        /**
          * @brief Constructor
          */
        VLSIFT(const size_t height, const size_t width, 
          const size_t n_intervals, const size_t n_octaves, 
          const int octave_min, const double peak_thres=0.03, 
          const double edge_thres=10., const double magnif=3.);

        /**
          * @brief Copy constructor
          */
        VLSIFT(const VLSIFT& other);

        /**
          * @brief Destructor
          */
        virtual ~VLSIFT();

        /**
          * @brief Assignment operator
          */
        VLSIFT& operator=(const VLSIFT& other);

        /**
          * @brief Equal to
          */
        bool operator==(const VLSIFT& b) const;
        /**
          * @brief Not equal to
          */
        bool operator!=(const VLSIFT& b) const; 

        /**
          * @brief Getters
          */
        size_t getHeight() const { return m_height; }
        size_t getWidth() const { return m_width; }
        size_t getNIntervals() const { return m_n_intervals; }
        size_t getNOctaves() const { return m_n_octaves; }
        int getOctaveMin() const { return m_octave_min; }
        double getPeakThres() const { return m_peak_thres; }
        double getEdgeThres() const { return m_edge_thres; }
        double getMagnif() const { return m_magnif; }
       
        /**
          * @brief Setters
          */
        void setHeight(const size_t height) 
        { m_height = height; cleanup(); allocateAndSet(); }
        void setWidth(const size_t width) 
        { m_width = width; cleanup(); allocateAndSet(); }
        void setNIntervals(const size_t n_intervals) 
        { m_n_intervals = n_intervals; cleanupFilter(); 
          allocateFilterAndSet(); }
        void setNOctaves(const size_t n_octaves) 
        { m_n_octaves = n_octaves; cleanupFilter(); allocateFilterAndSet(); }
        void setOctaveMin(const int octave_min) 
        { m_octave_min = octave_min; cleanupFilter(); allocateFilterAndSet(); }
        void setPeakThres(const double peak_thres) 
        { m_peak_thres = peak_thres; 
          vl_sift_set_peak_thresh(m_filt, m_peak_thres); }
        void setEdgeThres(const double edge_thres) 
        { m_edge_thres = edge_thres; 
          vl_sift_set_edge_thresh(m_filt, m_edge_thres); }
        void setMagnif(const double magnif) 
        { m_magnif = magnif; vl_sift_set_magnif(m_filt, m_magnif); }

        /**
          * @brief Extract SIFT features from a 2D blitz::Array, and save 
          *   the resulting features in the dst vector of 1D blitz::Arrays.
          */
        void operator()(const blitz::Array<uint8_t,2>& src, 
          std::vector<blitz::Array<double,1> >& dst);
        /**
          * @brief Extract SIFT features from a 2D blitz::Array, at the 
          *   keypoints specified by the 2D blitz::Array (Each row of length 3
          *   or 4 corresponds to a keypoint: y,x,sigma,[orientation]). The
          *   the resulting features are saved in the dst vector of 
          *   1D blitz::Arrays.
          */
        void operator()(const blitz::Array<uint8_t,2>& src, 
          const blitz::Array<double,2>& keypoints,
          std::vector<blitz::Array<double,1> >& dst);


      protected:
        /**
          * @brief Allocation methods
          */
        void allocateBuffers();
        void allocateFilter();
        void allocate();
        /**
          * @brief Resets the properties of the VLfeat filter object
          */
        void setFilterProperties(); 
        /**
          * @brief Reallocate and resets the properties of the VLfeat filter 
          * object
          */
        void allocateFilterAndSet();
        /**
          * @brief Reallocate and resets the properties of the VLfeat objects
          */
        void allocateAndSet();

        /**
          * @brief Deallocation methods
          */
        void cleanupBuffers();
        void cleanupFilter();
        void cleanup();

        /**
          * @brief Attributes
          */
        size_t m_height;
        size_t m_width;
        size_t m_n_intervals;
        size_t m_n_octaves;
        int m_octave_min; // might be negative
        double m_peak_thres;
        double m_edge_thres;
        double m_magnif;
        
        VlSiftFilt *m_filt;
        vl_uint8 *m_data;
        vl_sift_pix *m_fdata;
    };

  }
}

#endif /* BOB_IP_VLSIFT_H */
