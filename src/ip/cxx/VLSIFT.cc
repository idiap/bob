/**
 * @file ip/cxx/VLSIFT.cc
 * @date Mon Dec 19 16:35:13 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief SIFT implementation
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

#include "bob/ip/VLSIFT.h"

#include <vl/pgm.h>
#include "bob/core/array_assert.h"
#include "bob/ip/Exception.h"

bob::ip::VLSIFT::VLSIFT(const size_t height, const size_t width, 
    const size_t n_intervals, const size_t n_octaves, const int octave_min,
    const double peak_thres, const double edge_thres, const double magnif):
  m_height(height), m_width(width), m_n_intervals(n_intervals), 
  m_n_octaves(n_octaves), m_octave_min(octave_min),
  m_peak_thres(peak_thres), m_edge_thres(edge_thres), m_magnif(magnif)
{
  // Allocates buffers and filter, and set filter properties
  allocateAndSet();
}

bob::ip::VLSIFT::VLSIFT(const VLSIFT& other):
  m_height(other.m_height), m_width(other.m_width), 
  m_n_intervals(other.m_n_intervals), m_n_octaves(other.m_n_octaves), 
  m_octave_min(other.m_octave_min), m_peak_thres(other.m_peak_thres), 
  m_edge_thres(other.m_edge_thres), m_magnif(other.m_magnif)
{
  // Allocates buffers and filter, and set filter properties
  allocateAndSet();
}

bob::ip::VLSIFT& bob::ip::VLSIFT::operator=(const bob::ip::VLSIFT& other)
{
  if (this != &other)
  {
    m_height = other.m_height;
    m_width = other.m_width;
    m_n_intervals = other.m_n_intervals;
    m_n_octaves = other.m_n_octaves; 
    m_octave_min = other.m_octave_min;
    m_peak_thres = other.m_peak_thres;
    m_edge_thres = other.m_edge_thres;
    m_magnif = other.m_magnif;
  
    // Allocates buffers and filter, and set filter properties
    allocateAndSet();
  }
  return *this;
}

bool bob::ip::VLSIFT::operator==(const bob::ip::VLSIFT& b) const
{
  return (this->m_height == b.m_height && this->m_width == b.m_width && 
          this->m_n_intervals == b.m_n_intervals && 
          this->m_n_octaves == b.m_n_octaves && 
          this->m_octave_min == b.m_octave_min && 
          this->m_peak_thres == b.m_peak_thres && 
          this->m_edge_thres == b.m_edge_thres &&
          this->m_magnif == b.m_magnif);
}

bool bob::ip::VLSIFT::operator!=(const bob::ip::VLSIFT& b) const
{
  return !(this->operator==(b));
}

void bob::ip::VLSIFT::operator()(const blitz::Array<uint8_t,2>& src, 
  std::vector<blitz::Array<double,1> >& dst)
{
  // Clears the vector
  dst.clear();
  vl_bool err=VL_ERR_OK;

  // Copies data
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q) 
    m_data[q] = src((int)(q/m_width), (int)(q%m_width));
  // Converts data type
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q) 
    m_fdata[q] = m_data[q];

  // Processes each octave
  int i=0;
  bool first=true;
  while(1) 
  {
    VlSiftKeypoint const *keys = 0;
    int nkeys;

    // Calculates the GSS for the next octave
    if(first) 
    {
      first = false;
      err = vl_sift_process_first_octave(m_filt, m_fdata);
    } 
    else 
      err = vl_sift_process_next_octave(m_filt);

    if(err)
    {
      err = VL_ERR_OK;
      break;
    }

    // Runs the detector
    vl_sift_detect(m_filt);
    keys = vl_sift_get_keypoints(m_filt);
    nkeys = vl_sift_get_nkeypoints(m_filt);
    i = 0;
    
    // Loops over the keypoint
    for(; i < nkeys ; ++i) {
      double angles[4];
      int nangles;
      VlSiftKeypoint const *k;

      // Obtains keypoint orientations
      k = keys + i;
      nangles = vl_sift_calc_keypoint_orientations(m_filt, angles, k);

      // For each orientation
      for(unsigned int q=0; q<(unsigned)nangles; ++q) {
        blitz::Array<double,1> res(128+4);
        vl_sift_pix descr[128];

        // Computes the descriptor
        vl_sift_calc_keypoint_descriptor(m_filt, descr, k, angles[q]);

        int l;
        res(0) = k->x;
        res(1) = k->y;
        res(2) = k->sigma;
        res(3) = angles[q];
        for(l=0; l<128; ++l)
          res(4+l) = 512. * descr[l];

        // Adds it to the vector
        dst.push_back(res);
      }
    }
  }

}

void bob::ip::VLSIFT::operator()(const blitz::Array<uint8_t,2>& src, 
  const blitz::Array<double,2>& keypoints,
  std::vector<blitz::Array<double,1> >& dst)
{
  if(keypoints.extent(1) != 3 && keypoints.extent(1) != 4)
    throw bob::ip::Exception();
 
  // Clears the vector
  dst.clear();
  vl_bool err=VL_ERR_OK;

  // Copies data
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q) 
    m_data[q] = src((int)(q/m_width), (int)(q%m_width));
  // Converts data type
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q) 
    m_fdata[q] = m_data[q];

  // Processes each octave
  bool first=true;
  while(1) 
  {
    // Calculates the GSS for the next octave
    if(first) 
    {
      first = false;
      err = vl_sift_process_first_octave(m_filt, m_fdata);
    } 
    else 
      err = vl_sift_process_next_octave(m_filt);

    if(err)
    {
      err = VL_ERR_OK;
      break;
    }

    // Loops over the keypoint
    for(int i=0; i<keypoints.extent(0); ++i) {
      double angles[4];
      int nangles;
      VlSiftKeypoint ik;
      VlSiftKeypoint const *k;

      // Obtain keypoint orientations 
      vl_sift_keypoint_init(m_filt, &ik,
        keypoints(i,1), keypoints(i,0), keypoints(i,2)); // x, y, sigma

      if(ik.o != vl_sift_get_octave_index(m_filt))
        continue; // Not current scale/octave

      k = &ik ;

      // Compute orientations if required
      if(keypoints.extent(1) == 4)
      {
        angles[0] = keypoints(i,3);
        nangles = 1;
      }
      else
        // TODO: No way to know if several keypoints are generated from one location
        nangles = vl_sift_calc_keypoint_orientations(m_filt, angles, k);

      // For each orientation
      for(unsigned int q=0; q<(unsigned)nangles; ++q) {
        blitz::Array<double,1> res(128+4);
        vl_sift_pix descr[128];

        // Computes the descriptor
        vl_sift_calc_keypoint_descriptor(m_filt, descr, k, angles[q]);

        int l;
        res(0) = k->x;
        res(1) = k->y;
        res(2) = k->sigma;
        res(3) = angles[q];
        for(l=0; l<128; ++l)
          res(4+l) = 512. * descr[l];

        // Adds it to the vector
        dst.push_back(res);
      }
    }
  }
}


void bob::ip::VLSIFT::allocateBuffers()
{
  const size_t npixels = m_height * m_width;
  // Allocates buffers
  m_data  = (vl_uint8*)malloc(npixels * sizeof(vl_uint8)); 
  m_fdata = (vl_sift_pix*)malloc(npixels * sizeof(vl_sift_pix));
  // TODO: deals with allocation error?
}

void bob::ip::VLSIFT::allocateFilter()
{
  // Generates the filter
  m_filt = vl_sift_new(m_width, m_height, m_n_octaves, m_n_intervals, m_octave_min);
  // TODO: deals with allocation error?
}

void bob::ip::VLSIFT::allocate()
{
  allocateBuffers();
  allocateFilter();
}

void bob::ip::VLSIFT::setFilterProperties()
{
  // Set filter properties
  vl_sift_set_edge_thresh(m_filt, m_edge_thres);
  vl_sift_set_peak_thresh(m_filt, m_peak_thres);
  vl_sift_set_magnif(m_filt, m_magnif);
}

void bob::ip::VLSIFT::allocateFilterAndSet()
{
  allocateFilter();
  setFilterProperties();
}

void bob::ip::VLSIFT::allocateAndSet()
{
  allocateBuffers();
  allocateFilterAndSet();
}

void bob::ip::VLSIFT::cleanupBuffers()
{
  // Releases image data
  free(m_fdata);
  m_fdata = 0;
  free(m_data);
  m_data = 0;
}

void bob::ip::VLSIFT::cleanupFilter()
{
  // Releases filter
  vl_sift_delete(m_filt);
  m_filt = 0;
}

void bob::ip::VLSIFT::cleanup()
{
  cleanupBuffers();
  cleanupFilter();
}

bob::ip::VLSIFT::~VLSIFT()
{
  cleanup();
}
