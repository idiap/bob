/**
 * @file cxx/ip/src/VLSIFT.cc
 * @date Mon Dec 19 16:35:13 2011 +0100
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief SIFT implementation
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

#include "ip/VLSIFT.h"

#include <vl/pgm.h>
#include "core/array_assert.h"
#include "core/logging.h"
#include "core/convert.h"
#include "io/Array.h"

namespace ip = bob::ip;
namespace tca = bob::core::array;

ip::VLSIFT::VLSIFT(const int height, const int width, const int n_intervals,
    const int n_octaves, const int octave_min,
    const double peak_thres, const double edge_thres, const double magnif): 
  m_height(height), m_width(width), m_n_intervals(n_intervals), 
  m_n_octaves(n_octaves), m_octave_min(octave_min),
  m_peak_thres(peak_thres), m_edge_thres(edge_thres), m_magnif(magnif)
{
  const int npixels = height * width;

  // Allocates buffers
  m_data  = (vl_uint8*)malloc(npixels * sizeof(vl_uint8)); 
  m_fdata = (vl_sift_pix*)malloc(npixels * sizeof(vl_sift_pix));

  // Generates the filters
  m_filt = vl_sift_new(m_width, m_height, m_n_octaves, m_n_intervals, m_octave_min);
  vl_sift_set_edge_thresh(m_filt, m_edge_thres);
  vl_sift_set_peak_thresh(m_filt, m_peak_thres);
  vl_sift_set_magnif(m_filt, m_magnif);

  // TODO: deals with allocation error?
}

void ip::VLSIFT::operator()(const blitz::Array<uint8_t,2>& src, 
 std::vector<blitz::Array<double,1> >& dst)
{
  // Clears the vector
  dst.clear();
  vl_bool err=VL_ERR_OK;

  // Copies data
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q) {
    m_data[q] = src((int)(q/m_width), (int)(q%m_width));
  }
  // Converts data type
  for(unsigned int q=0; q<(unsigned)(m_width * m_height); ++q) {
    m_fdata[q] = m_data[q];
  }

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

ip::VLSIFT::~VLSIFT()
{
  // Releases filter
  if(m_filt) {
    vl_sift_delete(m_filt);
    m_filt = 0;
  }

  // Releases image data
  if(m_fdata) {
    free(m_fdata);
    m_fdata = 0;
  }

  // Releases image data
  if(m_data) {
    free(m_data);
    m_data = 0;
  }
}
