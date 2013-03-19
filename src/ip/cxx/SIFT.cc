/**
 * @file src/ip/cxx/SIFT.cc
 * @date Sun Sep 9 19:22:00 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#include <bob/ip/SIFT.h>
#include <bob/core/assert.h>
#include <algorithm>

bob::ip::SIFT::SIFT(const size_t height, const size_t width, 
    const size_t n_octaves, const size_t n_intervals, const int octave_min,
    const double sigma_n, const double sigma0, 
    const double contrast_thres, const double edge_thres,
    const double norm_thres, const double kernel_radius_factor,
    const bob::sp::Extrapolation::BorderType border_type):
  m_gss(new bob::ip::GaussianScaleSpace(height, width, n_octaves, 
    n_intervals, octave_min, sigma_n, sigma0, kernel_radius_factor, 
    border_type)),
  m_contrast_thres(contrast_thres), 
  m_edge_thres(edge_thres),
  m_norm_thres(norm_thres),
  m_descr_n_blocks(4),
  m_descr_n_bins(8),
  m_descr_gaussian_window_size(m_descr_n_blocks/2.),
  m_descr_magnif(3.),
  m_norm_eps(1e-10)
{   
  updateEdgeEffThreshold();
  resetCache();
}   

bob::ip::SIFT::SIFT(const SIFT& other):
  m_gss(new bob::ip::GaussianScaleSpace(*(other.m_gss))),
  m_contrast_thres(other.m_contrast_thres),
  m_edge_thres(other.m_edge_thres), m_norm_thres(other.m_norm_thres),
  m_descr_n_blocks(other.m_descr_n_blocks), 
  m_descr_n_bins(other.m_descr_n_bins),
  m_descr_gaussian_window_size(other.m_descr_gaussian_window_size),
  m_descr_magnif(other.m_descr_magnif), m_norm_eps(other.m_norm_eps)
{
  updateEdgeEffThreshold();
  resetCache();
}

bob::ip::SIFT::~SIFT()
{
}

bob::ip::SIFT&
bob::ip::SIFT::operator=(const bob::ip::SIFT& other)
{
  if (this != &other)
  {
    m_gss.reset(new bob::ip::GaussianScaleSpace(*(other.m_gss)));
    m_contrast_thres = other.m_contrast_thres;
    m_edge_thres = other.m_edge_thres;
    m_descr_n_blocks = other.m_descr_n_blocks;
    m_descr_n_bins = other.m_descr_n_bins;
    m_descr_gaussian_window_size = other.m_descr_gaussian_window_size;
    m_descr_magnif = other.m_descr_magnif;
    m_norm_eps = other.m_norm_eps;
    updateEdgeEffThreshold();
    m_norm_thres = other.m_norm_thres;
    resetCache();
  }
  return *this;
}

bool 
bob::ip::SIFT::operator==(const bob::ip::SIFT& b) const
{
  if (*(this->m_gss) != *(b.m_gss) ||
        this->m_contrast_thres != b.m_contrast_thres ||
        this->m_edge_thres != b.m_edge_thres ||
        this->m_edge_eff_thres != b.m_edge_eff_thres ||
        this->m_norm_thres != b.m_norm_thres ||
        this->m_descr_n_blocks != b.m_descr_n_blocks ||
        this->m_descr_n_bins != b.m_descr_n_bins ||
        this->m_descr_gaussian_window_size != b.m_descr_gaussian_window_size ||
        this->m_descr_magnif != b.m_descr_magnif ||
        this->m_norm_thres != b.m_norm_thres)
    return false;

 if (this->m_gss_pyr.size() != b.m_gss_pyr.size() ||
     this->m_dog_pyr.size() != b.m_dog_pyr.size() ||
     this->m_gss_pyr_grad_mag.size() != b.m_gss_pyr_grad_mag.size() ||
     this->m_gss_pyr_grad_or.size() != b.m_gss_pyr_grad_or.size() ||
     this->m_gradient_maps.size() != b.m_gradient_maps.size())
    return false;

  for (size_t i=0; i<m_gss_pyr.size(); ++i)
    if (!bob::core::array::isEqual(this->m_gss_pyr[i], b.m_gss_pyr[i]))
      return false;

  for (size_t i=0; i<m_dog_pyr.size(); ++i)
    if (!bob::core::array::isEqual(this->m_dog_pyr[i], b.m_dog_pyr[i]))
      return false;

  for (size_t i=0; i<m_gss_pyr_grad_mag.size(); ++i)
    if (!bob::core::array::isEqual(this->m_gss_pyr_grad_mag[i], b.m_gss_pyr_grad_mag[i]))
      return false;

  for (size_t i=0; i<m_gss_pyr_grad_or.size(); ++i)
    if (!bob::core::array::isEqual(this->m_gss_pyr_grad_or[i], b.m_gss_pyr_grad_or[i]))
      return false;

  for (size_t i=0; i<m_gradient_maps.size(); ++i)
    if (*(this->m_gradient_maps[i]) != *(b.m_gradient_maps[i]))
      return false;

  return true;
}

bool 
bob::ip::SIFT::operator!=(const bob::ip::SIFT& b) const
{
  return !(this->operator==(b));
}

const blitz::TinyVector<int,3> 
bob::ip::SIFT::getDescriptorShape() const
{
  return blitz::TinyVector<int,3>(m_descr_n_blocks, m_descr_n_blocks, m_descr_n_bins);
}


void bob::ip::SIFT::resetCache()
{
  m_gss->allocateOutputPyramid(m_gss_pyr);
  m_dog_pyr.clear();
  m_gss_pyr_grad_mag.clear();
  m_gss_pyr_grad_or.clear();
  for (size_t i=0; i<m_gss_pyr.size(); ++i)
  {
    m_dog_pyr.push_back(blitz::Array<double,3>(m_gss_pyr[i].extent(0)-1,
      m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2)));
    m_gss_pyr_grad_mag.push_back(blitz::Array<double,3>(m_gss_pyr[i].extent(0)-3,
      m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2)));
    m_gss_pyr_grad_or.push_back(blitz::Array<double,3>(m_gss_pyr[i].extent(0)-3,
      m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2)));
    m_gradient_maps.push_back(boost::shared_ptr<bob::ip::GradientMaps>(new 
      bob::ip::GradientMaps(m_gss_pyr[i].extent(1), m_gss_pyr[i].extent(2))));
  }
}

const blitz::TinyVector<int,3> 
bob::ip::SIFT::getGaussianOutputShape(const int octave) const
{
  return m_gss->getOutputShape(octave);
}

void bob::ip::SIFT::computeDog()
{
  // Computes the Difference of Gaussians pyramid
  blitz::Range rall = blitz::Range::all();
  for (size_t o=0; o<m_gss_pyr.size(); ++o)
    for (size_t s=0; s<(size_t)(m_gss_pyr[o].extent(0)-1); ++s)
    {
      blitz::Array<double,2> dst_os = m_dog_pyr[o](s, rall, rall);
      blitz::Array<double,2> src1 = m_gss_pyr[o](s, rall, rall);
      blitz::Array<double,2> src2 = m_gss_pyr[o](s+1, rall, rall);
      dst_os = src2 - src1;
    }
}

void bob::ip::SIFT::computeGradient()
{
  blitz::Range rall = blitz::Range::all();
  for (size_t i=0; i<m_gss_pyr.size(); ++i)
  {
    blitz::Array<double,3>& gss = m_gss_pyr[i];
    blitz::Array<double,3>& gmag = m_gss_pyr_grad_mag[i];
    blitz::Array<double,3>& gor = m_gss_pyr_grad_or[i];
    boost::shared_ptr<bob::ip::GradientMaps> gmap = m_gradient_maps[i];
    for (int s=0; s<gmag.extent(0); ++s)
    {
      blitz::Array<double,2> gss_s = gss(s+1, rall, rall);
      blitz::Array<double,2> gmag_s = gmag(s, rall, rall);
      blitz::Array<double,2> gor_s = gor(s, rall, rall);
      gmap->forward(gss_s, gmag_s, gor_s);
    }
  }
}

void bob::ip::SIFT::computeDescriptor(const std::vector<boost::shared_ptr<bob::ip::GSSKeypoint> >& keypoints,
  blitz::Array<double,4>& dst) const
{
  blitz::Range rall = blitz::Range::all();
  for (size_t k=0; k<keypoints.size(); ++k)
  {
    blitz::Array<double,3> dst_k = dst(k, rall, rall, rall);
    computeDescriptor(*(keypoints[k]), dst_k);
  }
}

void bob::ip::SIFT::computeDescriptor(const bob::ip::GSSKeypoint& keypoint,
  blitz::Array<double,3>& dst) const
{
  // Extracts more detailed information about the keypoint (octave and scale)
  bob::ip::GSSKeypointInfo keypoint_info;
  computeKeypointInfo(keypoint, keypoint_info);
  computeDescriptor(keypoint, keypoint_info, dst);
}

void bob::ip::SIFT::computeDescriptor(const bob::ip::GSSKeypoint& keypoint, 
  const bob::ip::GSSKeypointInfo& keypoint_info, blitz::Array<double,3>& dst) const
{
  // Check output dimensionality
  const blitz::TinyVector<int,3> shape = bob::ip::SIFT::getDescriptorShape();
  bob::core::array::assertSameShape(dst, shape);

  // Get gradient
  blitz::Range rall = blitz::Range::all();
  // Index scale has a -1, as the gradients are not computed for scale -1, Ns and Ns+1
  // but the provided index is the one, for which scale -1 corresponds to keypoint_info.s=0.
  blitz::Array<double,2> gmag = m_gss_pyr_grad_mag[keypoint_info.o](keypoint_info.s-1,rall,rall);
  blitz::Array<double,2> gor = m_gss_pyr_grad_or[keypoint_info.o](keypoint_info.s-1,rall,rall);

  // Dimensions of the image at the octave associated with the keypoint
  const int H = gmag.extent(0);
  const int W = gmag.extent(1);

  // Coordinates and sigma wrt. to the image size at the octave associated with the keypoint
  const double factor = pow(2., m_gss->getOctaveMin()+(double)keypoint_info.o);
  const double sigma = keypoint.sigma / factor;
  const double yc = keypoint.y / factor;
  const double xc = keypoint.x / factor;

  // Cosine and sine of the keypoint orientation
  const double cosk = cos(keypoint.orientation);
  const double sink = sin(keypoint.orientation);

  // Each local spatial histogram has an extension hist_width = MAGNIF*sigma 
  // pixels. Furthermore, the concatenated histogram has a spatial support of
  // hist_width * DESCR_NBLOCKS pixels. Because of the interpolation, 1 extra
  // pixel might be used, leading to hist_width * (DESCR_NBLOCKS+1). Finally,
  // this square support might be arbitrarily rotated, leading to an effective
  // support of sqrt(2) * hist_width * (DESCR_NBLOCKS+1).
  const double hist_width = m_descr_magnif * sigma;
  const int descr_radius = (int)floor(sqrt(2)*hist_width*(m_descr_n_blocks+1)/2. + 0.5);
  const double window_factor = 0.5 / (m_descr_gaussian_window_size*m_descr_gaussian_window_size);
  static const double two_pi = 2.*M_PI;

  // Determines boundaries to make sure that we remain on the image while
  // computing the descriptor
  const int yci = (int)floor(yc+0.5);
  const int xci = (int)floor(xc+0.5);

  const int dymin = std::max(-descr_radius,1-yci);
  const int dymax = std::min(descr_radius,H-2-yci);
  const int dxmin = std::max(-descr_radius,1-xci);
  const int dxmax = std::min(descr_radius,W-2-xci);
  
  // Loop over the pixels
  // Initializes descriptor to zero
  dst = 0.;
  for (int dyi=dymin; dyi<=dymax; ++dyi)
    for (int dxi=dxmin; dxi<=dxmax; ++dxi)
    {
      // Current integer indices
      int yi = yci + dyi;
      int xi = xci + dxi;
      // Values of the current gradient (magnitude and orientation)
      double mag = gmag(yi,xi);
      double ori = gor(yi,xi);
      // Angle between keypoint orientation and gradient orientation
      double theta = fmod(ori-keypoint.orientation, two_pi);
      if (theta < 0.) theta += two_pi;
      if (theta >= two_pi) theta -= two_pi;

      // Current floating point offset wrt. descriptor center
      double dy = yi - yc;
      double dx = xi - xc;

      // Normalized offset wrt. the keypoint orientation, offset and scale
      double ny = (-sink*dx + cosk*dy) / hist_width;
      double nx = ( cosk*dx + sink*dy) / hist_width;
      double no = (theta / two_pi) * m_descr_n_bins;
     
      // Gaussian weight for the current pixel 
      double window_value = exp(-(nx*nx+ny*ny)*window_factor);

      // Indices of the first bin used in the interpolation
      // Substract -0.5 before flooring such as the weight rbiny=0.5 when 
      // we are between the two centered pixels (assuming that DESCR_NBLOCKS
      // is {equal to 4/even}), for which ny=0.
      // (ny=0-> rbiny=0.5 -> (final) biny = DESCR_NBLOCKS/2-1 (which
      // corresponds to the left centered pixel)
      int biny = (int)floor(ny-0.5);
      int binx = (int)floor(nx-0.5);
      int bino = (int)floor(no);
      double rbiny = ny - (biny + 0.5);
      double rbinx = nx - (binx + 0.5);
      double rbino = no - bino;
      // Make indices start at 0
      biny += m_descr_n_blocks/2;
      binx += m_descr_n_blocks/2;
      
      for (int dbiny=0; dbiny<2; ++dbiny)
      {
        int biny_ = biny+dbiny;
        if (biny_ >= 0 && biny_ < (int)m_descr_n_blocks)
        {
          double wy = ( dbiny==0 ? fabs(1.-rbiny) : fabs(rbiny) );
          for (int dbinx=0; dbinx<2; ++dbinx)
          {
            int binx_ = binx+dbinx;
            if (binx_ >= 0 && binx_ < (int)m_descr_n_blocks)
            {
              double wx = ( dbinx==0 ? fabs(1.-rbinx) : fabs(rbinx) );
              for (int dbino=0; dbino<2; ++dbino)
              {
                double wo = ( dbino==0 ? fabs(1.-rbino) : fabs(rbino) );
                dst(biny_, binx_, (bino+dbino) % (int)m_descr_n_bins) += window_value * mag * wy * wx * wo;
              }
            }
          }
        }
      }
    }

  // Normalization
  double norm = sqrt(blitz::sum(blitz::pow2(dst))) + m_norm_eps;
  dst /= norm;
  // Clip values above norm threshold
  dst = blitz::where(dst > m_norm_thres, m_norm_thres, dst);
  // Renormalize
  norm = sqrt(blitz::sum(blitz::pow2(dst))) + m_norm_eps;
  dst /= norm;
}

void bob::ip::SIFT::computeKeypointInfo(const bob::ip::GSSKeypoint& keypoint,
  bob::ip::GSSKeypointInfo& keypoint_i) const
{
  const int No = (int)getNOctaves();
  const int Ns = (int)getNIntervals();
  const int& omin = getOctaveMin();

  // sigma_{o,s} = sigma0 * 2^{o+s/N_SCALES}, where
  //   o is the octave index, and s the scale index
  // Define phi = log2(sigma_{o,s} / sigma0) = o+s/N_SCALES
  const double phi = log(keypoint.sigma / getSigma0()) / log(2.);

  // A. Octave index
  // Use -0.5/NIntervals term in order to center around scales of indices [1,S]
  // TODO: check if +0.5/Ns or 0!
  int o = (int)floor(phi + 0.5/Ns);
  // Check boundaries
  if ( o< omin) o = omin; // min
  if (o > omin+No-1) o = omin+No-1; // max
  keypoint_i.o = o-omin;

  // B. Scale index
  // Adds 1 after the flooring for the conversion of the scale location into 
  // a scale index (first scale is located at -1 in the GSS pyramid)
  size_t s = (int)floor(Ns*(phi-o) + 0.5) + 1;
  if (s < 1) s = 1; // min
  if (s > (size_t)Ns) s = Ns; // max
  keypoint_i.s = s;

  // C. (y,x) coordinates
  const double factor = pow(2.,o);
  keypoint_i.iy = (int)floor(keypoint.y/factor + 0.5);
  keypoint_i.ix = (int)floor(keypoint.x/factor + 0.5);
}
