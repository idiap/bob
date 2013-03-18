/**
 * @file bob/ip/GaussianScaleSpace.h
 * @date Thu Apr 7 19:52:29 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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

#ifndef BOB_IP_GAUSSIANSCALESPACE_H
#define BOB_IP_GAUSSIANSCALESPACE_H

#include <blitz/array.h>
#include <bob/ip/Gaussian.h>
#include <bob/core/assert.h>

#include <boost/shared_ptr.hpp>
#include <vector>

namespace bob {
/**
 * \ingroup libip_api
 * @{
 *
 */
namespace ip {

/** 
 * @brief Structure for the GSS keypoints (location)
 */
typedef struct GSSKeypoint_
{
  GSSKeypoint_(const double s, const double y_, const double x_, 
    const double o_=0.): 
    sigma(s), y(y_), x(x_), orientation(o_)
  {   
  }

  double sigma; //< sigma
  double y; //< y-coordinate
  double x; //< x-coordinate
  double orientation; //< orientation (0 if not estimated)
} GSSKeypoint;

/** 
 * @brief Additional structure for the GSS keypoints (location)
 * This keeps track of more detailed information. This can be useful for 
 * paramater tuning.
 */
typedef struct GSSKeypointInfo_
{
  GSSKeypointInfo_(const size_t o_, const size_t s_, const int iy_, 
    const int ix_, const double peak_score_=0., const double edge_score_=0.):
    o(o_), s(s_), iy(iy_), ix(ix_), peak_score(peak_score_),
    edge_score(edge_score_)
  {   
  }

  size_t o; //< octave index in the vector of octaves
  size_t s; //< scale index in the vector of the GSS pyramid
  int iy; //< integer unnormalized y coordinate
  int ix; //< integer unnormalized x coordinate
  double peak_score; // score of the peak (D(x) in section 4 of Lowe's paper)
  double edge_score; // score of the edge response (ratio Tr(H)^2/det(H) in section 4.1 of Lowe's paper)
} GSSKeypointInfo;


/**
 * @brief This class can be used to extract a Gaussian Scale Space
 * Reference: "Distinctive Image Features from Scale-Invariant Keypoints", 
 * D. Lowe, International Journal of Computer Vision, 2004.
 */
class GaussianScaleSpace
{
  public:
    /**
     * @brief Constructor: generates a GaussianScaleSpace extractor
     * @param height Height of the images to process
     * @param width Width of the images to process
     * @param n_octaves Number of octaves. All the images of a given octave
     *   have the same dimensions, but different scales (smoothing levels).
     * @param n_intervals Number of intervals per octave. SIFT extraction
     *   requires additional (n_invervals+3) scales per octave, because of
     *   the use of Difference of Gaussians (DoG) during detection, and of
     *   scale interpolation during descriptor computation.
     * @param octave_min Index of the first octave. It should be in the 
     *   range [-1,+inf]. If set to -1, the image will be upsampled in
     *   the first octave, if set to 0 the initial image dimensions will
     *   be used in the first octave, if set to 1, the image will be 
     *   downsampled, etc.
     * @param sigma_n Input image is assumed to be smoothed at this sigma
     *   level (to take finite resolution into consideration)
     * @param sigma0 Sigma of the base scale (octave 0 and scale -1). The
     *   sigma smoothing coefficient of the image of octave o at scale s
     *   is equal to sigma0.2^{o+s/n_intervals}
     * @param kernel_radius_factor Factor used to compute the radius of any
     *   Gaussian kernel. The radius of the Gaussian kernel with a 
     *   standard deviation/sigma will be equal to 
     *   radius=ceil(kernel_radius_factor*sigma), the size of the kernel 
     *   being: size=2*radius + 1 = 2*ceil(kernel_radius_factor*sigma) + 1.
     *   A value of 3 leads to a support that contains more than 99.6% of 
     *   the energy of the perfect continuous filter. Large values lead to 
     *   large kernels, and might cause run-time exceptions, as our
     *   implementation requires kernels too be smaller than inputs while 
     *   performing convolutions.
     * @param border_type The way we deal with convolution at the boundaries
     *   of the input image.
     */
    GaussianScaleSpace(const size_t height, const size_t width,
      const size_t n_octaves, const size_t n_intervals, const int octave_min,
      const double sigma_n=0.5, const double sigma0=1.6,
      const double kernel_radius_factor=4.,
      const bob::sp::Extrapolation::BorderType border_type = 
        bob::sp::Extrapolation::Mirror);

    /**
     * @brief Copy constructor
     */
    GaussianScaleSpace(const GaussianScaleSpace& other);

    /**
     * @brief Destructor
     */
    virtual ~GaussianScaleSpace();

    /**
     * @brief Assignment operator
     */
    GaussianScaleSpace& operator=(const GaussianScaleSpace& other);

    /**
     * @brief Equal to
     */
    bool operator==(const GaussianScaleSpace& b) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const GaussianScaleSpace& b) const; 

    /**
     * @brief Getters
     */
    size_t getHeight() const { return m_height; }
    size_t getWidth() const { return m_width; }
    size_t getNOctaves() const { return m_n_octaves; }
    size_t getNIntervals() const { return m_n_intervals; }
    int getOctaveMin() const { return m_octave_min; }
    int getOctaveMax() const { return m_octave_min+(int)m_n_octaves-1; }
    double getSigmaN() const { return m_sigma_n; }
    double getSigma0() const { return m_sigma0; }
    double getKernelRadiusFactor() const { return m_kernel_radius_factor; }
    bob::sp::Extrapolation::BorderType getConvBorder() const 
    { return m_conv_border; }
    boost::shared_ptr<bob::ip::Gaussian> getGaussian(const size_t i) const 
    { return m_gaussians[i]; }

    /**
     * @brief Setters
     */
    void setHeight(const size_t height) 
    { m_height = height; }
    void setWidth(const size_t width) 
    { m_width = width; }
    void setNOctaves(const size_t n_octaves) 
    { m_n_octaves = n_octaves; resetGaussians(); }
    void setNIntervals(const size_t n_intervals) 
    { m_n_intervals = n_intervals; resetGaussians(); }
    void setOctaveMin(const int octave_min) 
    { m_octave_min = octave_min; checkOctaveMin(); resetGaussians(); }
    void setSigmaN(const double sigma_n) 
    { m_sigma_n = sigma_n; resetGaussians(); }
    void setSigma0(const double sigma0) 
    { m_sigma0 = sigma0; resetGaussians(); }
    void setKernelRadiusFactor(const double kernel_radius_factor) 
    { m_kernel_radius_factor = kernel_radius_factor; resetGaussians(); }
    void setConvBorder(const bob::sp::Extrapolation::BorderType border_type)
    { m_conv_border = border_type; resetGaussians(); }

    /**
     * Automatically sets sigma0 to a value such that there is no smoothing
     * initially. sigma0 is then set such that the sigma value for the 
     * first scale (index -1) of the octave octave_min is equal to 
     * sigma_n*2^(-octave_min).
     */ 
    void setSigma0NoInitSmoothing();

    /**
     * @brief Process a 2D blitz Array/Image by extracting a Gaussian Pyramid.
     * @param src The 2D input blitz array
     * @param dst A vector of 3D blitz Arrays. Each octave is described by
     *   one element of the vector. The bliz Arrays should have the 
     *   expected size.
     */
    template <typename T> 
    void operator()(const blitz::Array<T,2>& src, std::vector<blitz::Array<double,3> >& dst) const;

    /**
     * @brief Allocate output vector of blitz Arrays.
     * @param dst A vector of 3D blitz Arrays. Previous content will be erased.
     *   New blitz Arrays of suitable sizes will be allocated and will populate the vector.
     */
    void allocateOutputPyramid(std::vector<blitz::Array<double,3> >& dst) const;

    /**
     * @brief Returns the output shape for a given octave. 
     * @param octave The index of the octave. This should be in the range 
     *   [octave_min, octave_min+N_octaves]
     * @return A TinyVector with the length of each of the three dimensions.
     *   The first dimension corresponds to the number of scales/images at 
     *   this octave. This is equal to Nintervals+3. The last two dimensions
     *   are the height and width of Gaussian filtered images at this octave.
     */
    const blitz::TinyVector<int,3> getOutputShape(const int octave) const;

  private:
    /**
     * Attributes
     */
    size_t m_height;
    size_t m_width;
    size_t m_n_octaves;
    size_t m_n_intervals;
    int m_octave_min;
    double m_sigma_n;
    double m_sigma0;
    double m_kernel_radius_factor;
    bob::sp::Extrapolation::BorderType m_conv_border;

    std::vector<boost::shared_ptr<bob::ip::Gaussian> > m_gaussians;
    bool m_smooth_at_init;

    /**
     * Working arrays/variables in cache
     */
    mutable blitz::Array<double,2> m_cache_array0;
    void resetCache() const;
    void resetGaussians();

    /**
     * Checks that minimum octave index is in the range [-1,+infty], and
     * throws an exception otherwise.
     */
    void checkOctaveMin() const;
};


namespace detail 
{
  template <typename T>
  void upsample(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst)
  {
    // Check dimensions
    bob::core::array::assertSameDimensionLength(src.extent(0)*2, dst.extent(0));
    bob::core::array::assertSameDimensionLength(src.extent(1)*2, dst.extent(1));

    // Define used ranges
    blitz::Range rdst_y0(0, dst.extent(0)-2, 2);
    blitz::Range rdst_y1(1, dst.extent(0)-1, 2);
    blitz::Range rdst_y1m(1, dst.extent(0)-3, 2);
    blitz::Range rdst_x0(0, dst.extent(1)-2, 2);
    blitz::Range rdst_x1(1, dst.extent(1)-1, 2);
    blitz::Range rdst_x1m(1, dst.extent(1)-3, 2);
    blitz::Range rsrc_y0(0, src.extent(0)-2);
    blitz::Range rsrc_y1(1, src.extent(0)-1);
    blitz::Range rsrc_x0(0, src.extent(1)-2);
    blitz::Range rsrc_x1(1, src.extent(1)-1);
    blitz::Range rall = blitz::Range::all();

    // Non interpolated values
    blitz::Array<double,2> dst1 = dst(rdst_y0, rdst_x0);
    dst1 = src;

    // Interpolated values
    blitz::Array<double,2> dst2 = dst(rdst_y0, rdst_x1m);
    dst2 = 0.5 * (src(rall, rsrc_x0) + src(rall, rsrc_x1));
    blitz::Array<double,2> dst3 = dst(rdst_y1m, rdst_x0);
    dst3 = 0.5 * (src(rsrc_y0, rall) + src(rsrc_y1, rall));
    blitz::Array<double,2> dst4 = dst(rdst_y1m, rdst_x1m);
    dst4 = 0.5 * (dst3(rall, rsrc_x0) + dst3(rall, rsrc_x1)); // = 0.5 * (dst2(rsrc_y0, rall) + dst2(rsrc_y1, rall))

    // Right and bottom borders
    dst(rall, dst.extent(1)-1) = dst(rall, dst.extent(1)-2);
    dst(dst.extent(0)-1, rall) = dst(dst.extent(0)-2, rall);
  }

  template <typename T>
  void downsample(const blitz::Array<T,2>& src, blitz::Array<double,2>& dst, 
    const size_t d)
  {
    // Checks dimensions
    const int factor = (1 << d);
    const int l_dst0 = src.extent(0) / factor;
    const int l_dst1 = src.extent(1) / factor;
    bob::core::array::assertSameDimensionLength(dst.extent(0), l_dst0);
    bob::core::array::assertSameDimensionLength(dst.extent(1), l_dst1);

    // Defines used ranges
    blitz::Range rsrc_y(0, factor*(dst.extent(0)-1), factor);
    blitz::Range rsrc_x(0, factor*(dst.extent(1)-1), factor);

    // Updates dst values
    dst = src(rsrc_y, rsrc_x);
  }
}


template <typename T>
void bob::ip::GaussianScaleSpace::operator()(const blitz::Array<T,2>& src, 
  std::vector<blitz::Array<double,3> >& dst) const
{
  // Checks
  bob::core::array::assertZeroBase(src);
  for (size_t i=0; i<dst.size(); ++i)
    bob::core::array::assertZeroBase(dst[i]);

  for (size_t i=0; i<dst.size(); ++i)
  {
    const blitz::TinyVector<int,3> shape = getOutputShape(m_octave_min+i);
    bob::core::array::assertSameShape(dst[i], shape);
  }

  if (m_octave_min < 0)
    bob::ip::detail::upsample(src, m_cache_array0);
  else if (m_octave_min > 0)
    bob::ip::detail::downsample(src, m_cache_array0, m_octave_min);
  else // 0
    m_cache_array0 = src;

  blitz::Range rall = blitz::Range::all();
  // Iterates over the scales
  for (size_t o=0; o<m_n_octaves; ++o)
  {
    blitz::Array<double,2> dst_m1 = dst[o](0, rall, rall);
    if (o==0) {
      if (m_smooth_at_init)
        m_gaussians[0]->operator()(m_cache_array0, dst_m1);
      else
        dst_m1 = m_cache_array0;
    }
    else {
      // Copy from previous octave and downsample
      blitz::Array<double,2> dst_prev = dst[o-1]((int)m_n_intervals, rall, rall);
      bob::ip::detail::downsample(dst_prev, dst_m1, 1);
    }

    for (size_t s=1; s<m_n_intervals+3; ++s)
    {
      blitz::Array<double,2> dst_prev = dst[o](s-1, rall, rall);
      blitz::Array<double,2> dst_cur = dst[o](s, rall, rall);
      m_gaussians[s]->operator()(dst_prev, dst_cur);
    }
  }
}

}}

#endif /* BOB_IP_GAUSSIANSCALESPACE_H */
