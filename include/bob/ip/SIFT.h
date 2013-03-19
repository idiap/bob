/**
 * @file bob/ip/SIFT.h
 * @date Sun Sep 9 19:21:00 2012 +0200
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

#ifndef BOB_IP_SIFT_H
#define BOB_IP_SIFT_H

#include <blitz/array.h>
#include <bob/ip/GaussianScaleSpace.h>
#include <bob/ip/BlockCellGradientDescriptors.h>
#include <bob/sp/conv.h>
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
 * @brief This class can be used to extract SIFT descriptors
 */
class SIFT
{
  public:
    /**
     * @brief Constructor: generates a SIFT extractor
     */
    SIFT(const size_t height, const size_t width, const size_t n_octaves, 
        const size_t n_intervals, const int octave_min,
        const double sigma_n=0.5, const double sigma0=1.6, 
        const double contrast_thres=0.03, const double edge_thres=10., 
        const double norm_thres=0.2, const double kernel_radius_factor=4.,
        const bob::sp::Extrapolation::BorderType border_type = 
        bob::sp::Extrapolation::Mirror);

    /**
     * @brief Copy constructor
     */
    SIFT(const SIFT& other);

    /**
     * @brief Destructor
     */
    virtual ~SIFT();

    /**
     * @brief Assignment operator
     */
    SIFT& operator=(const SIFT& other);

    /**
     * @brief Equal to
     */
    bool operator==(const SIFT& b) const;
    /**
     * @brief Not equal to
     */
    bool operator!=(const SIFT& b) const; 

    /**
     * @brief Getters
     */
    size_t getHeight() const { return m_gss->getHeight(); }
    size_t getWidth() const { return m_gss->getWidth(); }
    size_t getNOctaves() const { return m_gss->getNOctaves(); }
    size_t getNIntervals() const { return m_gss->getNIntervals(); }
    int getOctaveMin() const { return m_gss->getOctaveMin(); }
    int getOctaveMax() const { return m_gss->getOctaveMax(); }
    double getSigmaN() const { return m_gss->getSigmaN(); }
    double getSigma0() const { return m_gss->getSigma0(); }
    double getKernelRadiusFactor() const 
    { return m_gss->getKernelRadiusFactor(); }
    bob::sp::Extrapolation::BorderType getConvBorder() const 
    { return m_gss->getConvBorder(); }
    double getContrastThreshold() const { return m_contrast_thres; }
    double getEdgeThreshold() const { return m_edge_thres; }
    double getNormThreshold() const { return m_norm_thres; }
    size_t getNBlocks() const { return m_descr_n_blocks; }
    size_t getNBins() const { return m_descr_n_bins; }
    double getGaussianWindowSize() const 
    { return m_descr_gaussian_window_size; }
    double getMagnif() const { return m_descr_magnif; }
    double getNormEpsilon() const { return m_norm_eps; }

    /**
     * @brief Setters
     */
    void setHeight(const size_t height) 
    { m_gss->setHeight(height); }
    void setWidth(const size_t width) 
    { m_gss->setWidth(width); }
    void setNOctaves(const size_t n_octaves) 
    { m_gss->setNOctaves(n_octaves); }
    void setNIntervals(const size_t n_intervals) 
    { m_gss->setNIntervals(n_intervals); } 
    void setOctaveMin(const int octave_min) 
    { m_gss->setOctaveMin(octave_min); }
    void setSigmaN(const double sigma_n) 
    { m_gss->setSigmaN(sigma_n); }
    void setSigma0(const double sigma0) 
    { m_gss->setSigma0(sigma0); }
    void setKernelRadiusFactor(const double kernel_radius_factor) 
    { m_gss->setKernelRadiusFactor(kernel_radius_factor); }
    void setConvBorder(const bob::sp::Extrapolation::BorderType border_type)
    { m_gss->setConvBorder(border_type); }
    void setContrastThreshold(const double threshold) 
    { m_contrast_thres = threshold; }
    void setEdgeThreshold(const double threshold) 
    { m_edge_thres = threshold; updateEdgeEffThreshold(); }
    void setNormThreshold(const double threshold) 
    { m_norm_thres = threshold; }
    void setNBlocks(const size_t n_blocks)
    { m_descr_n_blocks = n_blocks; }
    void setNBins(const size_t n_bins)
    { m_descr_n_bins = n_bins; }
    void setGaussianWindowSize(const double size)
    { m_descr_gaussian_window_size = size; }
    void setMagnif(const double magnif)
    { m_descr_magnif = magnif; }
    void setNormEpsilon(const double norm_eps)
    { m_norm_eps = norm_eps; }

    /** 
     * @brief  Automatically sets sigma0 to a value such that there is no
     * smoothing initially. sigma0 is then set such that the sigma value for
     * the first scale (index -1) of the octave octave_min is equal to 
     * sigma_n*2^(-octave_min).
     */ 
    void setSigma0NoInitSmoothing()
    { m_gss->setSigma0NoInitSmoothing(); }

    /**
     * @brief Compute SIFT descriptors for the given keypoints
     * @param src The 2D input blitz array/image
     * @param keypoints The keypoints
     * @param dst The descriptor for the keypoints
     */
    template <typename T>
    void computeDescriptor(const blitz::Array<T,2>& src, 
      const std::vector<boost::shared_ptr<bob::ip::GSSKeypoint> >& keypoints,
      blitz::Array<double,4>& dst);

    /**
     * @brief Get the shape of a descriptor for a given keypoint (y,x,orientation)
     */
    const blitz::TinyVector<int,3> getDescriptorShape() const;

  private:
    /**
     * @brief Resets the cache
     */
    void resetCache();

    /**
     * @brief Recomputes the value effectively used in the edge-like rejection
     * from the curvature/edge threshold
     */
    void updateEdgeEffThreshold()
    { m_edge_eff_thres = (m_edge_thres+1.)*(m_edge_thres+1.)/m_edge_thres; }

    /**
     * @brief Get the size of Gaussian filtered images for a given octave
     */
    const blitz::TinyVector<int,3> 
    getGaussianOutputShape(const int octave) const;

    /**
     * @brief Computes the Gaussian pyramid
     */
    template <typename T> 
    void computeGaussianPyramid(const blitz::Array<T,2>& src);
    /**
     * @brief Computes the Difference of Gaussians pyramid
     * @warning assumes that the Gaussian pyramid has already been computed
     */
    void computeDog();

    /**
     * @brief Computes gradients from the Gaussian pyramid
     */
    void computeGradient();

    /**
     * @brief Compute SIFT descriptors for the given keypoints
     * @param keypoints The keypoints
     * @param dst The descriptor for the keypoints
     * @warning Assume that the Gaussian scale-space is already in cache
     */
    void computeDescriptor(const std::vector<boost::shared_ptr<bob::ip::GSSKeypoint> >& keypoints,
      blitz::Array<double,4>& dst) const;
    /**
     * @brief Compute SIFT descriptor for a given keypoint
     */
    void computeDescriptor(const bob::ip::GSSKeypoint& keypoint, 
      const bob::ip::GSSKeypointInfo& keypoint_i, blitz::Array<double,3>& dst) const;
    void computeDescriptor(const bob::ip::GSSKeypoint& keypoint, 
      blitz::Array<double,3>& dst) const;
    /**
     * @brief Compute SIFT keypoint additional information, from a regular
     * SIFT keypoint
     */
    void computeKeypointInfo(const bob::ip::GSSKeypoint& keypoint, 
      bob::ip::GSSKeypointInfo& keypoint_info) const;


    /**
     * Attributes
     */
    boost::shared_ptr<bob::ip::GaussianScaleSpace> m_gss;
    double m_contrast_thres; //< Threshold for low-contrast keypoint rejection
    double m_edge_thres; //< Threshold (for the ratio of principal curvatures) for edge-like keypoint rejection
    double m_edge_eff_thres; //< Effective threshold for edge-like keypoint rejection.
    double m_norm_thres; //< Threshold used to clip high values during the descriptor normalization step
    // This is equal to the (r+1)^2/r, r being the regular edge threshold.

    size_t m_descr_n_blocks;
    size_t m_descr_n_bins;
    double m_descr_gaussian_window_size;
    double m_descr_magnif;
    double m_norm_eps;

    /**
     * Cache
     */
    std::vector<blitz::Array<double,3> > m_gss_pyr;
    std::vector<blitz::Array<double,3> > m_dog_pyr;
    std::vector<blitz::Array<double,3> > m_gss_pyr_grad_mag;
    std::vector<blitz::Array<double,3> > m_gss_pyr_grad_or;
    std::vector<boost::shared_ptr<bob::ip::GradientMaps> > m_gradient_maps;
   

    /**
     * For testing purposes only
     */
    friend class SIFTtest;
    void setGaussianPyramid(const std::vector<blitz::Array<double,3> >& gss_pyr)
    { m_gss_pyr = gss_pyr; }
};

template <typename T>
void bob::ip::SIFT::computeDescriptor(const blitz::Array<T,2>& src,
  const std::vector<boost::shared_ptr<bob::ip::GSSKeypoint> >& keypoints,
  blitz::Array<double,4>& dst)
{
  // Computes the Gaussian pyramid
  computeGaussianPyramid(src);
  // Computes the Difference of Gaussians pyramid
  computeDog();
  // Computes the Gradient of the Gaussians pyramid
  computeGradient();
  // Computes the descriptors for the given keypoints
  computeDescriptor(keypoints, dst);
}

template <typename T>
void bob::ip::SIFT::computeGaussianPyramid(const blitz::Array<T,2>& src)
{
  // Computes the Gaussian pyramid
  m_gss->operator()(src, m_gss_pyr);
}

}}

#endif /* BOB_IP_SIFT_H */
