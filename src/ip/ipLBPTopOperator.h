/**
 * @file src/ip/ipLBPTopOperator.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class can be used to calculate the LBP-Top  of a set of image
 * frames representing a video sequence (c.f. Dynamic Texture
 * Recognition Using Local Binary Patterns with an Application to Facial
 * Expression from Zhao & Pietikäinen, IEEE Trans. on PAMI, 2007)
 */

#ifndef TORCH_IPLBPTOP_OPERATOR_H 
#define TORCH_IPLBPTOP_OPERATOR_H

#include "core/Object.h"
#include "core/Tensor.h"
#include "Image.h"
#include "ipLBP.h"

namespace Torch {

  /**
   * The ipLBPTopOperator class is designed to calculate the LBP-Top
   * coefficients given a set of images. 
   *
   * The workflow is as follows:
   * 1. You initialize the class, defining the radius and number of points in
   * each of the three directions: XY, XT, YT for the LBP calculations
   * 2. For each image you have in the frame sequence, you push into the class
   * 3. An internal FIFO queue (length = radius in T direction) keeps track 
   * of the current image and their order. As a new image is pushed in, the
   * oldest on the queue is pushed out. 
   * 4. After pushing an image, you read the current LBP-Top coefficients and
   * may save it somewhere.
   */
  class ipLBPTopOperator: public Torch::Object {

    public:
  
    /**
     * Constructs a new ipLBPTop object starting from the algorithm
     * configuration. Please note this object will always produce rotation
     * invariant 2D codes, also taking into consideration pattern uniformity
     * (u2 variant). 
     *
     * The radius in X (width) direction is combied with the radius in the Y
     * (height) direction for the calculation of the LBP on the XY (frame)
     * direction. The radius in T is taken from the number of frames input, so
     * it is dependent on the input to ipLBPTopOperator::process().
     *
     * All input parameters are changeable throught the Torch::Object
     * interface, following the same nomenclature as for the variables in this
     * constructor.
     *
     * @warning The current number of points supported in torch is either 8 or
     * 4. Any values differing from that need implementation of specialized
     * functionality.
     *
     * @param radius_xy The radius to be used at the XY plane
     * @param points_xy The number of points to use for the calculation of the
     * 2D LBP on the XY plane (frame)
     * @param radius_xt The radius to be used at the XT plane
     * @param points_xt The number of points to use for the calculation of the
     * 2D LBP on the XT plane
     * @param radius_yt The radius to be used at the YT plane
     * @param points_yt The number of points to use for the calculation of the
     * 2D LBP on the YT plane
     */
    ipLBPTopOperator(int radius_xy, 
                     int points_xy, 
                     int radius_xt, 
                     int points_xt, 
                     int radius_yt, 
                     int points_yt);

    /**
     * Destructor
     */
    virtual ~ipLBPTopOperator();

    /**
     * Processes a 4D tensor representing a set of <b>grayscale</b> images and
     * returns (by argument) the three LBP planes calculated. The 4D tensor has
     * to be arranged in this way:
     *
     * 1st dimension => frame height
     * 2nd dimension => frame width
     * 3rd dimension => grayscale frame values
     * 4th dimension => time
     *
     * The number of frames in the tensor has to be always an odd number. The
     * central frame is taken as the frame where the LBP planes have to be
     * calculated from. The radius in dimension T (4th dimension) is taken to
     * be (N-1)/2 where N is the number of frames input.
     *
     * @param tensor The input 4D tensor as described in the documentation of
     * this method.
     * @param xy The result of the LBP operator in the XY plane (frame), for
     * the central frame of the input tensor. This is an image.
     * @param xt The result of the LBP operator in the XT plane for the whole
     * image, taking into consideration the size of the width of the input
     * tensor along the time direction. 
     * @param yt The result of the LBP operator in the YT plane for the whole
     * image, taking into consideration the size of the width of the input
     * tensor along the time direction. 
     *
     * @return true if the processing went just fine.
     */
    bool process (const Torch::ShortTensor& tensor,
                  Torch::Image& xy,
                  Torch::Image& xt,
                  Torch::Image& yt) const;

    /**
     * Updates internal variables, using the Torch::Object method.
     *
     * @param name The name of the parameter changed.
     */
    virtual void optionChanged(const char* name);

    private:

    int m_radius_xy; ///< The LBPu2,i radius in XY
    int m_radius_xt; ///< The LBPu2,i radius in XT
    int m_radius_yt; ///< The LBPu2,i radius in YT
    int m_points_xy; ///< The number of points in the XY LBPu2,i
    int m_points_xt; ///< The number of points in the XT LBPu2,i
    int m_points_yt; ///< The number of points in the YT LBPu2,i
    Torch::ipLBP* m_lbp_xy; ///< The operator for the XY calculation
    Torch::ipLBP* m_lbp_xt; ///< The operator for the XT calculation
    Torch::ipLBP* m_lbp_yt; ///< The operator for the YT calculation

  };

}

#endif /* TORCH_IPLBPTOP_OPERATOR_H */
