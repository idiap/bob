/**
 * @file src/ip/ipLBPTop.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief This class can be used to calculate the LBP-Top (c.f. Dynamic Texture
 * Recognition Using Local Binary Patterns with an Application to Facial
 * Expression from Zhao & Pietikäinen, IEEE Trans. on PAMI, 2007)
 */

#ifndef TORCH_IPLBPTOP_H 
#define TORCH_IPLBPTOP_H

namespace Torch {

  /**
   * The ipLBPTop class is designed to calculate the LBP-Top coefficients given
   * a set of images. 
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
  class ipLBPTop {

    public:
  
    /**
     * Constructs a new ipLBPTop object starting from the algorithm
     * configuration. Please note this object will always produce rotation
     * invariant 2D codes, also taking into consideration pattern uniformity
     * (u2 variant).
     *
     * @param radius_xy The radius to use for the calculation of the 2D LBP on
     * the XY plane (frame)
     * @param points_xy The number of points to use for the calculation of the
     * 2D LBP on the XY plane (frame)
     * @param radius_xt The radius to use for the calculation of the 2D LBP on
     * the XT plane
     * @param points_xt The number of points to use for the calculation of the
     * 2D LBP on the XT plane
     * @param radius_yt The radius to use for the calculation of the 2D LBP on
     * the YT plane
     * @param points_yt The number of points to use for the calculation of the
     * 2D LBP on the YT plane
     */
    ipLBPTop(int radius_xy, int points_xy, 
             int radius_xt, int points_xt, 
             int radius_yt, int points_yt);

    /**
     * Destructor
     */
    virtual ~ipLBPTop();

    /**
     * Pushes a new image into this operator
     *
     * @param i The new image to be pushed in.
     */
    bool pushImage(const Torch::Image* i);

    /**
     * Returns an Image with the pixel values corresponding to the output of
     * the LBP operator in the plane XY, for the center image, i.e., the image
     * that is in the exact middle of the internal FIFO.
     */
    const Torch::Image& lbpXY() const;

    /**
     * Returns an Image with the pixel values corresponding to the output of
     * the LBP operator in the plane XT, for the center image, i.e., the image
     * that is in the exact middle of the internal FIFO.
     */
    const Torch::Image& lbpXT() const;

    /**
     * Returns an Image with the pixel values corresponding to the output of
     * the LBP operator in the plane YT, for the center image, i.e., the image
     * that is in the exact middle of the internal FIFO.
     */
    const Torch::Image& lbpYT() const;

    private:

    int m_radius_xy; ///< The LBPu2,i radius in XY
    int m_points_xy; ///< The number of points in the XY LBPu2,i
    int m_radius_xt; ///< The LBPu2,i radius in XT
    int m_points_xt; ///< The number of points in the XT LBPu2,i
    int m_radius_yt; ///< The LBPu2,i radius in YT
    int m_points_xt; ///< The number of points in the YT LBPu2,i
    const Torch::Image* m_fifo; ///< FIFO of running images

  };

}

#endif /* TORCH_IPLBPTOP_H */

