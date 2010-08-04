/**
 * @file src/ip/VideoTensor.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief As an Image, VideoTensor represents a Video object as a Tensor and
 * tries to re-adapt the Torch base concepts to video streams.
 */

#ifndef TORCH_VIDEOTENSOR_H 
#define TORCH_VIDEOTENSOR_H

#include "core/Tensor.h"
#include "core/TensorFile.h"
#include "Image.h"
#include "Video.h"

namespace Torch {

  /**
   * A video tensor is a representation of a video sequence as a Torch::Tensor.
   * We follow the same implementation strategy as for Torch::Image, choosing a
   * Torch::ShortTensor as a representation for the image sequence. We add a
   * few methods to render the handling of videos and images a bit more
   * pleasant to the end-user.
   */
  class VideoTensor: public Torch::ShortTensor {

    public:

    /**
     * Builds a new VideoTensor from an existing Video object.
     *
     * @param video The video object that will be read and from which we are
     * loading the image sequence.
     * @param color_planes The number of color planes we need for the tensor.
     * The default is 3 (RGB video), but you can cause color squashing if you
     * specify 1. Please note that any other value will not work.
     */
    VideoTensor (Torch::Video& video, int color_planes=3);

    /**
     * Copies the data from another VideoTensor
     *
     * @param video_tensor The input video tensor to copy the data from.
     */
    VideoTensor (const Torch::VideoTensor& other);

    /**
     * Builds a new VideoTensor from data existing in a TensorFile. This method
     * will read the next available tensor in the file.
     *
     * @param tensor_file The tensor file to be used for reading the data.
     */
    VideoTensor (Torch::TensorFile& tensor_file);

    /**
     * Builds a new VideoTensor specifying the size, number of color planes and
     * total number of frames
     *
     * @param width The width of each individual frame
     * @param height The height of each individual frame
     * @param color_planes The number of color planes to use
     * @param frames The number of frames this video sequence will have
     */
    VideoTensor (int width, int height, int color_planes, int frames);

    /**
     * Copies the data from another VideoTensor. Please note that in this case,
     * the dimensions of the video tensor must match.
     *
     * @param other The other video tensor that will be copied
     *
     * @return myself
     */
    VideoTensor& operator=(const Torch::VideoTensor& other);

    /**
     * Destructor
     */
    virtual ~VideoTensor() {}

    /**
     * Sets the image object to the i-th image on the sequence (this will set
     * by reference, so it is fast!). 
     *
     * @param image The image where to set the selected frame
     * @param frame The 0-based frame number of the image you want to retrieve.
     *
     * @return true if I was able to retrieve the given frame. false otherwise.
     */
    bool getFrame(Torch::Image& image, int frame);

    /**
     * Resets a certain image in the video sequence to the value given as
     * input. Please note that the image specifications (width and height)
     * should respect the values in the video tensor. If the number of planes
     * varies between this and the source image, the adaptation also found in
     * Image::copyFrom() will be used.
     *
     * @param image The image to be set in place
     * @param frame Which frame to set
     *
     * @return true if the setting happened fine.
     */
    bool setFrame(const Torch::Image& image, int frame);

    /**
     * Saves the current video tensor in a already opened TensorFile 
     *
     * @param tensor_file The tensor file where to save myself at.
     *
     * @return true if the saving was done correctly
     */
    bool save(Torch::TensorFile& tensor_file) const;

    /**
     * Saves the current video tensor in a Video file.
     *
     * @param video The video file to output the data.
     *
     * @return true if the saving was done correctly
     */
    bool save(Torch::Video& video) const;

  };

}

#endif /* TORCH_VIDEOTENSOR_H */

