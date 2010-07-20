/**
 * @file src/ip/VideoTensor.h
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief Implements the VideoTensor methods
 */

#include "VideoTensor.h"
#include "Convert.h"

Torch::VideoTensor::VideoTensor (Torch::Video& video, int color_planes)
  :ShortTensor(video.getIOption("height"), 
               video.getIOption("width"),
               color_planes,
               video.getNFrames())
{
  Torch::Image image(1, 1, color_planes);
  int frame = 0;
	while (video.read(image) == true) {
    this->setFrame(image, frame++);
  }
}

Torch::VideoTensor::VideoTensor (Torch::TensorFile& tensor_file)
  :ShortTensor(tensor_file.getHeader().m_size[0],
               tensor_file.getHeader().m_size[1],
               tensor_file.getHeader().m_size[2],
               tensor_file.getHeader().m_size[3])
{
  tensor_file.load(*this);
}

Torch::VideoTensor::VideoTensor (int width, int height, int color_planes, int frames)
  :ShortTensor(height, width, color_planes, frames)
{
}

Torch::VideoTensor::VideoTensor (const Torch::VideoTensor& video_tensor)
  :ShortTensor(video_tensor.size(0),
               video_tensor.size(1),
               video_tensor.size(2),
               video_tensor.size(3))
{
  this->copy(&video_tensor);
}

Torch::VideoTensor& Torch::VideoTensor::operator=(const Torch::VideoTensor& vt)
{
  this->copy(&vt);
  return *this;
}

bool Torch::VideoTensor::getFrame(Torch::Image& image, int frame)
{
  if ((image.getHeight() != this->size(0)) ||
      (image.getWidth() != this->size(1)) || 
      (frame >= this->size(3))) {
    //avoids straight exit from torch!
    return false;
  }
  Torch::Image t;
  t.select(this, 3, frame); //reads: select from this, set into self
  image.copyFrom(t);
  return true;
}

bool Torch::VideoTensor::setFrame(const Torch::Image& image, int frame)
{
  if ((image.getHeight() != this->size(0)) ||
      (image.getWidth() != this->size(1)) || 
      (frame >= this->size(3))) {
    //avoids straight exit from torch!
    return false;
  }
  Torch::Image t;
  t.select(this, 3, frame);
  t.copyFrom(image);
  return true;
}

bool Torch::VideoTensor::save(Torch::TensorFile& tensor_file) const 
{
  return tensor_file.save(*this);
}

bool Torch::VideoTensor::save(Torch::Video& video) const
{
  Torch::Image tmp(this->size(1), this->size(0), 3); //always color for video
  for(unsigned i=0; i<this->size(3); ++i) {
    if (!const_cast<Torch::VideoTensor*>(this)->getFrame(tmp, i)) return false;
    if (!video.write(tmp)) return false;
  }
  return true;
}
