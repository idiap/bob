/**
 * @file cxx/daq/src/V4LCamera.cc
 * @date Thu Feb 2 11:22:57 2012 +0100
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "daq/V4LCamera.h"
#include <cstring>

#include <cstdio>
#include <assert.h>
#include <pthread.h>

#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/time.h>
#include <sys/mman.h>
#include <sys/ioctl.h>
#include <asm/types.h>
#include <linux/videodev2.h>

namespace bob { namespace daq {

/// Size of the custom char arrays which contains text
static const int TEXT_MAX_SIZE = 1024;
/// Number of buffer requested to the driver
static const unsigned int NB_REQUESTED_BUFFER = 20;
/// Minimal number of buffer needed to start the capture
static const unsigned int MIN_NB_BUFFER = 5;

struct Buffer {
  unsigned char *start;
  size_t length;
};

struct V4LStruct {
  char device_name[TEXT_MAX_SIZE];
  bool opened;

  /// File descriptor of the device
  int device;


  /**
   * @defgroup v4l2_param v4l2 parameters
   *
   * @{
   */
  v4l2_capability cap;
  v4l2_pix_format format;
  v4l2_captureparm param;
  v4l2_requestbuffers reqbuf;
  /**
   * @}
   */

  /**
   * Shared buffer with camera
   * The number of buffers is stored in reqbuf.count
   */
  Buffer* buffers;

  /// Thread id
  pthread_t thread;

  V4LStruct() {
    device_name[0] = '\0';
    opened = false;
    device = 0;
    memset(&cap, 0, sizeof(v4l2_capability));
    memset(&format, 0, sizeof(v4l2_pix_format));
    memset(&param, 0, sizeof(v4l2_captureparm));
    memset(&reqbuf, 0, sizeof(v4l2_requestbuffers));
    buffers = NULL;
    thread = 0;
  }
};


/**
 * ioctl function. Handle the case where an interrupted system call occurs
 * 
 * @see http://v4l2spec.bytesex.org/spec/capture-example.html
 */
static int xioctl(int device, unsigned long request, void *arg) {
  int r;
  do {
    r = ioctl(device, request, arg);
  }
  while(r == -1 && errno == EINTR);
  
  return r;
}

V4LCamera::V4LCamera(const char* device) : v4lstruct(new V4LStruct()), mustStop(false) {
  strncpy(v4lstruct->device_name, device, TEXT_MAX_SIZE);
  
  open();
}

V4LCamera::~V4LCamera() {
  if (v4lstruct->opened) {
    close();
  }

  // TODO Check that a munmap is not required before releasing the buffer array
  if (v4lstruct->buffers != NULL) {
    delete[] v4lstruct->buffers;
  }
  
  delete v4lstruct;
}


int V4LCamera::open() {
  if (!v4lstruct->opened) {
    v4lstruct->device = ::open(v4lstruct->device_name, O_RDWR, 0);

    if (v4lstruct->device == -1) {
      perror("Can't open() Video capture device");
      return -1;
    }

    v4lstruct->opened = true;

    // Read capabilities and check if it's a V4L2 compliant device
    if (xioctl(v4lstruct->device, VIDIOC_QUERYCAP, &v4lstruct->cap) == -1) {
      perror("The device is not a V4L2 device");
      close();
      return -1;
    }

    // Check if the device supports capture
    if (!(v4lstruct->cap.capabilities & V4L2_CAP_VIDEO_CAPTURE)) {
      fprintf(stderr, "The device doesn't support video capture");
      close();
      return -1;
    }

    // Check if the camera supports streaming
    if (!(v4lstruct->cap.capabilities & V4L2_CAP_STREAMING)) {
      fprintf(stderr, "The device doesn't support video streaming");
      close();
      return -1;
    }

    /*
     // List and set video input, maybe useful in the future*
    v4l2_input input;
    input.index = 0;
    while (xioctl(v4lstruct->device, VIDIOC_ENUMINPUT, &input) != -1) {
      printf("%i %s\n", input.index, input.name);
      input.index++;
    }

    unsigned int index = 0;
    if (xioctl(v4lstruct->device, VIDIOC_S_INPUT, &index) == -1) {
      perror("VIDIOC_S_INPUT");
    }
    */

    v4l2_queryctrl query;
    query.id = V4L2_CID_BASE;
    
    // This is a trick that comes from VLC:
    // Sometime the driver has some problems and returns an EIO.
    // We retry to query the device until it works
    int retry_count = 10;
    while (--retry_count &&
           xioctl(v4lstruct->device, VIDIOC_QUERYCTRL, &query) == -1 &&
          (errno == EIO || errno == EPIPE || errno == ETIMEDOUT)) {
      query.id = V4L2_CID_BASE;
    }

    /*
    // List controls and set default value, maybe useful in the future
    while (xioctl(v4lstruct->device, VIDIOC_QUERYCTRL, &query) != -1) {
      printf("%i %s\n", query.id, query.name);
      bool set_default = false;
      switch (query.type) {
        case V4L2_CTRL_TYPE_INTEGER:
          printf("  int [%d;%d] step %d default %d", query.minimum, query.maximum, query.step, query.default_value);
          set_default = true;
          break;
        case V4L2_CTRL_TYPE_BOOLEAN:
          printf("  bool default %d", query.default_value);
          set_default = true;
          break;
        case V4L2_CTRL_TYPE_MENU:
          printf("  menu [%d;%d] default %d", query.minimum, query.maximum, query.default_value);
          set_default = true;
          break;
        case V4L2_CTRL_TYPE_BUTTON:
          printf("  bouton");
          break;
        case V4L2_CTRL_TYPE_INTEGER64 :
          printf("  int64");
          break;
        default:
          break;
      }

      printf("\n");
      
      v4l2_control ctl;
      ctl.id = query.id;
      if (xioctl(v4lstruct->device, VIDIOC_G_CTRL, &ctl) == -1) {
        perror("VIDIOC_G_CTRL");
      }

      printf("  Current value: %d\n", ctl.value);

      if (set_default) {
        ctl.value = query.default_value;
        if (xioctl(v4lstruct->device, VIDIOC_S_CTRL, &ctl) == -1) {
          perror("VIDIOC_S_CTRL");
        }
      }

      query.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
    }
    */

    // Reset cropping (see http://v4l2spec.bytesex.org/spec/c6488.htm#CAPTURE)
    v4l2_cropcap cropcap;
    v4l2_crop crop;

    memset (&cropcap, 0, sizeof (cropcap));
    cropcap.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(v4lstruct->device, VIDIOC_CROPCAP, &cropcap) == -1) {
      perror ("VIDIOC_CROPCAP");
      //return -1;
    }

    memset (&crop, 0, sizeof (crop));
    crop.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    crop.c = cropcap.defrect;

    if (xioctl (v4lstruct->device, VIDIOC_S_CROP, &crop) == -1) {
      if (errno != EINVAL) {
        perror ("VIDIOC_S_CROP");
        //return -1;
      }
    }

    // Get current device format
    v4l2_format fmt;
    memset(&fmt, 0, sizeof(v4l2_format));
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

    if (xioctl(v4lstruct->device, VIDIOC_G_FMT, &fmt) == -1) {
      perror("The device doesn't support V4L2_BUF_TYPE_VIDEO_CAPTURE");
      close();
      return -1;
    }
    
    v4lstruct->format = fmt.fmt.pix;


    // Get current streaming parameters
    v4l2_streamparm param;
    memset(&param, 0, sizeof(v4l2_streamparm));
    param.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (xioctl(v4lstruct->device, VIDIOC_G_PARM, &param) == -1) {
      perror("The device doesn't support VIDIOC_G_PARAM");
      close();
      return -1;
    }

    v4lstruct->param = param.parm.capture;

  }

  return 0;
}

/**
 * Convert v4l2 pixel format to Camera::PixelFormat
 */
Camera::PixelFormat convertPixelFormat(unsigned int v4l2_pixelFormat) {
  switch (v4l2_pixelFormat) {
    case V4L2_PIX_FMT_YUYV:
      return Camera::YUYV;
    case V4L2_PIX_FMT_MJPEG:
      return Camera::MJPEG;
    case V4L2_PIX_FMT_RGB24:
      return Camera::RGB24;
    default:
      return Camera::OTHER;
  }
}

/**
 * Convert Camera::PixelFormat to v4l2 pixel format
 */
unsigned int convertPixelFormat(Camera::PixelFormat pixelFormat) {
  switch (pixelFormat) {
    case Camera::YUYV:
      return V4L2_PIX_FMT_YUYV;
    case Camera::MJPEG:
      return V4L2_PIX_FMT_MJPEG;
    case Camera::RGB24:
      return V4L2_PIX_FMT_RGB24;
    case Camera::OTHER:
      assert(false);
      return 0;
    default:
      assert(false);
      return 0;
  }
}

int V4LCamera::getSupportedPixelFormats(std::vector<PixelFormat>& pixelFormats) {
  if (v4lstruct->opened) {
    pixelFormats.clear();
    
    v4l2_fmtdesc fmtdesc;
    memset(&fmtdesc, 0, sizeof(v4l2_fmtdesc));
    fmtdesc.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmtdesc.index = 0;

    while (xioctl(v4lstruct->device, VIDIOC_ENUM_FMT, &fmtdesc) != -1) {
      pixelFormats.push_back(convertPixelFormat(fmtdesc.pixelformat));
      fmtdesc.index++;
    }
  }

  return 0;
}

int V4LCamera::getSupportedFrameSizes(PixelFormat pixelFormat, std::vector<FrameSize>& frameSizes) {
  if (v4lstruct->opened) {
    frameSizes.clear();
    
    v4l2_frmsizeenum frmsizeenum;
    memset(&frmsizeenum, 0, sizeof(v4l2_frmsizeenum));
    frmsizeenum.pixel_format = convertPixelFormat(pixelFormat);
    frmsizeenum.index = 0;
    
    while (xioctl(v4lstruct->device, VIDIOC_ENUM_FRAMESIZES, &frmsizeenum) != -1) {
      if (frmsizeenum.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
        fprintf(stderr, "Unsupported type of framesizes enumeration\n");
        return -1;
      }
      
      frameSizes.push_back(FrameSize(frmsizeenum.discrete.width, frmsizeenum.discrete.height));

      frmsizeenum.index++;
    }
  }

  return 0;
}

int V4LCamera::getSupportedFrameIntervals(PixelFormat pixelFormat, FrameSize& frameSize,
                                         std::vector<FrameInterval>& frameIntervals) {
  if (v4lstruct->opened) {
    v4l2_frmivalenum frmivalenum;
    memset(&frmivalenum, 0, sizeof(v4l2_frmivalenum));
    frmivalenum.pixel_format = convertPixelFormat(pixelFormat);
    frmivalenum.height = frameSize.height;
    frmivalenum.width = frameSize.width;
    frmivalenum.index = 0;
    while (xioctl(v4lstruct->device, VIDIOC_ENUM_FRAMEINTERVALS, &frmivalenum) != -1) {
      if (frmivalenum.type != V4L2_FRMSIZE_TYPE_DISCRETE) {
        fprintf(stderr, "Unsupported type of frameinterval enumeration\n");
        //return -1;
        break;
      }

      frameIntervals.push_back(FrameInterval(frmivalenum.discrete.numerator, frmivalenum.discrete.denominator));

      frmivalenum.index++;
    }
  }

  return 0;
}

Camera::PixelFormat V4LCamera::getPixelFormat() const {
  if (v4lstruct->opened) {
    return convertPixelFormat(v4lstruct->format.pixelformat);
  }
  else {
    return Camera::OTHER;
  }
}

void V4LCamera::setPixelFormat(PixelFormat pixelFormat) {
  if (v4lstruct->opened) {
    v4l2_format format;
    memset(&format, 0, sizeof(v4l2_format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix = v4lstruct->format;

    format.fmt.pix.pixelformat = convertPixelFormat(pixelFormat);

    if (xioctl(v4lstruct->device, VIDIOC_S_FMT, &format) == -1) {
      perror("Error setting parameters (VIDIOC_S_FMT)");
      return;
    }

    v4lstruct->format = format.fmt.pix;
  }
}

Camera::FrameSize V4LCamera::getFrameSize() const {
  if (v4lstruct->opened) {
    return FrameSize(v4lstruct->format.width, v4lstruct->format.height);
  }
  else {
   return FrameSize(0, 0);
  }
}

void V4LCamera::setFrameSize(FrameSize& frameSize) {
  if (v4lstruct->opened) {
    v4l2_format format;
    memset(&format, 0, sizeof(v4l2_format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix = v4lstruct->format;

    format.fmt.pix.width = frameSize.width;
    format.fmt.pix.height = frameSize.height;

    if (xioctl(v4lstruct->device, VIDIOC_S_FMT, &format) == -1) {
      perror("Error setting parameters (VIDIOC_S_FMT)");
      return;
    }

    v4lstruct->format = format.fmt.pix;
  }
}

Camera::FrameInterval V4LCamera::getFrameInterval() const {
  if (v4lstruct->opened) {
    return FrameInterval(v4lstruct->param.timeperframe.numerator, v4lstruct->param.timeperframe.denominator);
  }
  else {
    return FrameInterval(0, 0);
  }
}

void V4LCamera::setFrameInterval(FrameInterval& frameInterval) {
  if (v4lstruct->opened) {
    v4l2_streamparm param;
    memset(&param, 0, sizeof(v4l2_streamparm));
    param.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    param.parm.capture = v4lstruct->param;
    param.parm.capture.timeperframe.numerator = frameInterval.numerator;
    param.parm.capture.timeperframe.denominator = frameInterval.denominator;

    if (xioctl(v4lstruct->device, VIDIOC_S_PARM, &param) == -1) {
      perror("The device doesn't support VIDIOC_S_PARAM");
      return;
    }

    v4lstruct->param = param.parm.capture;
  }
}

void V4LCamera::printSummary() {
  if (v4lstruct->opened) {
    printf("Device name: %s\n", v4lstruct->device_name);
    printf("Bus info: %s\n", v4lstruct->cap.bus_info);
    printf("Card: %s\n", v4lstruct->cap.card);
    printf("Driver: %s\n", v4lstruct->cap.driver);
    printf("Version: %d\n", v4lstruct->cap.version);

    printf("Pixel format: %d\n", v4lstruct->format.pixelformat);
    printf("  %dx%d @ %d/%d\n", v4lstruct->format.width, v4lstruct->format.height,
          v4lstruct->param.timeperframe.numerator, v4lstruct->param.timeperframe.denominator);
  }
  else {
    printf("Device name: %s\n", v4lstruct->device_name);
    printf("Device not open\n");
  }
}

static void* captureLoop_(void* param) {
  V4LCamera* me = (V4LCamera*)param;
  me->captureLoop();

  return NULL;
}

int V4LCamera::start() {
  if (!v4lstruct->opened) {
    return -1;
  }

  mustStop = false;
  
  v4lstruct->reqbuf.count = NB_REQUESTED_BUFFER;
  v4lstruct->reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  v4lstruct->reqbuf.memory = V4L2_MEMORY_MMAP;

  // Request the buffers
  if (xioctl(v4lstruct->device, VIDIOC_REQBUFS, &v4lstruct->reqbuf) == -1) {
    perror("The device doesn't support MMap streaming");
    return -1;
  }
  
  if (v4lstruct->reqbuf.count < MIN_NB_BUFFER) {
    perror("Insufficient buffer memory on device");
    return -1;
  }

  if (v4lstruct->buffers != NULL) {
    delete v4lstruct->buffers;
  }

  // Allocate the buffers
  v4lstruct->buffers = new Buffer[v4lstruct->reqbuf.count];
  assert(v4lstruct->buffers != NULL);

  // Initialize the buffers with NULL (useful to release memory in case of
  // problems
  for(unsigned int i = 0; i < v4lstruct->reqbuf.count; i++) {
    v4lstruct->buffers[i].start = NULL;
  }

  // Get the buffers
  for(unsigned int i = 0; i < v4lstruct->reqbuf.count; i++) {
    v4l2_buffer buffer = {0};
    buffer.type = v4lstruct->reqbuf.type;
    buffer.memory = v4lstruct->reqbuf.memory;
    buffer.index = i;

    if (xioctl(v4lstruct->device, VIDIOC_QUERYBUF, &buffer) == -1) {
      perror("Can't query buffer");
      return -1;
    }

    v4lstruct->buffers[i].length = buffer.length;
    v4lstruct->buffers[i].start = (unsigned char*) mmap(NULL, buffer.length,
                                  PROT_READ | PROT_WRITE, MAP_SHARED,
                                  v4lstruct->device, buffer.m.offset);
    
    if (v4lstruct->buffers[i].start == MAP_FAILED) {
      fprintf(stderr, "mmap error\n");
      
      for (unsigned int j = 0; j < i; j++) {
        munmap(v4lstruct->buffers[j].start, v4lstruct->buffers[j].length);
        v4lstruct->buffers[j].start = NULL;
      }
      
      return -1;
    }
  }

  // Set all buffers as enqueued state
  for (unsigned int i = 0; i < v4lstruct->reqbuf.count; i++) {
    v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;
    buf.index = i;
    
    if (xioctl(v4lstruct->device, VIDIOC_QBUF, &buf) == -1) {
      perror("Can't query buffer");
      return -1;
    }
  }


  // Start streaming
  if (xioctl(v4lstruct->device, VIDIOC_STREAMON, &v4lstruct->reqbuf.type) == -1) {
    perror("Can't start streaming");
    return -1;
  }

  // Start the thread
  int error = pthread_create(&v4lstruct->thread, NULL, captureLoop_, (void*) this);

  if (error != 0) {
    v4lstruct->thread = 0;
    return -1;
  }

  return 0;
}


void V4LCamera::captureLoop() {
  const PixelFormat pixelformat = getPixelFormat();
  
  while(!mustStop) {
    // Select manpage say that we can't trust the value of timeout after
    // a select call, so we can't define this before the loop.
    // Moreover if an error occures with select, we can't trust the set either
    
    // Set of file descriptor
    fd_set fdset;
    // Initialize the set
    FD_ZERO(&fdset);
    // Add our fd to the set
    FD_SET(v4lstruct->device, &fdset);

    // Highest-numbered fd in the set, plus 1
    int nfds = v4lstruct->device + 1;
    
    // Timeout of 1 seconds
    timeval timeout;
    timeout.tv_sec = 1;
    timeout.tv_usec = 0;

    // Wait for a notification
    int ret = select(nfds, &fdset, NULL, NULL, &timeout);

    if (ret == 0) {
      // Timeout expired
      fprintf(stderr, "select timout: %d seconds without frame\n", 1);
      continue;
    }
    else if (ret == -1) {
      if (errno == EINTR) {
        // Interrupted system call
        // We ignore the signal
        continue;
      }
      else {
        // Some error occurs
        perror("Select error");
        continue;
      }
    }
    else {
      // Successful select
      v4l2_buffer buffer;
      memset(&buffer, 0, sizeof(v4l2_buffer));
      buffer.type = v4lstruct->reqbuf.type;
      buffer.memory = v4lstruct->reqbuf.memory;
      
      // Dequeue the buffer
      if (xioctl(v4lstruct->device, VIDIOC_DQBUF, &buffer) == -1) {
        perror("Can't dequeue buffer");
        continue;
      }

      assert(buffer.index < v4lstruct->reqbuf.count);

      double timestamp = buffer.timestamp.tv_sec + buffer.timestamp.tv_usec / 1.e6;

      pthread_mutex_lock(&callbacks_mutex);
      for(std::vector<CameraCallback*>::iterator it = callbacks.begin(); it != callbacks.end(); it++) {
        (*it)->imageReceived(v4lstruct->buffers[buffer.index].start, pixelformat,
                             v4lstruct->format.width, v4lstruct->format.height,
                             v4lstruct->format.bytesperline,
                             v4lstruct->format.sizeimage, buffer.sequence,
                             timestamp);
      }
      pthread_mutex_unlock(&callbacks_mutex);
      

      // Re-enqueue the buffer
      if (xioctl(v4lstruct->device, VIDIOC_QBUF, &buffer) == -1) {
        perror("Can't enqueue buffer");
        continue;
      }
    }
  }


  // Stop streaming
  if (xioctl(v4lstruct->device, VIDIOC_STREAMOFF, &v4lstruct->reqbuf.type) == -1) {
    perror("Can't stop streaming");
    //return -1;
  }

  // Release buffers
  for (unsigned int j = 0; j < v4lstruct->reqbuf.count; j++) {
    munmap(v4lstruct->buffers[j].start, v4lstruct->buffers[j].length);
    v4lstruct->buffers[j].start = NULL;
  }

  delete[] v4lstruct->buffers;
  v4lstruct->buffers = NULL;
}

void V4LCamera::wait() {
  if (v4lstruct->thread != 0) {
    pthread_join(v4lstruct->thread, NULL);
  }
}

void V4LCamera::stop() {
  mustStop = true;
  wait();
}

static void close_(int fd) {
  close(fd);
}

void V4LCamera::close() {
  if(v4lstruct->opened) {
    stop();
    close_(v4lstruct->device);
  }
}

}}