#include "daq/BobOutputWriter.h"
#include <io/VideoWriter.h>
#include <core/blitz_array.h>

namespace bob { namespace daq {
  
BobOutputWriter::BobOutputWriter() {
  videoWriter = NULL;
  textFile = NULL;
}

BobOutputWriter::~BobOutputWriter() {
  delete videoWriter;
  delete textFile;
}

void BobOutputWriter::close() {
  if (videoWriter != NULL) {
    videoWriter->close();
    textFile->close();
    
    delete videoWriter;
    delete textFile;

    videoWriter = NULL;
    textFile = NULL;
  }
}

void BobOutputWriter::open(int width, int height, int fps) {
  if (videoWriter != NULL) {
    close();
  }
  
  videoWriter = new io::VideoWriter(dir + "/" + name + ".avi", height, width, fps);
  textFile = new std::ofstream((dir + "/" + name + ".txt").c_str(), std::ios::out);
}

void BobOutputWriter::writeFrame(blitz::Array<unsigned char, 2>& image, int frameNb, double timestamp) {
  if (videoWriter != NULL) {
    // Convert image in Bob format (3D array)
    blitz::Array<unsigned char, 3> image3(image.data(), blitz::shape(image.rows(), image.cols() / 3, 3), blitz::neverDeleteData);

    core::array::blitz_array array(image3.transpose(2, 0, 1));
    videoWriter->append(array);
    *textFile << frameNb << " " << timestamp << std::endl;
  }
}


}}