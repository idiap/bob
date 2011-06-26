/**
 * @file src/python/ip/src/block.cc 
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a> 
 *
 * @brief Binds block decomposition into python 
 */

#include <boost/python.hpp>

#include <stdint.h>
#include "ip/block.h"

using namespace boost::python;

static const char* BLOCK2D_DOC = "Perform a block decomposition of a 2D blitz array/image.";
static const char* GETBLOCKSHAPE2D_DOC = "Return the shape of the output 2D blitz array/image, when calling block which performs a block decomposition of a 2D blitz array/image.";
static const char* GETNBLOCKS2D_DOC = "Return the number of blocks of the output 2D blitz array/image, when calling block which performs a block decomposition of a 2D blitz array/image.";


#define BLOCK_DEF(T,N) \
 def("block", (void (*)(const blitz::Array<T,2>&, blitz::Array<T,3>&, const int, const int, const int, const int))&Torch::ip::block<T>, (arg("src"), arg("dst"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), BLOCK2D_DOC); \
  def("getBlockShape", (const blitz::TinyVector<int,3> (*)(const blitz::Array<T,2>&, const int, const int, const int, const int))&Torch::ip::getBlockShape<T>, (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), GETBLOCKSHAPE2D_DOC); \
  def("getNBlocks", &Torch::ip::getNBlocks<T>, (arg("src"), arg("block_h"), arg("block_w"), arg("overlap_h"), arg("overlap_w")), GETNBLOCKS2D_DOC); 


void bind_ip_block()
{
/*
  BLOCK_DEF(bool,bool)
  BLOCK_DEF(int8_t,int8)
  BLOCK_DEF(int16_t,int16)
  BLOCK_DEF(int32_t,int32)
  BLOCK_DEF(int64_t,int64)
*/
  BLOCK_DEF(uint8_t,uint8)
  BLOCK_DEF(uint16_t,uint16)
/*
  BLOCK_DEF(uint32_t,uint32)
  BLOCK_DEF(uint64_t,uint64)
  BLOCK_DEF(float,float32)
*/
  BLOCK_DEF(double,float64)
/*
  BLOCK_DEF(std::complex<float>,complex64)
  BLOCK_DEF(std::complex<double>,complex128)
*/
}
