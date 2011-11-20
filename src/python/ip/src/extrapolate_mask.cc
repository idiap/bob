/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @Sun 20 Nov 17:39:02 2011 CET
 *
 * @brief Binds the extrapolateMask operation into python 
 */

#include "core/python/ndarray.h"
#include "ip/extrapolateMask.h"

using namespace boost::python;
namespace tp = Torch::python;
namespace ip = Torch::ip;
namespace ca = Torch::core::array;

template <typename T>
static void inner_extrapolateMask(tp::const_ndarray src, tp::ndarray img) {
  blitz::Array<T,2> img_ = img.bz<T,2>();
  ip::extrapolateMask<T>(src.bz<bool,2>(), img_);
}

static void extrapolateMask (tp::const_ndarray src, tp::const_ndarray img) {
  
  const ca::typeinfo& info = img.type();
  
  if (info.nd != 2) PYTHON_ERROR(TypeError, "mask extrapolation does not support input of type '%s'", info.str().c_str());

  switch (info.dtype) {
    case ca::t_uint8: return inner_extrapolateMask<uint8_t>(src, img);
    case ca::t_uint16: return inner_extrapolateMask<uint16_t>(src, img);
    case ca::t_float64: return inner_extrapolateMask<double>(src, img);
    default: PYTHON_ERROR(TypeError, "mask extrapolation does not support type '%s'", info.str().c_str());
  }

}

void bind_ip_extrapolate_mask() {
  def("extrapolateMask", &extrapolateMask, (arg("src_mask"), arg("img")), "Extrapolate a 2D array/image, taking mask into account.");
}
