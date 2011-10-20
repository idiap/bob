/**
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Mon 29 Sep 2011
 *
 * @brief Implements a WienerMachine
 */

#include "core/array_copy.h"
#include "core/cast.h"
#include "io/Arrayset.h"
#include "machine/WienerMachine.h"
#include "machine/Exception.h"
#include <complex>

namespace mach = Torch::machine;

mach::WienerMachine::WienerMachine(const blitz::Array<double,2>& Ps, const double Pn,
    const double variance_threshold):
  m_Ps(Torch::core::array::ccopy(Ps)),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_fft(new Torch::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_ifft(new Torch::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)),
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
  m_W.resize(m_Ps.shape());
  computeW();
}

mach::WienerMachine::WienerMachine():
  m_Ps(0,0),
  m_variance_threshold(1e-8),
  m_Pn(0),
  m_W(0,0),
  m_fft(boost::shared_ptr<Torch::sp::FFT2D>()),
  m_ifft(boost::shared_ptr<Torch::sp::IFFT2D>()),
  m_buffer1(0,0), m_buffer2(0,0)
{
}

mach::WienerMachine::WienerMachine(size_t height, size_t width, const double Pn,
    const double variance_threshold):
  m_Ps(height,width),
  m_variance_threshold(variance_threshold),
  m_Pn(Pn),
  m_W(height,width),
  m_fft(new Torch::sp::FFT2D(height,width)),
  m_ifft(new Torch::sp::IFFT2D(height,width)),
  m_buffer1(0,0), m_buffer2(0,0)
{
  m_Ps = 0.;
  m_W = 0.;
}

mach::WienerMachine::WienerMachine(const mach::WienerMachine& other):
  m_Ps(Torch::core::array::ccopy(other.m_Ps)),
  m_variance_threshold(other.m_variance_threshold),
  m_Pn(other.m_Pn),
  m_W(Torch::core::array::ccopy(other.m_W)),
  m_fft(new Torch::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_ifft(new Torch::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1))),
  m_buffer1(m_Ps.extent(0),m_Ps.extent(1)), 
  m_buffer2(m_Ps.extent(0),m_Ps.extent(1))
{
}

mach::WienerMachine::WienerMachine (Torch::io::HDF5File& config) {
  load(config);
}

mach::WienerMachine::~WienerMachine() {}

mach::WienerMachine& mach::WienerMachine::operator=
(const mach::WienerMachine& other) {
  m_Ps.reference(Torch::core::array::ccopy(other.m_Ps));
  m_Pn = other.m_Pn;
  m_variance_threshold = other.m_variance_threshold;
  m_W.reference(Torch::core::array::ccopy(other.m_W));
  m_fft.reset(new Torch::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_ifft.reset(new Torch::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_buffer1.resize(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer2.resize(m_Ps.extent(0),m_Ps.extent(1));
  return *this;
}

void mach::WienerMachine::load (Torch::io::HDF5File& config) {
  //reads all data directly into the member variables
  m_Ps.reference(config.readArray<double,2>("Ps"));
  config.read("Pn", m_Pn);
  config.read("variance_threshold", m_variance_threshold);
  m_W.reference(config.readArray<double,2>("W"));
  m_fft.reset(new Torch::sp::FFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_ifft.reset(new Torch::sp::IFFT2D(m_Ps.extent(0),m_Ps.extent(1)));
  m_buffer1.resize(m_Ps.extent(0),m_Ps.extent(1));
  m_buffer2.resize(m_Ps.extent(0),m_Ps.extent(1));
}

void mach::WienerMachine::resize (size_t height, size_t width) {
  m_Ps.resizeAndPreserve(height,width);
  m_W.resizeAndPreserve(height,width);
  m_fft.reset(new Torch::sp::FFT2D(height,width));
  m_ifft.reset(new Torch::sp::IFFT2D(height,width));
  m_buffer1.resizeAndPreserve(height,width);
  m_buffer2.resizeAndPreserve(height,width);
}

void mach::WienerMachine::save (Torch::io::HDF5File& config) const {
  config.setArray("Ps", m_Ps);
  config.set("Pn", m_Pn);
  config.set("variance_threshold", m_variance_threshold);
  config.setArray("W", m_W);
}

void mach::WienerMachine::computeW () {
  m_W = m_Ps;
  // Apply variance flooring threshold
  blitz::Array<bool,2> isTooSmall(m_W.shape());
  isTooSmall = m_Ps < m_variance_threshold;
  m_W += (m_variance_threshold - m_W) * isTooSmall; // W = Pn_thresholded
  // W = 1 / (1 + Pn / Ps_thresholded)
  m_W = 1. / (1. + m_Pn / m_W);
}


void mach::WienerMachine::forward_
(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const {
  m_fft->operator()(Torch::core::cast<std::complex<double> >(input), m_buffer1);
  m_buffer1 *= m_W;
  m_ifft->operator()(m_buffer1, m_buffer2);
  output = blitz::abs(m_buffer2);
}

void mach::WienerMachine::forward
(const blitz::Array<double,2>& input, blitz::Array<double,2>& output) const {
  if (m_W.extent(0) != input.extent(0)) //checks input
    throw mach::NInputsMismatch(m_W.extent(0),
        input.extent(0));
  if (m_W.extent(1) != input.extent(1)) //checks input
    throw mach::NInputsMismatch(m_W.extent(1),
        input.extent(1));
  if (m_W.extent(0) != output.extent(0)) //checks output
    throw mach::NOutputsMismatch(m_W.extent(0),
        output.extent(0));
  if (m_W.extent(1) != output.extent(1)) //checks output
    throw mach::NOutputsMismatch(m_W.extent(1),
        output.extent(1));
  forward_(input, output);
}

void mach::WienerMachine::setPs(const blitz::Array<double,2>& Ps) { 
  if (m_Ps.extent(0) != Ps.extent(0)) {
    throw mach::NInputsMismatch(m_Ps.extent(0), Ps.extent(0));
  }
  if (m_Ps.extent(1) != Ps.extent(1)) {
    throw mach::NInputsMismatch(m_Ps.extent(1), Ps.extent(0));
  }
  m_Ps = Torch::core::array::ccopy(Ps);
  computeW(); 
}
