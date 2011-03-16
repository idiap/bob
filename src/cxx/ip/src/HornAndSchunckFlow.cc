/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Tue  8 Mar 08:20:02 2011 
 *
 * @brief Defines the HornAndSchunckFlow methods
 */

#include "ip/HornAndSchunckFlow.h"

namespace ip = Torch::ip;

ip::HornAndSchunckFlow::HornAndSchunckFlow(float alpha, size_t iterations) :
  m_alpha(alpha),
  m_iterations(iterations),
  m_ex(),
  m_ey(),
  m_et(),
  m_u0(),
  m_v0(),
  m_common_term()
{
}

ip::HornAndSchunckFlow::HornAndSchunckFlow(const ip::HornAndSchunckFlow& other)
  :
  m_alpha(other.m_alpha),
  m_iterations(other.m_iterations),
  m_ex(other.m_ex.copy()),
  m_ey(other.m_ey.copy()),
  m_et(other.m_et.copy()),
  m_u0(other.m_u0.copy()),
  m_v0(other.m_v0.copy()),
  m_common_term(other.m_common_term.copy())
{
}

ip::HornAndSchunckFlow::~HornAndSchunckFlow() { }

/**
 * Compares two arrays for their shape
 */
template<typename T1, typename T2>
static bool shapeq(const blitz::Array<T1,2>& a1, const blitz::Array<T2,2>& a2) {
  return (a1.rows() == a2.rows()) && (a1.columns() == a2.columns());
}

ip::HornAndSchunckFlow& ip::HornAndSchunckFlow::operator=
(const ip::HornAndSchunckFlow& other)
{
  m_alpha = other.m_alpha;
  m_iterations = other.m_iterations;
  if (!shapeq(m_ex, other.m_ex)) m_ex.resize(other.m_ex.shape());
  m_ex = other.m_ex; //copy
  if (!shapeq(m_ey, other.m_ey)) m_ey.resize(other.m_ey.shape());
  m_ey = other.m_ey; //copy
  if (!shapeq(m_et, other.m_et)) m_et.resize(other.m_et.shape());
  m_et = other.m_et; //copy
  if (!shapeq(m_u0, other.m_u0)) m_u0.resize(other.m_u0.shape());
  m_u0 = other.m_u0; //copy
  if (!shapeq(m_v0, other.m_v0)) m_v0.resize(other.m_v0.shape());
  m_v0 = other.m_v0; //copy
  if (!shapeq(m_common_term, other.m_common_term)) 
    m_common_term.resize(other.m_common_term.shape());
  m_common_term = other.m_common_term; //copy
  return *this;
}

/**
 * Estimates the partial derivative in the x direction. The formula is:
 *
 * Ex = 1/4 { E(i,j+1,k) - E(i,j,k) + 
 *            E(i+1,j+1,k) - E(i+1,j,k) +
 *            E(i,j+1,k+1) - E(i,j,k+1) +
 *            E(i+1,j+1,k+1) - E(i+1,j,k+1) }
 *
 * Please note we have separate images for t = k (i1) and t = k+1 (i2). So we
 * can re-write like this:
 * 
 * Ex = 1/4 { i1(i,j+1) - i1(i,j) +
 *            i1(i+1,j+1) - i1(i+1,j) +
 *            i2(i,j+1) - i2(i,j) +
 *            i2(i+1,j+1) - i2(i+1,j) }
 *
 * Please note that this formula will not be bound for the last column and the
 * last row of the image, in which case we will just set it to zero.
 */
static void Ex(const blitz::Array<uint8_t,2>& i1, 
    const blitz::Array<uint8_t,2>& i2, blitz::Array<double,2>& result) {
  blitz::Range i(0,i1.extent(0)-1), j(0,i1.extent(1)-1);
  result = 0;
  result = 0.25 * ( i1(i,j+1) - i1(i,j) +
                    i1(i+1,j+1) - i1(i+1,j) +
                    i2(i,j+1) - i2(i,j) +
                    i2(i+1,j+1) - i2(i+1,j) );
}

/**
 * Estimates the partial derivative in the y direction. The formula is:
 *
 * Ey = 1/4 { E(i+1,j,k) - E(i,j,k) + 
 *            E(i+1,j+1,k) - E(i,j+1,k) +
 *            E(i+1,j,k+1) - E(i,j,k+1) +
 *            E(i+1,j+1,k+1) - E(i,j+1,k+1) }
 *
 * Please note we have separate images for t = k (i1) and t = k+1 (i2). So we
 * can re-write like this:
 * 
 * Ey = 1/4 { i1(i+1,j) - i1(i,j) +
 *            i1(i+1,j+1) - i1(i,j+1) +
 *            i2(i+1,j) - i2(i,j) }
 *            i2(i+1,j+1) - i2(i,j+1) }
 *
 * Please note that this formula will not be bound for the last column and the
 * last row of the image, in which case we will just set it to zero.
 */
static void Ey(const blitz::Array<uint8_t,2>& i1, 
    const blitz::Array<uint8_t,2>& i2, blitz::Array<double,2>& result) {
  blitz::Range i(0,i1.extent(0)-1), j(0,i1.extent(1)-1);
  result = 0;
  result = 0.25 * ( i1(i+1,j) - i1(i,j) +
                    i1(i+1,j+1) - i1(i,j+1) +
                    i2(i+1,j) - i2(i,j) +
                    i2(i+1,j+1) - i2(i,j+1) );
}

/**
 * Estimates the partial derivative in the t (time) direction. The formula is:
 *
 * Et = 1/4 { E(i,j,k+1) - E(i,j,k) + 
 *            E(i+1,j,k+1) - E(i+1,j,k) +
 *            E(i,j+1,k+1) - E(i,j+1,k) +
 *            E(i+1,j+1,k+1) - E(i+1,j+1,k) }
 *
 * Please note we have separate images for t = k (i1) and t = k+1 (i2). So we
 * can re-write like this:
 * 
 * Et = 1/4 { i2(i,j) - i1(i,j) +
 *            i2(i+1,j) - i1(i+1,j) +
 *            i2(i,j+1) - i1(i,j+1) }
 *            i2(i+1,j+1) - i1(i+1,j+1) }
 *
 * Please note that this formula will not be bound for the last column and the
 * last row of the image, in which case we will just set it to zero.
 */
static void Et(const blitz::Array<uint8_t,2>& i1, 
    const blitz::Array<uint8_t,2>& i2, blitz::Array<double,2>& result) {
  blitz::Range i(0,i1.extent(0)-1), j(0,i1.extent(1)-1);
  result = 0;
  result = 0.25 * ( i2(i,j) - i1(i,j) +
                    i2(i+1,j) - i1(i+1,j) +
                    i2(i,j+1) - i1(i,j+1) +
                    i2(i+1,j+1) - i1(i+1,j+1) );
}

/**
 * Estimates the U(n) component in the iterative formula. Its formulation is
 * given like this:
 *
 * U(n) = 1/6  { u(i-1,j,k) + u(i,j+1,k) + u(i+1,j,k) + u(i,j-1,k) } +
 *        1/12 { u(i-1,j-1,k) + u(i-1,j+1,k) + u(i+1,j+1,k) + u(i+1,j-1,k) }
 *
 * This quantity is only calculated for the current image being analyzed. In
 * the case of our nomenclature, we are talking about i1 (t = k).
 *
 * Please note that the above formula only exists in a subrange of the input
 * velocities (namely 1:extent-1), because it has a 1 pixel window.
 */
static void U(const blitz::Array<double,2>& u, blitz::Array<double,2>& result) {
  blitz::Range i(1,u.extent(0)-1), j(1,u.extent(1)-1);
  result = 0;
  result = 1.0/6 *  ( u(i-1,j) + u(i,j+1) + u(i+1,j) + u(i,j-1) ) +
           1.0/12 * ( u(i-1,j-1) + u(i-1,j+1) + u(i+1,j+1), u(i+1,j-1) );
}

/**
 * Estimates the V(n) component in the iterative formula. Its formulation is
 * given like this:
 *
 * V(n) = 1/6  { v(i-1,j,k) + v(i,j+1,k) + v(i+1,j,k) + v(i,j-1,k) } +
 *        1/12 { v(i-1,j-1,k) + v(i-1,j+1,k) + v(i+1,j+1,k) + v(i+1,j-1,k) }
 *
 * This quantity is only calculated for the current image being analyzed. In
 * the case of our nomenclature, we are talking about i1 (t = k).
 *
 * Please note that the above formula only exists in a subrange of the input
 * velocities (namely 1:extent-1), because it has a 1 pixel window.
 */
static void V(const blitz::Array<double,2>& v, blitz::Array<double,2>& result) {
  blitz::Range i(1,v.extent(0)-1), j(1,v.extent(1)-1);
  result = 0;
  result = 1.0/6 *  ( v(i-1,j) + v(i,j+1) + v(i+1,j) + v(i,j-1) ) +
           1.0/12 * ( v(i-1,j-1) + v(i-1,j+1) + v(i+1,j+1), v(i+1,j-1) );
}

void ip::HornAndSchunckFlow::operator() (
    const blitz::Array<uint8_t,2>& i1, const blitz::Array<uint8_t,2>& i2,
    blitz::Array<double,2>& u, blitz::Array<double,2>& v) {
  
  if (!shapeq(m_ex, i1)) { //resize internal cache as needed
    m_ex.resize(i1.shape());
    m_ey.resize(i1.shape());
    m_et.resize(i1.shape());
    m_u0.resize(i1.shape());
    m_v0.resize(i1.shape());
    m_common_term.resize(i1.shape());
  }

  if (!shapeq(u, i1)) { //reset output
    u.resize(i1.shape()); u = 0;
    v.resize(i1.shape()); v = 0;
  }

  //iterative flow calculation proposed by Horn & Schunck
  Ex(i1, i2, m_ex);
  Ey(i1, i2, m_ey);
  Et(i1, i2, m_et);
  double a2 = std::pow(m_alpha, 2);
  for (size_t i=0; i<m_iterations; ++i) {
    U(u, m_u0);
    V(v, m_v0);
    m_common_term = (m_ex*m_u0 + m_ey*m_v0 + m_et)/(a2 + blitz::pow2(m_ex) + blitz::pow2(m_ey));
    u = m_u0 - m_ex*m_common_term;
    v = m_v0 - m_ey*m_common_term;
  }
}
