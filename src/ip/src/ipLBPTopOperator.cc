/**
 * @file ipLBPTopOperator.cc
 * @author <a href="mailto:andre.anjos@idiap.ch">Andre Anjos</a> 
 *
 * @brief  
 */

#include "ip/ipLBPTopOperator.h"
#include "ip/ipLBP.h"
#include "ip/ipLBP4R.h"
#include "ip/ipLBP8R.h"
#include "core/general.h"

Torch::ipLBPTopOperator::ipLBPTopOperator(int radius_xy, 
                                          int points_xy, 
                                          int radius_xt, 
                                          int points_xt, 
                                          int radius_yt,
                                          int points_yt)
: m_radius_xy(0),
  m_points_xy(0),
  m_radius_xt(0),
  m_points_xt(0),
  m_radius_yt(0),
  m_points_yt(0),
  m_lbp_xy(0),
  m_lbp_xt(0),
  m_lbp_yt(0)
{
  addIOption("radius_xy", radius_xy);
  addIOption("points_xy", points_xy);
  addIOption("radius_xt", radius_xt);
  addIOption("points_xt", points_xt);
  addIOption("radius_yt", radius_yt);
  addIOption("points_yt", points_yt);
  optionChanged("");
}

/**
 * A little helper to create the LBP operators in an homogene way.
 */
static Torch::ipLBP* make_lbp(int radius, int points)
{
  Torch::ipLBP* retval = 0;
  if (points != 4 && points != 8) {
    Torch::error("Cannot create %d-point LBP operator (use 4 or 8 only)!",
        points);
  }
  else {
    if (points == 4) retval = new Torch::ipLBP4R(radius);
    else retval = new Torch::ipLBP8R(radius);
    retval->setBOption("ToAverage", false);
    retval->setBOption("AddAvgBit", false);
    retval->setBOption("Uniform", true);
    retval->setBOption("RotInvariant", true);
  }
  return retval;
}

void Torch::ipLBPTopOperator::optionChanged(const char* /* name */)
{
  m_radius_xy = getIOption("radius_xy");
  m_points_xy = getIOption("points_xy");
  m_radius_xt = getIOption("radius_xt");
  m_points_xt = getIOption("points_xt");
  m_radius_yt = getIOption("radius_yt");
  m_points_yt = getIOption("points_yt");
  m_lbp_xy = make_lbp(m_radius_xy, m_points_xy);
  m_lbp_xt = make_lbp(m_radius_xt, m_points_xt);
  m_lbp_xy = make_lbp(m_radius_yt, m_points_yt);
}

Torch::ipLBPTopOperator::~ipLBPTopOperator() {
  delete m_lbp_xy;
  m_lbp_xy = 0;
  delete m_lbp_xt;
  m_lbp_xt = 0;
  delete m_lbp_yt;
  m_lbp_yt = 0;
}

bool Torch::ipLBPTopOperator::process(const Torch::ShortTensor& tensor,
    Torch::Image& xy, Torch::Image& xt, Torch::Image& yt) const
{
  // we need an odd number, at (2N+1), where N = max(radius_xt, radius_yt)
  if (tensor.size(4)%2 == 0) {
    Torch::warning("Cannot process a even-numbered set of frames");
    return false;
  }
  const int N = max(m_radius_xt, m_radius_yt);
  if (tensor.size(4) != (2*N+1) ) {
    Torch::warning("The number of input frames should be %d", 2*N+1);
    return false;
  }

  // only grayscale images accepted
  if (tensor.size(3) != 1) {
    Torch::warning("Cannot work with colored images, please use grayscale");
    return false;
  }

  // XY plane calculation
  Torch::ShortTensor k;
  k.select(&tensor, 3, 2*N);
  int width = k.size(1);
  int height = k.size(0);
	const int max_lbp_xy = m_lbp_xy->getMaxLabel();
	const float inv_max_lbp_xy = 255.0f / (max_lbp_xy + 0.0f);
  for (int x=m_radius_xy; x < (width-m_radius_xy); ++x) {
    for (int y=m_radius_xy; y < (height-m_radius_xy); ++y) {
      m_lbp_xy->setXY(x, y);
      m_lbp_xy->process(k);
      xy.set(y, x, 0, (short)(inv_max_lbp_xy * m_lbp_xy->getLBP() + 0.5f));
    }
  }

  // XT plane calculation
	const int max_lbp_xt = m_lbp_xt->getMaxLabel();
	const float inv_max_lbp_xt = 255.0f / (max_lbp_xt + 0.0f);
  for (int y=m_radius_xt; y < (tensor.size(0)-m_radius_xt); ++y) {
    Torch::ShortTensor k;
    k.select(&tensor, 0, y);
    Torch::ShortTensor kt;
    kt.transpose(&k, 1, 2); //get the gray levels on the last dimension
    for (int x = m_radius_xt; x < (width-m_radius_xt); ++x) {
      m_lbp_xt->setXY(x, 2*N);
      m_lbp_xt->process(kt);
      xt.set(y, x, 0, (short)(inv_max_lbp_xt * m_lbp_xt->getLBP() + 0.5f));
    }
  }

  // YT plane calculation
	const int max_lbp_yt = m_lbp_yt->getMaxLabel();
	const float inv_max_lbp_yt = 255.0f / (max_lbp_yt + 0.0f);
  for (int x=m_radius_yt; x < (tensor.size(1)-m_radius_yt); ++x) {
    Torch::ShortTensor k;
    k.select(&tensor, 1, x);
    Torch::ShortTensor kt;
    kt.transpose(&k, 1, 2); //get the gray levels on the last dimension
    for (int y = m_radius_yt; y < (height-m_radius_yt); ++y) {
      m_lbp_yt->setXY(y, 2*N);
      m_lbp_yt->process(kt);
      yt.set(y, x, 0, (short)(inv_max_lbp_yt * m_lbp_yt->getLBP() + 0.5f));
    }
  }

  return true;
}
