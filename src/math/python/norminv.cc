/**
 * @file math/python/norminv.cc
 * @date Wed Apr 13 09:20:40 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief Binds the inverse normal cumulative distribution into python
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <boost/python.hpp>

#include <bob/math/norminv.h>

using namespace boost::python;

static const char* NORMSINV_DOC = "Compute the inverse normal cumulative distribution for a probability p, given a distribution with zero mean and and unit variance.\nReference: http://home.online.no/~pjacklam/notes/invnorm/";
static const char* NORMINV_DOC = "Compute the inverse normal cumulative distribution for a probability p, given a distribution with mean mu and standard deviation sigma.\nReference: http://home.online.no/~pjacklam/notes/invnorm/";

void bind_math_norminv()
{
  def("normsinv", &bob::math::normsinv, (arg("p")), NORMSINV_DOC);
  def("norminv", &bob::math::norminv, (arg("p"), arg("mu"), arg("sigma")), NORMINV_DOC);
}

