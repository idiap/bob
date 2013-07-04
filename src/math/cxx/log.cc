/**
 * @file math/cxx/log.cc
 * @date Fri Feb 10 20:02:07 2012 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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

#include <stdexcept>
#include <boost/format.hpp>
#include <bob/math/log.h>
#include <bob/core/logging.h>

/**
 * Computes log(a+b)=log(exp(log(a))+exp(log(b))) from log(a) and log(b),
 * while dealing with numerical issues
 */
double bob::math::Log::logAdd(double log_a, double log_b)
{
  if(log_a < log_b)
  {
    double tmp = log_a;
    log_a = log_b;
    log_b = tmp;
  }

  double minusdif = log_b - log_a;
  //#ifdef DEBUG
  if(std::isnan(minusdif))
  {
    boost::format m("logadd: minusdif (%f) log_b (%f) or log_a (%f) is nan");
    m % minusdif % log_b % log_a;
    throw std::runtime_error(m.str());
  }
  //#endif
  if(minusdif < MINUS_LOG_THRESHOLD) return log_a;
  else return log_a + log1p(exp(minusdif));
}

/**
 * Computes log(a-b)=log(exp(log(a))-exp(log(b))) from log(a) and log(b),
 * while dealing with numerical issues
 */
double bob::math::Log::logSub(double log_a, double log_b)
{
  double minusdif;

  if(log_a < log_b)
  {
    boost::format m("logsub: log_a (%f) should be greater than log_b(%f)");
    m % log_a % log_b;
    throw std::runtime_error(m.str());
  }

  minusdif = log_b - log_a;
  //#ifdef DEBUG
  if(std::isnan(minusdif))
  {
    boost::format m("logsub: minusdif (%f) log_b (%f) or log_a (%f) is nan");
    m % minusdif % log_b % log_a;
    throw std::runtime_error(m.str());
  }
  //#endif
  if(log_a == log_b) return bob::math::Log::LogZero;
  else if(minusdif < MINUS_LOG_THRESHOLD) return log_a;
  else return log_a + log1p(-exp(minusdif));
}

