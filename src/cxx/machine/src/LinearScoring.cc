/**
 * @file cxx/machine/src/LinearScoring.cc
 * @date Wed Jul 13 16:00:04 2011 +0200
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
#include "machine/LinearScoring.h"
#include "math/linear.h"

namespace ca = bob::core::array;


namespace bob { namespace machine {

namespace detail {

  void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                     const blitz::Array<double,1>& ubm_mean,
                     const blitz::Array<double,1>& ubm_variance,
                     const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                     const std::vector<blitz::Array<double,1> >* test_channelOffset,
                     const bool frame_length_normalisation,
                     blitz::Array<double,2>& scores) 
  {
    int C = test_stats[0]->sumPx.extent(0);
    int D = test_stats[0]->sumPx.extent(1);
    int CD = C*D;
    int Tt = test_stats.size();
    int Tm = models.size();

    // Check output size
    ca::assertSameDimensionLength(scores.extent(0), models.size());
    ca::assertSameDimensionLength(scores.extent(1), test_stats.size());

    blitz::Array<double,2> A(Tm, CD);
    blitz::Array<double,2> B(CD, Tt);

    // 1) Compute A
    for(int t=0; t<Tm; ++t) {
      blitz::Array<double, 1> tmp = A(t, blitz::Range::all());
      tmp = (models[t] - ubm_mean) / ubm_variance;
    }

    // 2) Compute B
    if(test_channelOffset == 0) {
      for(int t=0; t<Tt; ++t) 
        for(int s=0; s<CD; ++s)
          B(s, t) = test_stats[t]->sumPx(s/D, s%D) - (ubm_mean(s) * test_stats[t]->n(s/D));
    }
    else {
      ca::assertSameDimensionLength((*test_channelOffset).size(), Tt);
      
      for(int t=0; t<Tt; ++t) {
        ca::assertSameDimensionLength((*test_channelOffset)[t].extent(0), CD);
        for(int s=0; s<CD; ++s) 
          B(s, t) = test_stats[t]->sumPx(s/D, s%D) - (test_stats[t]->n(s/D) * (ubm_mean(s) + (*test_channelOffset)[t](s)));
      }
    }

    // Apply the normalisation if needed
    if(frame_length_normalisation) {
      for(int t=0; t<Tt; ++t) {
        double sum_N = test_stats[t]->T;
        blitz::Array<double, 1> v_t = B(blitz::Range::all(),t);

        if (sum_N <= std::numeric_limits<double>::epsilon() && sum_N >= -std::numeric_limits<double>::epsilon())
          v_t = 0;
        else 
          v_t /= sum_N;
      }
    }

    // 3) Compute LLR
    bob::math::prod(A, B, scores);
  } 
}


void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                   const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const std::vector<blitz::Array<double,1> >& test_channelOffset,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores)
{
  detail::linearScoring(models, ubm_mean, ubm_variance, test_stats, &test_channelOffset, frame_length_normalisation, scores);
}

void linearScoring(const std::vector<blitz::Array<double,1> >& models,
                   const blitz::Array<double,1>& ubm_mean, const blitz::Array<double,1>& ubm_variance,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores)
{
  detail::linearScoring(models, ubm_mean, ubm_variance, test_stats, 0, frame_length_normalisation, scores);
}

void linearScoring(const std::vector<boost::shared_ptr<const bob::machine::GMMMachine> >& models,
                   const bob::machine::GMMMachine& ubm,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores) 
{
  int C = test_stats[0]->sumPx.extent(0);
  int D = test_stats[0]->sumPx.extent(1);
  int CD = C*D;
  std::vector<blitz::Array<double,1> > models_b;
  // Allocate and get the mean supervector
  for(size_t i=0; i<models.size(); ++i) {
    blitz::Array<double,1> mod(CD);
    models[i]->getMeanSupervector(mod);
    models_b.push_back(mod);
  }
  const blitz::Array<double,1>& ubm_mean = ubm.getMeanSupervector();
  const blitz::Array<double,1>& ubm_variance = ubm.getVarianceSupervector();
  detail::linearScoring(models_b, ubm_mean, ubm_variance, test_stats, 0, frame_length_normalisation, scores);
}

void linearScoring(const std::vector<boost::shared_ptr<const bob::machine::GMMMachine> >& models,
                   const bob::machine::GMMMachine& ubm,
                   const std::vector<boost::shared_ptr<const bob::machine::GMMStats> >& test_stats,
                   const std::vector<blitz::Array<double,1> >& test_channelOffset,
                   const bool frame_length_normalisation,
                   blitz::Array<double, 2>& scores) 
{
  int C = test_stats[0]->sumPx.extent(0);
  int D = test_stats[0]->sumPx.extent(1);
  int CD = C*D;
  std::vector<blitz::Array<double,1> > models_b;
  // Allocate and get the mean supervector
  for(size_t i=0; i<models.size(); ++i) {
    blitz::Array<double,1> mod(CD);
    models[i]->getMeanSupervector(mod);
    models_b.push_back(mod);
  }
  const blitz::Array<double,1>& ubm_mean = ubm.getMeanSupervector();
  const blitz::Array<double,1>& ubm_variance = ubm.getVarianceSupervector();
  detail::linearScoring(models_b, ubm_mean, ubm_variance, test_stats, &test_channelOffset, frame_length_normalisation, scores);
}

}}
