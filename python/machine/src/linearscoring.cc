/**
 * @file python/machine/src/linearscoring.cc
 * @date Wed Jul 13 16:00:04 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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
#include <boost/python.hpp>
#include <boost/shared_ptr.hpp>
#include "machine/LinearScoring.h"
#include <vector>

#include "core/python/ndarray.h"

using namespace boost::python;
namespace tp = bob::python;
namespace mach = bob::machine;

static void convertGMMMeanList(list models, std::vector<blitz::Array<double,1> >& models_c) {
  int size_models = len(models);
  for(int i=0; i<size_models; ++i) {
    models_c.push_back(extract<blitz::Array<double,1> >(models[i]));
  }
}

static void convertGMMStatsList(list test_stats, std::vector<boost::shared_ptr<const mach::GMMStats> >& test_stats_c) {
  int size_test_stats = len(test_stats);
  for(int i=0; i<size_test_stats; ++i) {
    boost::shared_ptr<mach::GMMStats> gs = extract<boost::shared_ptr<mach::GMMStats> >(test_stats[i]);
    test_stats_c.push_back(gs);
  }
}

static void convertChannelOffsetList(list test_channelOffset, std::vector<blitz::Array<double,1> >& test_channelOffset_c) {
  int size_test_channelOffset = len(test_channelOffset);
  for(int i=0; i<size_test_channelOffset; ++i) {
    test_channelOffset_c.push_back(extract<blitz::Array<double,1> >(test_channelOffset[i]));
  }
}

static void convertGMMMachineList(list models, std::vector<boost::shared_ptr<const mach::GMMMachine> >& models_c) {
  int size_models = len(models);
  for(int i=0; i<size_models; ++i) {
    boost::shared_ptr<mach::GMMMachine> gm = extract<boost::shared_ptr<mach::GMMMachine> >(models[i]);
    models_c.push_back(gm);
  }
}

static blitz::Array<double, 2> linearScoring1(list models,
    tp::const_ndarray ubm_mean, tp::const_ndarray ubm_variance,
    list test_stats, list test_channelOffset = list(), // Empty list
    bool frame_length_normalisation = false) 
{
  blitz::Array<double,1> ubm_mean_ = ubm_mean.bz<double,1>();
  blitz::Array<double,1> ubm_variance_ = ubm_variance.bz<double,1>();

  std::vector<blitz::Array<double,1> > models_c;
  convertGMMMeanList(models, models_c);

  std::vector<boost::shared_ptr<const mach::GMMStats> > test_stats_c;
  convertGMMStatsList(test_stats, test_stats_c);

  blitz::Array<double, 2> ret(len(models), len(test_stats));
  if (len(test_channelOffset) == 0) { //list is empty
    mach::linearScoring(models_c, ubm_mean_, ubm_variance_, test_stats_c, frame_length_normalisation, ret);
  }
  else { 
    std::vector<blitz::Array<double,1> > test_channelOffset_c;
    convertChannelOffsetList(test_channelOffset, test_channelOffset_c);
    mach::linearScoring(models_c, ubm_mean_, ubm_variance_, test_stats_c, test_channelOffset_c, frame_length_normalisation, ret);
  }
 
  return ret;
}

static blitz::Array<double, 2> linearScoring2(list models,
    mach::GMMMachine& ubm,
    list test_stats, list test_channelOffset = list(), // Empty list
    bool frame_length_normalisation = false) 
{
  std::vector<boost::shared_ptr<const mach::GMMMachine> > models_c;
  convertGMMMachineList(models, models_c);

  std::vector<boost::shared_ptr<const mach::GMMStats> > test_stats_c;
  convertGMMStatsList(test_stats, test_stats_c);

  blitz::Array<double, 2> ret(len(models), len(test_stats));
  if (len(test_channelOffset) == 0) { //list is empty
    mach::linearScoring(models_c, ubm, test_stats_c, frame_length_normalisation, ret);
  }
  else { 
    std::vector<blitz::Array<double,1> > test_channelOffset_c;
    convertChannelOffsetList(test_channelOffset, test_channelOffset_c);
    mach::linearScoring(models_c, ubm, test_stats_c, test_channelOffset_c, frame_length_normalisation, ret);
  }
  
  return ret;
}

BOOST_PYTHON_FUNCTION_OVERLOADS(linearScoring1_overloads, linearScoring1, 4, 6)
BOOST_PYTHON_FUNCTION_OVERLOADS(linearScoring2_overloads, linearScoring2, 3, 5)

void bind_machine_linear_scoring() {
  def("linearScoring", linearScoring1, linearScoring1_overloads(args("models", "ubm_mean", "ubm_variance", "test_stats", "test_channelOffset", "frame_length_normalisation"),
    "Compute a matrix of scores using linear scoring.\n"
    "Return a 2D matrix of scores, scores[m, s] is the score for model m against statistics s\n"
    "\n"
    "Warning Each GMM must have the same size.\n"
    "\n"
    "models       -- list of mean supervectors for the client models\n"
    "ubm_mean     -- mean supervector for the world model\n"
    "ubm_variance -- variance supervector for the world model\n"
    "test_stats   -- list of accumulate statistics for each test trial\n"
    "test_channelOffset -- \n"
    "frame_length_normlisation -- perform a normalisation by the number of feature vectors\n"
    ));
  def("linearScoring", linearScoring2, linearScoring2_overloads(args("models", "ubm", "test_stats", "test_channelOffset", "frame_length_normalisation"),
    "Compute a matrix of scores using linear scoring.\n"
    "Return a 2D matrix of scores, scores[m, s] is the score for model m against statistics s\n"
    "\n"
    "Warning Each GMM must have the same size.\n"
    "\n"
    "models      -- list of client models\n"
    "ubm         -- world model\n"
    "test_stats  -- list of accumulate statistics for each test trial\n"
    "test_channelOffset -- \n"
    "frame_length_normlisation -- perform a normalisation by the number of feature vectors\n"
  ));
}
