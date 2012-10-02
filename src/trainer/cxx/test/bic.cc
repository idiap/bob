/**
 * @file trainer/cxx/test/bic.cc
 * @date Wed Jun  6 10:29:09 CEST 2012
 * @author Manuel Guenther <Manuel.Guenther@idiap.ch>
 *
 * @brief Test the BIC trainer and machine
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

#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE Trainer-bic Tests
#define BOOST_TEST_MAIN
#include <boost/test/unit_test.hpp>
#include <blitz/array.h>
#include <stdint.h>

#include "bob/core/cast.h"
#include "bob/trainer/BICTrainer.h"
#include "bob/machine/BICMachine.h"


struct Data {
  double epsilon;
  blitz::Array<double,2> intra_data, extra_data;

  Data():
    epsilon(1e-10),
    intra_data(5,5),
    extra_data(5,5)
  {
    intra_data = 10., 4., 6., 8., 2., 8., 2., 4., 6., 0., 12., 6., 8., 10., 4., 9., 3., 5., 7., 1., 11., 5., 7., 9., 3.;
    extra_data = -intra_data;
  }
};



BOOST_FIXTURE_TEST_SUITE( test_setup, Data )

BOOST_AUTO_TEST_CASE( test_iec )
{
  bob::trainer::BICTrainer trainer;
  bob::machine::BICMachine machine;

  // train machine only with intrapersonal data
  trainer.train(machine, intra_data, intra_data);
  // => all projection results MUST be 0.
  blitz::Array<double,1> data(5), output(1);
  data = 10.;
  machine.forward(data, output);
  BOOST_CHECK_SMALL(output(0), epsilon);

  // now, re-train machine with both data
  trainer.train(machine, intra_data, extra_data);
  // hmm... the result should now (at least) be greater than 0
  machine.forward(data, output);
  BOOST_CHECK_GT(output(0), 0.);

  // due to the training data, input 0 should return in a 0 output
  data = 0.;
  machine.forward(data, output);
  BOOST_CHECK_SMALL(output(0), epsilon);
}

BOOST_AUTO_TEST_CASE( test_bic )
{
  bob::trainer::BICTrainer trainer(1,1);
  bob::machine::BICMachine machine;

  // train machine only with intrapersonal data
  trainer.train(machine, intra_data, intra_data);

  // => all projection results MUST be 0.
  blitz::Array<double,1> data(5), output(1);
  data = 10.;
  machine.forward(data, output);
  BOOST_CHECK_SMALL(output(0), epsilon);

  // now, re-train machine with both data
  trainer.train(machine, intra_data, extra_data);
  // hmm... the result should now (at least) be greater than 0
  machine.forward(data, output);
  BOOST_CHECK_GT(output(0), 0.);

  // due to the training data, input 0 should return in a 0 output
  data = 0.;
  machine.forward(data, output);
  BOOST_CHECK_SMALL(output(0), epsilon);
}


BOOST_AUTO_TEST_SUITE_END()
