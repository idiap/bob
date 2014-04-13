/**
 * @file machine/cxx/roll.cc
 * @date Tue Jun 25 18:52:26 CEST 2013
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
 */

#include <bob/core/assert.h>
#include <bob/machine/roll.h>

int bob::machine::detail::getNbParameters(const bob::machine::MLP& machine)
{
  const std::vector<blitz::Array<double,1> >& b = machine.getBiases();
  const std::vector<blitz::Array<double,2> >& w = machine.getWeights();
  return bob::machine::detail::getNbParameters(w, b);
}

int bob::machine::detail::getNbParameters(
  const std::vector<blitz::Array<double,2> >& w,
  const std::vector<blitz::Array<double,1> >& b)
{
  bob::core::array::assertSameDimensionLength(w.size(), b.size());
  int N = 0;
  for (int i=0; i<(int)w.size(); ++i)
    N += b[i].numElements() + w[i].numElements();
  return N;
}

void bob::machine::unroll(const bob::machine::MLP& machine,
  blitz::Array<double,1>& vec) 
{
  const std::vector<blitz::Array<double,1> >& b = machine.getBiases();
  const std::vector<blitz::Array<double,2> >& w = machine.getWeights();
  unroll(w, b, vec);
}

void bob::machine::unroll(const std::vector<blitz::Array<double,2> >& w,
  const std::vector<blitz::Array<double,1> >& b, blitz::Array<double,1>& vec)
{
  // 1/ Check number of elements
  const int N = bob::machine::detail::getNbParameters(w, b);
  bob::core::array::assertSameDimensionLength(vec.extent(0), N);

  // 2/ Roll
  blitz::Range rall = blitz::Range::all();
  int offset=0;
  for (int i=0; i<(int)w.size(); ++i)
  {
    const int Nb = b[i].extent(0);
    blitz::Range rb(offset,offset+Nb-1);
    vec(rb) = b[i];
    offset += Nb;

    const int Nw0 = w[i].extent(0);
    const int Nw1 = w[i].extent(1);
    blitz::TinyVector<int,1> tv(Nw1);
    for (int j=0; j<Nw0; ++j)
    {
      blitz::Range rw(offset,offset+Nw1-1);
      vec(rw) = w[i](j,rall);
      offset += Nw1;
    }
  }
}

void bob::machine::roll(bob::machine::MLP& machine,
  const blitz::Array<double,1>& vec)
{
  std::vector<blitz::Array<double,1> >& b = machine.updateBiases();
  std::vector<blitz::Array<double,2> >& w = machine.updateWeights();
  roll(w, b, vec);
}

void bob::machine::roll(std::vector<blitz::Array<double,2> >& w,
  std::vector<blitz::Array<double,1> >& b, const blitz::Array<double,1>& vec)
{
  // 1/ Check number of elements
  const int N = bob::machine::detail::getNbParameters(w, b);
  bob::core::array::assertSameDimensionLength(vec.extent(0), N);

  // 2/ Roll
  blitz::Range rall = blitz::Range::all();
  int offset=0;
  for (int i=0; i<(int)w.size(); ++i)
  {
    const int Nb = b[i].extent(0);
    blitz::Array<double,1> vb = vec(blitz::Range(offset,offset+Nb-1));
    b[i] = vb;
    offset += Nb;

    const int Nw0 = w[i].extent(0);
    const int Nw1 = w[i].extent(1);
    blitz::TinyVector<int,1> tv(Nw1);
    for (int j=0; j<Nw0; ++j)
    {
      blitz::Array<double,1> vw = vec(blitz::Range(offset,offset+Nw1-1));
      w[i](j,rall) = vw;
      offset += Nw1;
    }
  }
}
