/**
 * @author Laurent El-Shafey <Laurent.El-Shafey@idiap.ch>
 * @date Thu 14 Jul 2011 16:52:07
 *
 * @brief Joint Factor Analysis Trainer
 */

#include "trainer/JFATrainer.h"
#include "math/lu_det.h"
#include "math/linear.h"
#include "core/Exception.h"
#include "core/repmat.h"
#include <algorithm>

namespace train = Torch::trainer;

void train::jfa::updateEigen(const blitz::Array<double,3> &A, 
  const blitz::Array<double,2> &C, blitz::Array<double,2> &uv)
{
  // Check dimensions: ru
  int ru = A.extent(1);
  if(A.extent(2) != ru || C.extent(0) != ru || uv.extent(0) != ru)
    throw Torch::core::Exception();

  // Supervector length
  if(C.extent(1) != uv.extent(1))
    throw Torch::core::Exception();

  // Number of Gaussian components
  int Ng = A.extent(0); // Number of Gaussians
  if(C.extent(1) % Ng != 0)
    throw Torch::core::Exception();

  // Dimensionality, where cd = 
  //   c (Number of Gaussians) * d (Dimensionality of each Gaussian)
  int D = C.extent(1) / A.extent(0);
  
  // Intermediate array
  blitz::Array<double,2> tmp(ru, ru);
  // Initialize to 0
  uv = 0.;
  // Update U
  for(int c=0; c<Ng; ++c)
  {
    blitz::Array<double,2> c_elements = C(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,2> uv_elements = uv(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,2> A_c = A(c,blitz::Range::all(),blitz::Range::all());
    // Compute inverse of A
    Torch::math::inv(A_c, tmp);
    Torch::math::prod(tmp, c_elements, uv_elements);
  }
}

void train::jfa::estimateXandU(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E,
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v,
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z,
  const blitz::Array<double,2> &y, blitz::Array<double,2> &x,
  const blitz::Array<uint32_t,1> &spk_ids)
{
  // 1/ Check inputs
  // Number of Gaussians
  int C = N.extent(1);

  // Dimensionality
  int CD = F.extent(1);
  if( CD % C != 0)
    throw Torch::core::Exception();
  int D = CD / C;

  // Number of training segments
  int T = F.extent(0);
  if(N.extent(0) != T || spk_ids.extent(0) != T || x.extent(0) != T)
    throw Torch::core::Exception();

  // Supervector length
  if(m.extent(0) != CD || E.extent(0) != CD || d.extent(0) != CD || 
      v.extent(1) != CD || u.extent(1) != CD || z.extent(1) != CD)
    throw Torch::core::Exception();

  // rv and ru lengths
  int rv = v.extent(0);
  int ru = u.extent(0);
  if(y.extent(1) != rv || x.extent(1) != ru)
    throw Torch::core::Exception();
   
  // N speakers
  int Nspk = z.extent(0);
  if(y.extent(0) != Nspk)
    throw Torch::core::Exception();
  
  //index_map = reshape(repmat(1:size(N,2), dim,1),size(F,2),1);
  //x = zeros(size(spk_ids,1), size(u,1));

  // 2/ Initialize uEuT
  blitz::Array<double,3> uEuT(C, ru, ru);
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Array<double,2> tmp1(ru, D);
  for(int c=0; c<C; ++c)
  {
    blitz::Array<double,2> uEuT_c = uEuT(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> u_elements = u(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,1> e_elements = E(blitz::Range(c*D, (c+1)*D-1));

    tmp1 = u_elements(i,j) / e_elements(j);
    blitz::Array<double,2> u_transposed = u_elements.transpose(1,0);
    
    Torch::math::prod(tmp1, u_transposed, uEuT_c);
  }

  // 3/ Determine the vector of speaker ids
  //    Assume that samples from the same speaker are consecutive
  std::vector<uint32_t> ids;
  uint32_t last_elem;
  for(int ind=0; ind<T; ++ind)
  {
    if(ids.empty())
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
    else if(last_elem!=spk_ids(ind))
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
  }

  // 4/ Main computation
  // Allocate working arrays
  blitz::Array<double,1> spk_shift(CD);
  blitz::Array<double,1> tmp2(CD);
  blitz::Array<double,1> tmp3(ru);
  blitz::Array<double,1> Fh(CD);
  blitz::Array<double,2> L(ru, ru);
  blitz::Array<double,2> Linv(ru, ru);

  std::vector<uint32_t>::iterator it;
  int cur_start_ind=0;
  int cur_end_ind;
  for(it=ids.begin(); it<ids.end(); ++it)
  {
    // a/ Determine speaker sessions
    uint32_t cur_elem=*it;
    cur_start_ind = 0;
    while(spk_ids(cur_start_ind) != cur_elem)
      ++cur_start_ind;
    cur_end_ind = cur_start_ind;
    for(int ind=cur_start_ind+1; ind<T; ++ind)
    {
      cur_end_ind = ind;
      if(spk_ids(ind)!=cur_elem)
      {
        cur_end_ind = ind-1;
        break;
      }
    }

    // b/ Compute speaker shift
    spk_shift = m;
    blitz::Array<double,1> y_ii = y(cur_elem,blitz::Range::all());
    Torch::math::prod(y_ii, v, tmp2);
    spk_shift += tmp2;
    blitz::Array<double,1> z_ii = z(cur_elem,blitz::Range::all());
    spk_shift += z_ii * d;
   
    // c/ Loop over speaker session
    for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
    {
      blitz::Array<double,1> Nhint = N(jj, blitz::Range::all());
      Torch::core::repelem(Nhint, tmp2);  
      Fh = F(jj, blitz::Range::all()) - tmp2 * spk_shift;
      
      // L=Identity
      L = 0.;
      for(int k=0; k<CD; ++k)
        L(k,k) = 1.;
    
      for(int c=0; c<C; ++c)
      {
        blitz::Array<double,2> uEuT_c = uEuT(c, blitz::Range::all(), blitz::Range::all());
        L += uEuT_c * N(jj,c);
      }

      // inverse L
      Torch::math::inv(L, Linv);

      // update x
      blitz::Array<double,1> x_jj = x(jj,blitz::Range::all());
      Fh /= E;
      blitz::Array<double,2> u_t = u.copy().transpose(1,0);
      Torch::math::prod(Fh, u_t, tmp3);
      Torch::math::prod(tmp3, Linv, x_jj);
    }
  }
}


void train::jfa::estimateYandV(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N, 
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, const blitz::Array<double,2> &z, 
  blitz::Array<double,2> &y, const blitz::Array<double,2> &x, 
  const blitz::Array<uint32_t,1> &spk_ids)
{
  // 1/ Check inputs
  // Number of Gaussians
  int C = N.extent(1);
  
  // Dimensionality
  int CD = F.extent(1);
  if( CD % C != 0)
    throw Torch::core::Exception();
  int D = CD / C;

  // Number of training segments
  int T = F.extent(0);
  if(N.extent(0) != T || spk_ids.extent(0) != T || x.extent(0) != T)
    throw Torch::core::Exception();

  // Supervector length
  if(m.extent(0) != CD || E.extent(0) != CD || d.extent(0) != CD || 
      v.extent(1) != CD || u.extent(1) != CD || z.extent(1) != CD)
    throw Torch::core::Exception();

  // rv and ru lengths
  int rv = v.extent(0);
  int ru = u.extent(0);
  if(y.extent(1) != rv || x.extent(1) != ru)
    throw Torch::core::Exception();
   
  // N speakers
  int Nspk = z.extent(0);
  if(y.extent(0) != Nspk)
    throw Torch::core::Exception();


  blitz::Array<double,3> vEvT(C, rv, rv);
  blitz::firstIndex i;
  blitz::secondIndex j;
  blitz::Array<double,2> tmp1(rv,D);
  for(int c=0; c<C; ++c)
  {
    blitz::Array<double,2> vEvT_c = vEvT(c, blitz::Range::all(), blitz::Range::all());
    blitz::Array<double,2> v_elements = v(blitz::Range::all(), 
      blitz::Range(c*D, (c+1)*D-1));
    blitz::Array<double,1> e_elements = E(blitz::Range(c*D, (c+1)*D-1));

    tmp1 = v_elements(i,j) / e_elements(j);
    blitz::Array<double,2> v_transposed = v_elements.transpose(1,0);
    Torch::math::prod(tmp1, v_transposed, vEvT_c);
  }

  // Determine a vector of speaker ids
  std::vector<uint32_t> ids;
  uint32_t last_elem;
  for(int ind=0; ind<spk_ids.extent(0); ++ind)
  {
    if(ids.empty())
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
    else if(last_elem!=spk_ids(ind))
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
  }

  // Allocate working arrays
  blitz::Array<double,1> Fs(CD);
  blitz::Array<double,1> Nss(C);
  blitz::Array<double,1> Ns(CD);
  blitz::Array<double,1> tmp2(CD);
  blitz::Array<double,2> L(rv, rv);
  blitz::Array<double,2> Linv(rv, rv);
  blitz::Array<double,1> tmp3(rv);
  blitz::Array<double,1> tmp4(CD);

  std::vector<uint32_t>::iterator it;
  int cur_start_ind=0;
  int cur_end_ind;
  for(it=ids.begin(); it<ids.end(); ++it)
  {
    // Determine speaker sessions
    uint32_t cur_elem=*it;
    cur_start_ind=0;
    while(spk_ids(cur_start_ind) != cur_elem)
      ++cur_start_ind;
    cur_end_ind = cur_start_ind;
    for(int ind=cur_start_ind+1; ind<spk_ids.extent(0); ++ind)
    {
      cur_end_ind = ind;
      if(spk_ids(ind)!=cur_elem)
      {
        cur_end_ind = ind-1;
        break;
      }
    }

    // Extract speaker sessions
    blitz::Array<double,2> Fs_sessions = F(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());
    blitz::Array<double,2> Nss_sessions = N(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());

    blitz::firstIndex i;
    blitz::secondIndex j;
    Fs = blitz::sum(Fs_sessions(j,i), j);
    Nss = blitz::sum(Nss_sessions(j,i), j);
    Torch::core::repelem(Nss, Ns);

    blitz::Array<double,1> z_ii = z(cur_elem,blitz::Range::all());
    Fs -= ((m + z_ii * d) * Ns);

    // Loop over speaker session
    for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
    {
      // update x
      blitz::Array<double,1> x_jj = x(jj,blitz::Range::all());
      Torch::math::prod(x_jj, u, tmp2);
      blitz::Array<double,1> N_jj = N(jj,blitz::Range::all());
      Torch::core::repelem(N_jj, tmp4);
      Fs -= tmp2 * tmp4;
    }

    // L=Identity
    L = 0.;
    for(int k=0; k<rv; ++k)
      L(k,k) = 1.;

    for(int c=0; c<C; ++c)
    {
      blitz::Array<double,2> vEvT_c = vEvT(c, blitz::Range::all(), blitz::Range::all());
      L += vEvT_c * Nss(c);
    }

    // inverse L
    Torch::math::inv(L, Linv);

    // update y
    blitz::Array<double,1> y_ii = y(cur_elem,blitz::Range::all());
    Fs /= E;
    blitz::Array<double,2> v_t = v.transpose(1,0);
    Torch::math::prod(Fs, v_t, tmp3);
    Torch::math::prod(tmp3, Linv, y_ii);
  }    
}




void train::jfa::estimateZandD(const blitz::Array<double,2> &F, const blitz::Array<double,2> &N,
  const blitz::Array<double,1> &m, const blitz::Array<double,1> &E, 
  const blitz::Array<double,1> &d, const blitz::Array<double,2> &v, 
  const blitz::Array<double,2> &u, blitz::Array<double,2> &z, 
  const blitz::Array<double,2> &y, const blitz::Array<double,2> &x, 
  const blitz::Array<uint32_t,1> &spk_ids)
{
  // 1/ Check inputs
  // Number of Gaussians
  int C = N.extent(1);
  
  // Dimensionality
  int CD = F.extent(1);
  if( CD % C != 0)
    throw Torch::core::Exception();
  //int D = CD / C;

  // Number of training segments
  int T = F.extent(0);
  if(N.extent(0) != T || spk_ids.extent(0) != T || x.extent(0) != T)
    throw Torch::core::Exception();

  // Supervector length
  if(m.extent(0) != CD || E.extent(0) != CD || d.extent(0) != CD || 
      v.extent(1) != CD || u.extent(1) != CD || z.extent(1) != CD)
    throw Torch::core::Exception();

  // rv and ru lengths
  int rv = v.extent(0);
  int ru = u.extent(0);
  if(y.extent(1) != rv || x.extent(1) != ru)
    throw Torch::core::Exception();
   
  // N speakers
  int Nspk = z.extent(0);
  if(y.extent(0) != Nspk)
    throw Torch::core::Exception();

  // Determine a vector of speaker ids
  std::vector<uint32_t> ids;
  uint32_t last_elem;
  for(int ind=0; ind<T; ++ind)
  {
    if(ids.empty())
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
    else if(last_elem!=spk_ids(ind))
    {
      ids.push_back(spk_ids(ind));
      last_elem=spk_ids(ind);
    }
  }

  // Allocate arrays
  blitz::Array<double,1> shift(CD);
  blitz::Array<double,1> tmp1(CD);
  blitz::Array<double,1> Fs(CD);
  blitz::Array<double,1> Nss(C);
  blitz::Array<double,1> Ns(CD);
  blitz::Array<double,1> L(CD);

  std::vector<uint32_t>::iterator it;
  int cur_start_ind=0;
  int cur_end_ind;
  for(it=ids.begin(); it<ids.end(); ++it)
  {
    // Determine speaker sessions
    uint32_t cur_elem=*it;
    cur_start_ind=0;
    while(spk_ids(cur_start_ind) != cur_elem)
      ++cur_start_ind;
    cur_end_ind = cur_start_ind;
    for(int ind=cur_start_ind+1; ind<T; ++ind)
    {
      cur_end_ind = ind;
      if(spk_ids(ind)!=cur_elem)
      {
        cur_end_ind = ind-1;
        break;
      }
    }

    // Extract speaker sessions
    blitz::Array<double,2> Fs_sessions = F(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());
    blitz::Array<double,2> Nss_sessions = N(blitz::Range(cur_start_ind,cur_end_ind), blitz::Range::all());

    blitz::firstIndex i;
    blitz::secondIndex j;
    Fs = blitz::sum(Fs_sessions(j,i), j);
    Nss = blitz::sum(Nss_sessions(j,i), j);
    Torch::core::repelem(Nss, Ns);

    // Compute shift
    shift = m;
    blitz::Array<double,1> y_ii = y(cur_elem,blitz::Range::all());
    Torch::math::prod(y_ii, v, tmp1);
    shift += tmp1;
    Fs -= shift * Ns;

    // Loop over speaker session
    for(int jj=cur_start_ind; jj<=cur_end_ind; ++jj)
    {
      // update x
      blitz::Array<double,1> x_jj = x(jj,blitz::Range::all());
      Torch::math::prod(x_jj, u, shift);
      blitz::Array<double,1> N_jj = N(jj,blitz::Range::all());
      Torch::core::repelem(N_jj, tmp1);
      Fs -= shift * tmp1;
    }

    L = 1.;
    L += Ns / E * blitz::pow2(d);

    // Update z   
    // z(ii,:) = Fs ./ E .* d ./L;
    blitz::Array<double,1> z_ii = z(cur_elem,blitz::Range::all());
    z_ii = Fs / E * d / L;
  }
 
}

