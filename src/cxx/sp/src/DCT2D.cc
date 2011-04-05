/**
 * @file src/cxx/sp/src/DCT2D.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based Fast Cosine Transform using FFTPACK functions
 */

#include "sp/DCT2D.h"

// Declaration of FORTRAN functions from FCTPACK4.1
extern "C" void cosqi_( int *n, double *wsave);
extern "C" void cosqf_( int *n, double *x, double *wsave);
extern "C" void cosqb_( int *n, double *x, double *wsave);


namespace sp = Torch::sp;

sp::DCT2D::DCT2D( const int height, const int width):
  m_height(height), m_width(width), m_wsave_w(0), m_wsave_h(0), m_col_tmp(0)
{
  // Precompute multiplicative factors
  m_sqrt_1h=sqrt(1./m_height);
  m_sqrt_2h=sqrt(2./m_height);
  m_sqrt_1w=sqrt(1./m_width);
  m_sqrt_2w=sqrt(2./m_width);

  // Precompute working arrays to save computation time
  initWorkingArrays();
}

sp::DCT2D::~DCT2D()
{
  cleanup();
}

void sp::DCT2D::operator()(const blitz::Array<double,2>& src, 
  blitz::Array<double,2>& dst)
{
  // TODO: check input!

  // Check and reindex if required
  if( dst.base(0) != 0 || dst.base(1) != 0) {
    const blitz::TinyVector<int,2> zero_base = 0;
    dst.reindexSelf( zero_base );
  }
  // Check and resize dst if required
  if( dst.extent(0) != src.extent(0) || dst.extent(1) != src.extent(1) )
    dst.resize( src.extent(0), src.extent(1) );

  // Initialize a blitz array for the result
  blitz::Array<double,2> dst_int;

  // Check if dst can be directly used
  bool dst_ok = checkSafedata(dst);

  // Make a reference to dst if this is posible.
  if( dst_ok )
    dst_int.reference( dst );
  // Otherwise, allocate a new "safe data()" array
  else
    dst_int.resize( src.extent(0), src.extent(1) );


  // Apply 1D FCT to each column of the 2D array (array(id_row,id_column))
  for(int j=0; j<m_width; ++j)
  {
    // Copy the column into the C array
    for( int i=0; i<m_height; ++i)
      m_col_tmp[i] = src(i,j);

    // Compute the FCT of one column
    cosqb_( &m_height, m_col_tmp, m_wsave_h);

    // Update the column
    for( int i=0; i<m_height; ++i)
      dst_int(i,j) = m_col_tmp[i];
  }

  // Apply 1D FCT to each row of the resulting matrix
  for(int i=0; i<m_height; ++i)
  {
    // Compute the FCT of one row
    cosqb_( &m_width, &(dst_int.data()[i*m_width]), m_wsave_w);
  }

  // Rescale the result
  for(int i=0; i<m_height; ++i)
    for(int j=0; j<m_width; ++j)
      dst_int(i,j) = dst_int(i,j)/16.*(i==0?m_sqrt_1h:m_sqrt_2h)*(j==0?m_sqrt_1w:m_sqrt_2w);

  // If required, copy the result back to dst
  if( dst_ok )
  {
    blitz::Range  r_dst_int0( dst_int.lbound(0), dst_int.ubound(0) ),
      r_dst_int1( dst_int.lbound(1), dst_int.ubound(1) ),
      r_dst0( dst.lbound(0), dst.ubound(0) ),
      r_dst1( dst.lbound(1), dst.ubound(1) );
    dst(r_dst0, r_dst1) = dst_int(r_dst_int0,r_dst_int1);
  }
}

void sp::DCT2D::initWorkingArrays() 
{
  int n_wsave_h = 3*m_height+15;
  m_wsave_h = new double[n_wsave_h];
  cosqi_( &m_height, m_wsave_h);

  int n_wsave_w = 3*m_width+15;
  m_wsave_w = new double[n_wsave_w];
  cosqi_( &m_width, m_wsave_w);

  m_col_tmp = new double[m_height];
}

void sp::DCT2D::cleanup() {
  if(m_wsave_w)
    delete [] m_wsave_w;
  if(m_wsave_h)
    delete [] m_wsave_h;
  if(m_col_tmp)
    delete [] m_col_tmp;
}
