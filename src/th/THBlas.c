#include "THBlas.h"
#include "THCBlas.h"

#define USE_DOUBLE
#define real double

void THBlas_swap(long size, real *x, long xStride, real *y, long yStride)
{
  if(size == 1)
  {
    xStride = 1;
    yStride = 1;
  }

#if USE_CBLAS
  if( (size < INT_MAX) && (xStride < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    cblas_dswap(size, x, xStride, y, yStride);
#else
    cblas_sswap(size, x, xStride, y, yStride);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < size; i++)
    {
      real z = x[i*xStride];
      x[i*xStride] = y[i*yStride];
      y[i*yStride] = z;
    }
  }
}

void THBlas_scale(long size, real alpha, real *y, long yStride)
{
  if(size == 1)
    yStride = 1;

#if USE_CBLAS
  if( (size < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    cblas_dscal(size, alpha, y, yStride);
#else
    cblas_sscal(size, alpha, y, yStride);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < size; i++)
      y[i*yStride] *= alpha;
  }
}

void THBlas_copy(long size, const real *x, long xStride, real *y, long yStride)
{
  if(size == 1)
  {
    xStride = 1;
    yStride = 1;
  }

#if USE_CBLAS
  if( (size < INT_MAX) && (xStride < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    cblas_dcopy(size, x, xStride, y, yStride);
#else
    cblas_scopy(size, x, xStride, y, yStride);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < size; i++)
      y[i*yStride] = x[i*xStride];
  }
}

void THBlas_add(long size, real alpha, real *x, long xStride, real *y, long yStride)
{
  if(size == 1)
  {
    xStride = 1;
    yStride = 1;
  }

#if USE_CBLAS
  if( (size < INT_MAX) && (xStride < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    cblas_daxpy(size, alpha, x, xStride, y, yStride);
#else
    cblas_saxpy(size, alpha, x, xStride, y, yStride);
#endif
    return;
  }
#endif
  {
    long i;
    for(i = 0; i < size; i++)
      y[i*yStride] += alpha*x[i*xStride];
  }
}

real THBlas_dot(long size, real *x, long xStride, real *y, long yStride)
{
  if(size == 1)
  {
    xStride = 1;
    yStride = 1;
  }

#if USE_CBLAS
  if( (size < INT_MAX) && (xStride < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    return cblas_ddot(size, x, xStride, y, yStride);
#else
    return cblas_sdot(size, x, xStride, y, yStride);
#endif
  }
#endif
  {
    long i;
    real sum = 0;
    for(i = 0; i < size; i++)
    sum += x[i*xStride]*y[i*yStride];
    return sum;
  }
}

void THBlas_matVec(int trans, long nRow, long nColumn, real alpha, real *m, long mStride, real *x, long xStride, real beta, real *y, long yStride)
{
  if(nColumn == 1)
    mStride = nRow;
  
#if USE_CBLAS
  if( (nRow < INT_MAX) && (nColumn < INT_MAX) && (mStride < INT_MAX)  && (xStride < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    if(trans)
      cblas_dgemv(CblasColMajor, CblasTrans, nRow, nColumn, alpha, m, mStride, x, xStride, beta, y, yStride);
    else
      cblas_dgemv(CblasColMajor, CblasNoTrans, nRow, nColumn, alpha, m, mStride, x, xStride, beta, y, yStride);
#else
    if(trans)
      cblas_sgemv(CblasColMajor, CblasTrans, nRow, nColumn, alpha, m, mStride, x, xStride, beta, y, yStride);
    else
      cblas_sgemv(CblasColMajor, CblasNoTrans, nRow, nColumn, alpha, m, mStride, x, xStride, beta, y, yStride);
#endif
    return;
  }
#endif
  {
    long r, c;

    if(trans)
    {
      if(beta == 1)
      {
        for(c = 0; c < nColumn; c++)
        {
          real sum = 0;
          real *column_ = m+mStride*c;
          for(r = 0; r < nRow; r++)
            sum += x[r*xStride]*column_[r];
          y[yStride*c] += alpha*sum;
        }
      }
      else
      {
        for(c = 0; c < nColumn; c++)
        {
          real sum = 0;
          real *column_ = m+mStride*c;
          for(r = 0; r < nRow; r++)
            sum += x[r*xStride]*column_[r];
          y[yStride*c] = beta*y[yStride*c] + alpha*sum;
        }
      }
    }
    else
    {
      if(beta != 1)
        THBlas_scale(nRow, beta, y, yStride);
      
      for(c = 0; c < nColumn; c++)
      {
        real *column_ = m+mStride*c;
        real z = alpha*x[c*xStride];
        for(r = 0; r < nRow; r++)
          y[yStride*r] += z*column_[r];
      }
    }
  }
}

void THBlas_outerProduct(long nRow, long nColumn, real alpha, real *x, long xStride, real *y, long yStride, real *m, long mStride)
{
  if(nColumn == 1)
    mStride = nRow;

#if USE_CBLAS
  if( (nRow < INT_MAX) && (nColumn < INT_MAX) && (mStride < INT_MAX)  && (xStride < INT_MAX) && (yStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    cblas_dger(CblasColMajor, nRow, nColumn, alpha, x, xStride, y, yStride, m, mStride);
#else
    cblas_sger(CblasColMajor, nRow, nColumn, alpha, x, xStride, y, yStride, m, mStride);
#endif
    return;
  }
#endif
  {
    long r, c;
    for(c = 0; c < nColumn; c++)
    {
      real *column_ = m+c*mStride;
      real z = alpha*y[c*yStride];
      for(r = 0; r < nRow; r++)
        column_[r] += z*x[r*xStride] ;
    }
  }
}

void THBlas_matMat(int transA, int transB, long nRow, long nColumn, long nRC, real alpha, real *A, long AStride, real *B, long BStride, real beta, real *C, long CStride)
{  
  if(nColumn == 1)
    CStride = nRow;

  if(transA)
  {
    if(nRow == 1)
      AStride = nRC;
  }
  else
  {
    if(nRC == 1)
      AStride = nRow;
  }

  if(transB)
  {
    if(nRC == 1)
      BStride = nColumn;
  }
  else
  {
    if(nColumn == 1)
      BStride = nRC;
  }

#if USE_CBLAS
  if( (nRow < INT_MAX) && (nColumn < INT_MAX) && (nRC < INT_MAX) && (AStride < INT_MAX)  && (BStride < INT_MAX) && (CStride < INT_MAX) )
  {
#ifdef USE_DOUBLE
    cblas_dgemm(CblasColMajor, (transA ? CblasTrans : CblasNoTrans), (transB ? CblasTrans : CblasNoTrans), nRow, nColumn, nRC, alpha, A, AStride, B, BStride, beta, C, CStride);
#else
    cblas_sgemm(CblasColMajor, (transA ? CblasTrans : CblasNoTrans), (transB ? CblasTrans : CblasNoTrans), nRow, nColumn, nRC, alpha, A, AStride, B, BStride, beta, C, CStride);
#endif
    return;
  }
#endif
  {
    long r, c, i;
    if(!transA && !transB)
    {
      real *A_ = A;
      for(r = 0; r < nRow; r++)
      {
        real *B_ = B;
        for(c = 0; c < nColumn; c++)
        {
          real sum = 0;
          for(i = 0; i < nRC; i++)
            sum += A_[i*AStride]*B_[i];
          B_ += BStride;
          C[c*CStride+r] = beta*C[c*CStride+r]+alpha*sum;
        }
        A_++;
      }
    }
    else if(transA && !transB)
    {
      real *A_ = A;
      for(r = 0; r < nRow; r++)
      {
        real *B_ = B;
        for(c = 0; c < nColumn; c++)
        {
          real sum = 0;
          for(i = 0; i < nRC; i++)
            sum += A_[i]*B_[i];
          B_ += BStride;
          C[c*CStride+r] = beta*C[c*CStride+r]+alpha*sum;
        }
        A_ += AStride;
      }
    }
    else if(!transA && transB)
    {
      real *A_ = A;
      for(r = 0; r < nRow; r++)
      {
        real *B_ = B;
        for(c = 0; c < nColumn; c++)
        {
          real sum = 0;
          for(i = 0; i < nRC; i++)
            sum += A_[i*AStride]*B_[i*BStride];
          B_++;
          C[c*CStride+r] = beta*C[c*CStride+r]+alpha*sum;
        }
        A_++;
      }
    }
    else
    {
      real *A_ = A;
      for(r = 0; r < nRow; r++)
      {
        real *B_ = B;
        for(c = 0; c < nColumn; c++)
        {
          real sum = 0;
          for(i = 0; i < nRC; i++)
            sum += A_[i]*B_[i*BStride];
          B_++;
          C[c*CStride+r] = beta*C[c*CStride+r]+alpha*sum;
        }
        A_ += AStride;
      }
    }
  }
}
