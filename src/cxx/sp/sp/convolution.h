/**
 * @file src/cxx/sp/sp/convolution.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based convolution product with zero padding
 */

#ifndef TORCH5SPRO_SP_CONVOLVE_H
#define TORCH5SPRO_SP_CONVOLVE_H

#include "core/logging.h"
#include "core/common.h"
#include "core/Exception.h"
#include <blitz/array.h>

namespace tc = Torch::core;

namespace Torch {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    /**
     * @brief Enumerations of the possible options
     */
    namespace Convolution {
      enum SizeOption {
        Full,
        Same,
        Valid
      };

      enum BorderOption {
        Zero,
        NearestNeighbour,
        Circular,
        Mirror
      };
    }

    /**
     * @brief 1D convolution of blitz arrays: A=B*C
     * @param B The first input array B
     * @param C The second input array C
     * @param A The output array A=B*C
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between B and C
     *                   * Valid: valid (part without padding)
     * @param border_opt:  * Zero: zero padding
     *                     * Nearest Neighbour: extrapolate with nearest 
     *                         neighbour
     *                     * Circular: extrapolate by considering tiled arrays
     *                         for B and C (<-> modulo arrays)
     *                     * Mirror: extrapolate by mirroring the arrays
     *                         for B and C
     * @warning If size(C) < size(B),  B and C are reversed and the convolve
     *   function is called again.
     */
    template<typename T> void convolve(const blitz::Array<T,1>& B, 
      const blitz::Array<T,1>& C, blitz::Array<T,1>& A,
      const enum Convolution::SizeOption size_opt = Convolution::Full,
      const enum Convolution::BorderOption border_opt = Convolution::Zero)
    {
      int Bl = B.lbound(0);
      int Cl = C.lbound(0);
      int Bsize = B.extent(0);
      int Csize = C.extent(0);

      // Assume that B is larger than C otherwise reverse them
      if( Bsize < Csize)
      {
        convolve( C, B, A, size_opt, border_opt);
        return;
      }

      int Asize;

      // Size of "B + C - 1"
      if( size_opt == Convolution::Full )
        Asize = Bsize + Csize - 1;
      // Same size as B
      else if( size_opt == Convolution::Same )
        Asize = Bsize;
      // Size when not allowing any padding
      else if( size_opt == Convolution::Valid )
      {
        Asize = Bsize - Csize + 1;
        // Check that B is larger than C, otherwise, throw an exception
        if( Bsize < Csize )
          throw Torch::core::Exception();
      }

      // Check and resize A if required
      if( A.extent(0) != Asize )
        A.resize( Asize );
      // Check and reindex A if required
      if( A.base(0) != 0 ) {
        const blitz::TinyVector<int,1> zero_base = 0;
        A.reindexSelf( zero_base );
      }

      for(int i=0; i < Asize; ++i)
      {
        A(i) = 0;

        int i_shifted;
        if( size_opt == Convolution::Full )
          i_shifted = i;
        else if( size_opt == Convolution::Same )
          i_shifted = i + Csize / 2;

        if( size_opt == Convolution::Full || size_opt == Convolution::Same )
        {
          if( border_opt == Convolution::Zero )
          {
            int jl = ( i_shifted - (Csize-1) > 0 ? i_shifted - (Csize-1) : 0 );
            int jh = ( i_shifted < Bsize ? i_shifted : Bsize-1 ); 
            blitz::Range jb( jl, jh), jc( i_shifted-jl, i_shifted-jh, -1); 
            A(i) = blitz::sum(B(jb + Bl) * C(jc + Cl) );
          }
          else if( border_opt == Convolution::NearestNeighbour )
          {
            for(int j=i_shifted-(Csize-1); j <= i_shifted; ++j)
              A(i) += B( tc::keepInRange(j,0,Bsize-1) + Bl) * 
                C( tc::keepInRange(i_shifted-j,0,Csize-1) + Cl);
          }
          else if( border_opt == Convolution::Circular )
          {
            for(int j=i_shifted-(Csize-1); j <= i_shifted; ++j)
              A(i) += B( (((j%Bsize)+Bsize) % Bsize) + Bl) * 
                C( i_shifted-j + Cl);
          }
          else if( border_opt == Convolution::Mirror )
          {
            for(int j=i_shifted-(Csize-1); j <= i_shifted; ++j)
              A(i) += B( tc::mirrorInRange(j,0,Bsize-1) + Bl) * 
                C( tc::mirrorInRange(i_shifted-j,0,Csize-1) + Cl);
          }
        }
        else if( size_opt == Convolution::Valid )
        {
          // Interpolation is useless in this case
          blitz::Range jb( i, i + Csize - 1), jc( Csize - 1, 0, -1); 
          A(i) = blitz::sum(B(jb + Bl) * C(jc + Cl) );
        }
        else
          throw Torch::core::Exception();
      }
    }

    /**
     * @brief 2D convolution of 2D blitz arrays: A=B*C
     * @param B The first input array B
     * @param C The second input array C
     * @param A The output array A=B*C
     * @param size_opt:  * Full: full size (default)
     *                   * Same: same size as the largest between B and C
     *                   * Valid: valid (part without padding)
     * @param border_opt:  * Zero: zero padding
     *                     * Nearest Neighbour: extrapolate with nearest 
     *                         neighbour
     *                     * Circular: extrapolate by considering tiled arrays
     *                         fir B and C (<-> modulo arrays)
     *                     * Mirror: extrapolate by mirroring the arrays
     *                         for B and C
     * @warning If size(C) < size(B),  B and C are reversed and the convolve
     *   function is called again. If both B and C have a leading dimension,
     *   an exception is thrown
     */
    template<typename T> void convolve(const blitz::Array<T,2>& B, 
      const blitz::Array<T,2>& C, blitz::Array<T,2>& A,
      const enum Convolution::SizeOption size_opt = Convolution::Full,
      const enum Convolution::BorderOption border_opt = Convolution::Zero)
    {
      int Bl1 = B.lbound(0);
      int Bl2 = B.lbound(1);
      int Cl1 = C.lbound(0);
      int Cl2 = C.lbound(1);
      int Bsize1 = B.extent(0);
      int Bsize2 = B.extent(1);
      int Csize1 = C.extent(0);
      int Csize2 = C.extent(1);

      // Assume that B is larger than C otherwise reverse them or throw an
      // exception if there is no 'larger' 2D array.
      if( Bsize1 < Csize1 && Bsize2 < Csize2)
      {
        convolve( C, B, A, size_opt, border_opt);
        return;
      }
      else if( !(Bsize1 >= Csize1 && Bsize2 >= Csize2) ) 
      {
        throw Torch::core::Exception();
      }


      int Asize1;
      int Asize2;

      if( size_opt == Convolution::Full )
      {
        Asize1 = Bsize1 + Csize1 - 1;
        Asize2 = Bsize2 + Csize2 - 1;
      }
      else if( size_opt == Convolution::Same )
      {
        Asize1 = Bsize1;
        Asize2 = Bsize2;
      }
      else if( size_opt == Convolution::Valid )
      {
        Asize1 = Bsize1 - Csize1 + 1;
        Asize2 = Bsize2 - Csize2 + 1;
        // Check that B is larger than C, otherwise, throw an exception
        if( Bsize1 < Csize1 || Bsize2 < Csize2)
          throw Torch::core::Exception();
      }

      // Check and resize A if required
      if( A.extent(0) != Asize1 || A.extent(1) != Asize2)
        A.resize( Asize1, Asize2 );
      // Check and reindex A if required
      if( A.base(0) != 0 || A.base(1) != 0 ) {
        const blitz::TinyVector<int,2> zero_base = 0;
        A.reindexSelf( zero_base );
      }

      for(int i1=0; i1 < Asize1; ++i1)
      {
        int i1_shifted;
        if( size_opt == Convolution::Full )
          i1_shifted = i1;
        else if( size_opt == Convolution::Same )
          i1_shifted = i1 + Csize1 / 2;

        for(int i2=0; i2 < Asize2; ++i2)
        {
          A(i1,i2) = 0;

          int i2_shifted = i2;
          if( size_opt == Convolution::Full )
            i2_shifted = i2;
          else if( size_opt == Convolution::Same )
            i2_shifted = i2 + Csize2 / 2;

          if( size_opt == Convolution::Full || size_opt == Convolution::Same )
          {
            if( border_opt == Convolution::Zero )
            {
              int jl1 = ( i1_shifted - (Csize1-1) > 0 ? 
                i1_shifted - (Csize1-1) : 0 );
              int jh1 = ( i1_shifted < Bsize1 ? i1_shifted : Bsize1-1 );
              int jl2 = ( i2_shifted - (Csize2-1) > 0 ? 
                i2_shifted - (Csize2-1) : 0 );
              int jh2 = ( i2_shifted < Bsize2 ? i2_shifted : Bsize2-1 );
              blitz::Range jb1( jl1, jh1), jb2( jl2, jh2);
              blitz::Range jc1( i1_shifted-jl1, i1_shifted-jh1,-1), 
                           jc2( i2_shifted-jl2, i2_shifted-jh2,-1);
              A(i1,i2) = blitz::sum(B(jb1 + Bl1, jb2 + Bl2) * 
                            C(jc1 + Cl1, jc2 + Cl2) );
            }
            else if( border_opt == Convolution::NearestNeighbour )
            {
              for(int j1=i1_shifted-(Csize1-1); j1 <= i1_shifted; ++j1)
                for(int j2=i2_shifted-(Csize2-1); j2 <= i2_shifted; ++j2)
                  A(i1,i2) += B( tc::keepInRange(j1,0,Bsize1-1) + Bl1, 
                               tc::keepInRange(j2,0,Bsize2-1) + Bl2) *
                    C( tc::keepInRange(i1_shifted-j1,0,Csize1-1) + Cl1,
                       tc::keepInRange(i2_shifted-j2,0,Csize2-1) + Cl2);
            }
            else if( border_opt == Convolution::Circular )
            {
              for(int j1=i1_shifted-(Csize1-1); j1 <= i1_shifted; ++j1)
                for(int j2=i2_shifted-(Csize2-1); j2 <= i2_shifted; ++j2)
                  A(i1,i2) += B( (((j1%Bsize1)+Bsize1) % Bsize1) + Bl1, 
                               (((j2%Bsize2)+Bsize2) % Bsize2) + Bl2) * 
                    C( i1_shifted-j1 + Cl1, i2_shifted-j2 + Cl2);
            }
            else if( border_opt == Convolution::Mirror )
            {
              for(int j1=i1_shifted-(Csize1-1); j1 <= i1_shifted; ++j1)
                for(int j2=i2_shifted-(Csize2-1); j2 <= i2_shifted; ++j2)
                  A(i1,i2) += B( tc::mirrorInRange(j1,0,Bsize1-1) + Bl1, 
                               tc::mirrorInRange(j2,0,Bsize2-1) + Bl2) *
                    C( tc::mirrorInRange(i1_shifted-j1,0,Csize1-1) + Cl1,
                       tc::mirrorInRange(i2_shifted-j2,0,Csize2-1) + Cl2);
            }
          }
          else if( size_opt == Convolution::Valid )
          {
            // Interpolation is useless in this case
            blitz::Range jb1( i1, i1 + Csize1 - 1), jb2( i2, i2 + Csize2 - 1);
            blitz::Range jc1( Csize1 - 1, 0,-1), jc2( Csize2 - 1, 0,-1);
            A(i1,i2) = blitz::sum(B(jb1 + Bl1, jb2 + Bl2) * 
                         C(jc1 + Cl1, jc2 + Cl2) );
          }
          else
            throw Torch::core::Exception();
        }
      }
    }

  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_CONVOLVE_H */
