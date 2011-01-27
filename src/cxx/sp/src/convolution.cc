/**
 * @file src/cxx/sp/src/convolve.cc
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implement a blitz-based convolution product with zero padding
 */

#ifndef TORCH5SPRO_SP_CONVOLVE_CC
#define TORCH5SPRO_SP_CONVOLVE_CC

namespace Torch {
  /**
   * \ingroup libsp_api
   * @{
   *
   */
  namespace sp {

    template<typename T>
      blitz::Array<T,1> convolve(const blitz::Array<T,1>& B, 
        const blitz::Array<T,1>& C, size_t option = 0)
      {
        int Bl = B.lbound(0), Bh = B.ubound(0);
        int Cl = C.lbound(0), Ch = C.ubound(0);
        int Bsize = Bh - Bl + 1;
        int Csize = Ch - Cl + 1;

        int lbound = 0;
        int ubound;

        // Size of "B + C - 1"
        if( option == 0 )
          ubound = Bsize + Csize - 2;
        // Same size as B
        else if( option == 1 )
          ubound = Bsize - 1;
        // Size when not allowing any padding
        else if( option == 2 )
        {
          ubound = Bsize - Csize;
          // Check that B is larger than C, otherwise, throw an exception
          if( ubound <= 0 )
            throw Torch::core::Exception();
        }

        blitz::Array<T,1> A(blitz::Range(lbound,ubound));

        T result;
        for(int i=lbound; i <= ubound; ++i)
        {
          int jl;
          int jh;
          int i_shifted = i;

          if( option == 0 )
          {
            jl = ( i - (Csize-1) > 0 ? i - (Csize-1) : 0 );
            jh = ( i < Bsize ? i : Bsize-1 ); 
          }
          else if( option == 1 )
          {
            i_shifted += Csize / 2;

            jl = ( i_shifted - (Csize-1) > 0 ? i_shifted - (Csize-1) : 0 );
            jh = ( i_shifted < Bsize ? i_shifted : Bsize-1 ); 
          }
          else if( option == 2 )
          {
            i_shifted += Csize - 1;

            jl = ( i_shifted - (Csize-1) > 0 ? i_shifted - (Csize-1) : 0 );
            jh = ( i_shifted < Bsize ? i_shifted : Bsize-1 ); 
          }

          result = 0;
          for(int j=jl; j <= jh; ++j)
            result += B(j + Bl) * C(i_shifted-j + Cl);

          A(i) = result;
        }

        return A;
      }


    template<typename T>
      blitz::Array<T,2> convolve(const blitz::Array<T,2>& B, 
        const blitz::Array<T,2>& C, size_t option = 0)
      {
        int Bl1 = B.lbound(0), Bh1 = B.ubound(0);
        int Bl2 = B.lbound(1), Bh2 = B.ubound(1);
        int Cl1 = C.lbound(0), Ch1 = C.ubound(0);
        int Cl2 = C.lbound(1), Ch2 = C.ubound(1);
        int Bsize1 = Bh1 - Bl1 + 1;
        int Bsize2 = Bh2 - Bl2 + 1;
        int Csize1 = Ch1 - Cl1 + 1;
        int Csize2 = Ch2 - Cl2 + 1;

        int lbound1 = 0;
        int lbound2 = 0;
        int ubound1;
        int ubound2;

        if( option == 0 )
        {
          ubound1 = Bsize1 + Csize1 - 2;
          ubound2 = Bsize2 + Csize2 - 2;
        }
        else if( option == 1 )
        {
          ubound1 = Bsize1 - 1;
          ubound2 = Bsize2 - 1;
        }
        else if( option == 2 )
        {
          ubound1 = Bsize1 - Csize1;
          ubound2 = Bsize2 - Csize2;
          // Check that B is larger than C, otherwise, throw an exception
          if( ubound1 <= 0 || ubound2 <= 0)
            throw Torch::core::Exception();
        }

        blitz::Array<T,2> A(blitz::Range(lbound1,ubound1), 
          blitz::Range(lbound2,ubound2));

        T result;
        for(int i1=lbound1; i1 <= ubound1; ++i1)
        {
          int jl1;
          int jh1;

          int i1_shifted = i1;

          if( option == 0 )
          {
            jl1 = ( i1 - (Csize1-1) > 0 ? i1 - (Csize1-1) : 0 );
            jh1 = ( i1 < Bsize1 ? i1 : Bsize1-1 ); 
          }
          else if( option == 1 )
          {
            i1_shifted += Csize1 / 2;

            jl1 = ( i1_shifted-(Csize1-1) > 0 ? i1_shifted-(Csize1-1) : 0);
            jh1 = ( i1_shifted < Bsize1 ? i1_shifted : Bsize1-1 );
          }
          else if( option == 2 )
          {
            i1_shifted += Csize1 - 1;

            jl1 = ( i1_shifted-(Csize1-1) > 0 ? i1_shifted-(Csize1-1) : 0);
            jh1 = ( i1_shifted < Bsize1 ? i1_shifted : Bsize1-1 ); 
          }

          for(int i2=lbound2; i2 <= ubound2; ++i2)
          {
            int jl2;
            int jh2;
            int i2_shifted = i2;

            if( option == 0 )
            {
              jl2 = ( i2 - (Csize2-1) > 0 ? i2 - (Csize2-1) : 0 );
              jh2 = ( i2 < Bsize2 ? i2 : Bsize2-1 );
            }
            else if( option == 1 )
            {
              i2_shifted += Csize2 / 2;

              jl2 = ( i2_shifted-(Csize2-1) > 0 ? i2_shifted-(Csize2-1) : 0);
              jh2 = ( i2_shifted < Bsize2 ? i2_shifted : Bsize2-1 );
            }
            else if( option == 2 )
            {
              i2_shifted += Csize2 - 1;

              jl2 = ( i2_shifted-(Csize2-1) > 0 ? i2_shifted-(Csize2-1) : 0);
              jh2 = ( i2_shifted < Bsize2 ? i2_shifted : Bsize2-1 );
            }

            result = 0;
            for(int j1=jl1; j1 <= jh1; ++j1)
              for(int j2=jl2; j2 <= jh2; ++j2)
                result += B(j1+Bl1,j2+Bl2) * 
                  C(i1_shifted-j1 + Cl1, i2_shifted-j2 +Cl2);

            A(i1,i2) = result;
          }
        }

        return A;
      }

  }
}


#endif /* TORCH5SPRO_SP_CONVOLVE_CC */

