/**
 * @file src/cxx/sp/sp/extrapolate.h
 * @author <a href="mailto:Laurent.El-Shafey@idiap.ch">Laurent El Shafey</a>
 *
 * @brief Implements various extrapolation techniques for 1D and 2D blitz
 * arrays.
 */

#ifndef TORCH5SPRO_SP_EXTRAPOLATE_H
#define TORCH5SPRO_SP_EXTRAPOLATE_H

#include "core/array_type.h"
#include "core/Exception.h"

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
    namespace Extrapolation {
      enum AlgorithmOption {
        Zero,
        NearestNeighbour,
        Circular,
        Mirror
      };
    }

    namespace detail {
      template<typename T> void extrapolate(const blitz::Array<T,1>& src, 
        blitz::Array<T,1>& dst, const int border_beg, const int border_end,
        const enum Extrapolation::AlgorithmOption algo_opt=Extrapolation::Zero)
      {
        // Define constants
        const int src_length = src.extent(0);
        const int dst_length = src_length+border_beg+border_end;

        // Copy initial blitz array 
        blitz::Range r_src(0,src_length-1);
        blitz::Range r_dst_interior(border_beg, border_beg+src_length-1);
        dst(r_dst_interior) = src(r_src);

        // Add the border
        blitz::Range r_dst_beg(0,border_beg-1);
        blitz::Range r_dst_end(dst_length-border_end,dst_length-1);
        switch(algo_opt)
        {
          case Extrapolation::Zero:
            dst(r_dst_beg) = 0;
            dst(r_dst_end) = 0;
            break;
          case Extrapolation::NearestNeighbour:
            dst(r_dst_beg) = src(src.lbound(0));
            dst(r_dst_end) = src(src.ubound(0));
            break;
          case Extrapolation::Circular:
            {
              // Extrapolate "border_beg"
              int current_beg_last = border_beg-1;
              int step_length;
              while(current_beg_last>=0)
              {
                if(current_beg_last<=src.extent(0))
                  step_length = current_beg_last+1;
                else
                  step_length = src.extent(0);
                blitz::Range r_src_begIt(src.ubound(0)-step_length+1,
                              src.ubound(0));
                blitz::Range r_dst_begIt(current_beg_last+1-step_length,
                              current_beg_last);
                dst(r_dst_begIt) = src(r_src_begIt);
                current_beg_last-=step_length;
              } 
   
              // Extrapolate "border_end"
              int current_end_first = dst_length-border_end;
              while(current_end_first<dst_length)
              {
                if(dst_length-current_end_first<=src.extent(0))
                  step_length = dst_length-current_end_first;
                else
                  step_length = src.extent(0);
                blitz::Range r_src_endIt(0,step_length-1);
                blitz::Range r_dst_endIt(current_end_first,
                              current_end_first+step_length-1);
                dst(r_dst_endIt) = src(r_src_endIt);
                current_end_first+=step_length;
              }  
            }
            break;
          case Extrapolation::Mirror: //TODO
            {
              // Extrapolate "border_beg"
              int current_beg_last = border_beg-1;
              int step_length;
              bool reverse_direction=true;
              while(current_beg_last>=0)
              {
                if(current_beg_last<=src.extent(0))
                  step_length = current_beg_last+1;
                else
                  step_length = src.extent(0);
              
                blitz::Range r_src_begIt;
                if(reverse_direction)
                  r_src_begIt.setRange(src.lbound(0)+step_length-1,
                                src.lbound(0),-1);
                else
                  r_src_begIt.setRange(src.ubound(0)-step_length+1,
                                src.ubound(0));
                blitz::Range r_dst_begIt(current_beg_last+1-step_length,
                              current_beg_last);
                dst(r_dst_begIt) = src(r_src_begIt);
                current_beg_last-=step_length;
                reverse_direction=!reverse_direction; 
              } 
   
              // Extrapolate "border_end"
              int current_end_first = dst_length-border_end;
              reverse_direction=true;
              while(current_end_first<dst_length)
              {
                if(dst_length-current_end_first<=src.extent(0))
                  step_length = dst_length-current_end_first;
                else
                  step_length = src.extent(0);

                blitz::Range r_src_endIt;
                if(reverse_direction)
                  r_src_endIt.setRange(src.ubound(0),
                                src.ubound(0)-step_length+1,-1);
                else
                  r_src_endIt.setRange(src.lbound(0),
                                src.lbound(0)+step_length-1);
                blitz::Range r_dst_endIt(current_end_first,
                              current_end_first+step_length-1);
                dst(r_dst_endIt) = src(r_src_endIt);
                current_end_first+=step_length;
                reverse_direction=!reverse_direction; 
              }  
            }
            break;
          default:
            throw Torch::core::Exception();
        }
      }

    }


    /**
     * @brief Extrapolate a 1D blitz Array
     * @param src The input 1D blitz array to extrapolate
     * @param dst The output 1D blitz array
     * @param border_beg The size of the border to add in front of the initial
     *    1D blitz array.
     * @param border_end The size of the border to add after the initial
     *    1D blitz array.
     * @param algo The extrapolation technique to use
     *                     * Zero: zero padding
     *                     * Nearest Neighbour: extrapolate with nearest 
     *                         neighbour
     *                     * Circular: extrapolate by considering tiled array
     *                         for src (<-> modulo arrays)
     *                     * Mirror: extrapolate by mirroring the src array
     * @warning The dst array has to be of the correct size.
     *   The src and dst should have zero base indices.
     */
    template<typename T> void extrapolate(const blitz::Array<T,1>& src, 
      blitz::Array<T,1>& dst, const int border_beg, const int border_end,
      const enum Extrapolation::AlgorithmOption algo_opt = Extrapolation::Zero)
    {
      // Check input
      Torch::core::assertZeroBase(src);

      // Check border (positivity)
/*      if( border_beg<0 )
        throw ParamOutOfBoundaryError("border_beg", false, border_beg, 0);
      else if( border_end<0)
        throw ParamOutOfBoundaryError("border_end", false, border_end, 0);
*/
      // Check output
      Torch::core::assertZeroBase(dst);
      const int src_length = src.extent(0);
      const int dst_length = src_length+border_beg+border_end;
      const blitz::TinyVector<int,1> shape(dst_length);
      Torch::core::assertSameShape(dst,shape);

      // call the extrapolation function
      detail::extrapolate(src, dst, border_beg, border_end, algo_opt);
    }
  }
/**
 * @}
 */
}

#endif /* TORCH5SPRO_SP_EXTRAPOLATE_H */
