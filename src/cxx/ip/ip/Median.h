/**
 * @file cxx/ip/ip/Median.h
 * @date Wed Sep 28 13:34:10 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file provides a class to filter an image with a median filter
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

#ifndef BOB_IP_MEDIAN_H
#define BOB_IP_MEDIAN_H

#include <boost/shared_ptr.hpp>
#include <list>
#include "core/array_assert.h"
#include "core/cast.h"
#include "sp/convolution.h"
#include "ip/Exception.h"

namespace bob {

	/**
	 * \ingroup libip_api
	 * @{
	 *
	 */
	namespace ip {

    namespace detail {
      template <typename T>
      struct Pixel {
        int y;
        int x;
        T value;
        Pixel(): y(0), x(0), value(0) {}
        Pixel(int b, int a, T v): y(b), x(a), value(v) {}
      };

      template <typename T>
      void listInsertPixel(const boost::shared_ptr<Pixel<T> > p, 
                           std::list<boost::shared_ptr<Pixel<T> > >& l)
      {
        typename std::list<boost::shared_ptr<Pixel<T> > >::iterator it=l.begin();
        for( ; it!=l.end(); ++it)
          if( p->value > (*it)->value) break;
        l.insert(it, p);
      }

      template <typename T>
      T listGetValueAtPosition(const int pos, 
        const std::list<boost::shared_ptr<Pixel<T> > >& l)
      { 
        typename std::list<boost::shared_ptr<Pixel<T> > >::const_iterator it=l.begin();
        int c=0;
        for( ; it!=l.end(); ++it, ++c)
          if( c==pos) return (*it)->value;
        throw bob::ip::Exception();
      }
    
      template <typename T>
      void printList(const std::list<boost::shared_ptr<Pixel<T> > >& l)
      {
        typename std::list<boost::shared_ptr<Pixel<T> > >::const_iterator it=l.begin();
        for( ; it!=l.end(); ++it)
        {
          std::cout << "(" << (*it)->y << "," << (*it)->x << "," << (*it)->value << ") - ";
        }
        std::cout << std::endl;
      }
    }

    /**
      * @brief This class allows to filter an image with a median filter
      */
    template <typename T> 
		class Median
		{
  		public:
			  /**
  			 * @brief Creates an object to filter images with a median filter
	  		 * @param radius_y The height of the kernel along the y-axis
	  		 * @param radius_x The width of the kernel along the x-axis
			   */
	  		Median(const int radius_y=1, const int radius_x=1): 
          m_radius_y(radius_y), m_radius_x(radius_x),
          m_median_pos((2*radius_y+1)*(2*radius_x+1)/2)
  			{
        }

        virtual ~Median() 
        {
        }

        /**
          * @brief Resets the filter with different radius
          */
        void reset( const int radius_y=1, const int radius_x=1)
        {
          m_radius_y = radius_y;
          m_radius_x = radius_x;
          m_median_pos = (2*radius_y+1)*(2*radius_x+1)/2;
        }

        /**
         * @brief Processes a 2D blitz Array/Image
         * @param src The 2D input blitz array
         * @param dst The 2D input blitz array
         */
        void operator()(const blitz::Array<T,2>& src, 
          blitz::Array<T,2>& dst);

        /**
         * @brief Processes a 3D blitz Array/Image
         * @param src The 3D input blitz array
         * @param dst The 3D input blitz array
         */
        void operator()(const blitz::Array<T,3>& src, 
          blitz::Array<T,3>& dst);


      private:
        /**
          * @brief Initializes the ordered lists of values
          */
        void initLists(const blitz::Array<T,2>& src);
        /**
          * @brief Efficiently updates the ordered lists
          */
        void listRemoveAddColumn(const int j, const int i, 
                const blitz::Array<T,2>& src, 
                std::list<boost::shared_ptr<struct detail::Pixel<T> > >& l);
        void listRemoveAddRow(const int j, const int i, 
                const blitz::Array<T,2>& src, 
                std::list<boost::shared_ptr<struct detail::Pixel<T> > >& l);

        /**
         * @brief Attributes
         */	
        int m_radius_y;
        int m_radius_x;
        int m_median_pos;

        std::list<boost::shared_ptr<struct detail::Pixel<T> > > m_list_current;
        std::list<boost::shared_ptr<struct detail::Pixel<T> > > m_list_first_col;
    };

    template <typename T>
    void bob::ip::Median<T>::initLists(const blitz::Array<T,2>& src)
    {
      // Clears content
      m_list_first_col.clear();
      m_list_current.clear();
      typedef struct bob::ip::detail::Pixel<T> Pix;
      typedef typename std::list<boost::shared_ptr<Pix> >::iterator ListIterator;
      for(int j=0; j<2*m_radius_y+1; ++j)
        for(int i=0; i<2*m_radius_x+1; ++i)
        {
          boost::shared_ptr<Pix> pix(new Pix(j,i,src(j,i)));
          bob::ip::detail::listInsertPixel(pix, m_list_first_col);
        }
      m_list_current = m_list_first_col;
    }


    template <typename T>
    void bob::ip::Median<T>::listRemoveAddColumn(const int j, const int i,
                  const blitz::Array<T,2>& src,
                  std::list<boost::shared_ptr<struct detail::Pixel<T> > >& l)
    {
      // Erases elements from column i
      typename std::list<boost::shared_ptr<struct detail::Pixel<T> > >::iterator it=l.begin();
      for( ; it!=l.end(); )
      {
        if( (*it)->x == i) it = l.erase(it);
        else ++it;
      }
      // Adds elements from new column i+2*m_radius_x+1
      blitz::Array<T,1> col_new = src(blitz::Range(j,j+2*m_radius_y), i+2*m_radius_x+1);
      for(int k=0; k<col_new.extent(0); ++k)
      {
        boost::shared_ptr<struct detail::Pixel<T> > pix(
          new struct detail::Pixel<T>(j+k,i+2*m_radius_x+1,src(j+k,i+2*m_radius_x+1)));
        bob::ip::detail::listInsertPixel(pix, l);
      }
    }

    template <typename T>
    void bob::ip::Median<T>::listRemoveAddRow(const int j, const int i,
                  const blitz::Array<T,2>& src,
                  std::list<boost::shared_ptr<struct detail::Pixel<T> > >& l)
    {
      // Erases elements from row j 
      typename std::list<boost::shared_ptr<struct detail::Pixel<T> > >::iterator it=l.begin();
      for( ; it!=l.end(); )
      {
        if( (*it)->y == j) it = l.erase(it);
        else ++it;
      }
      // Adds elements from new row j+2*m_radius_y+2
      blitz::Array<T,1> col_new = src(j+2*m_radius_y+1,blitz::Range(i,i+2*m_radius_x));
      for(int k=0; k<col_new.extent(0); ++k)
      {
        boost::shared_ptr<struct detail::Pixel<T> > pix(
          new struct detail::Pixel<T>(j+2*m_radius_y+1,i+k,src(j+2*m_radius_y+1,i+k)));
        bob::ip::detail::listInsertPixel(pix, l);
      }
    }

    template <typename T> 
    void bob::ip::Median<T>::operator()(const blitz::Array<T,2>& src, 
      blitz::Array<T,2>& dst)
    {
      // Checks
      bob::core::array::assertZeroBase(src);
      bob::core::array::assertZeroBase(dst);
      blitz::TinyVector<int,2> dst_size;
      dst_size(0) = src.extent(0) - 2 * m_radius_y;
      dst_size(1) = src.extent(1) - 2 * m_radius_x;
      bob::core::array::assertSameShape(dst, dst_size);

      // Initializes the lists
      initLists(src);

      // Filters
      for(int j=0; j<dst.extent(0); ++j)
      {
        for(int i=0; i<dst.extent(1); ++i)
        {
          dst(j,i) = bob::ip::detail::listGetValueAtPosition( m_median_pos, m_list_current);
          // Updates current ordered list
          if( i<dst.extent(1) - 1) {
            listRemoveAddColumn(j, i, src, m_list_current);
          }
        }
        // Updates current ordered list
        if(j<dst.extent(0)-1)
        {
          listRemoveAddRow(j, 0, src, m_list_first_col);
          m_list_current = m_list_first_col;
        }
      }
    }

    template <typename T> 
    void bob::ip::Median<T>::operator()(const blitz::Array<T,3>& src, 
      blitz::Array<T,3>& dst)
    {
      for( int p=0; p<dst.extent(0); ++p) {
        const blitz::Array<T,2> src_slice = 
          src( p, blitz::Range::all(), blitz::Range::all() );
        blitz::Array<T,2> dst_slice = 
          dst( p, blitz::Range::all(), blitz::Range::all() );
        
        // Apply median filter to the plane
        operator()(src_slice, dst_slice);
      }
    }

	}
}

#endif
