/**
 * @file cxx/ip/src/Convert.cc
 * @date Tue Jan 18 17:07:26 2011 +0100
 * @author André Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
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
#include "ip/Convert.h"

namespace bob {

unsigned char rgb_to_gray(unsigned char r, unsigned char g, unsigned char b)
{
#ifdef CONVERT_FAST_RGB2GRAY_
	unsigned char gray = ((times77[r] + times150[g] + times29[b]) >> 8);

	return gray;
#else
   	float gray = 0.299f * (float) r + 0.587f * (float) g + 0.114f * (float) b;

  //Please note we have to round the "gray" result to avoid cases like
  //gray = 109.99999997 to be converted into 109 by this cast. We add 0.5 and
  //only then we cast.
	return (unsigned char) (gray+0.5f);
#endif	
}

void yuv_to_rgb(unsigned char y, unsigned char u, unsigned char v, unsigned char *r, unsigned char *g, unsigned char *b)
{
	int Y, U, V;

	Y = ICCIRY(y);
	U = ICCIRUV(u - 128);
	V = ICCIRUV(v - 128);

	*r = CLIP(GET_R_FROM_YUV(Y, U, V));
	*g = CLIP(GET_G_FROM_YUV(Y, U, V));
	*b = CLIP(GET_B_FROM_YUV(Y, U, V));
}

void rgb24_to_rgb24(unsigned char *data_in, unsigned char *data_out, unsigned int width, unsigned int height)
{
   	unsigned char r, g, b;
   	int size;
   	unsigned char *src;
	unsigned char *d;
   	int i;	

   	size = width * height;
	src = data_in;
	d = data_out;
	
	for(i = 0 ; i < size ; i++)
	{	 
	   	r = src[0];
	   	g = src[1];
	   	b = src[2];

		*d++ = r;
		*d++ = g;
		*d++ = b;

		src += 3;
	}	
}

void swap_(unsigned char *a, unsigned char *b)
{
        unsigned char tmp;

        tmp = *a;
        *a = *b;
        *b = tmp;
}

void rgb24_to_bgr24_flip(unsigned char *data_in, unsigned char *data_out, unsigned int width, unsigned int height)
{
	if(data_out == NULL)
	{
		// convert and swap in the same pixmap

		for(unsigned int h = 0; h < height/2 ; h++)
                	for(unsigned int w = 0; w < width ; w++)
	                {
        	                int base_in = w + h * width;
                	        int base_out = w + (height - h - 1) * width;

	                        base_in *= 3;
        	                base_out *= 3;

	                        swap_(&data_in[base_out], &data_in[base_in+2]);
        	                swap_(&data_in[base_out+1], &data_in[base_in+1]);
	                        swap_(&data_in[base_out+2], &data_in[base_in]);
        	        }
	}
	else
	{
		for(unsigned int h = 0; h < height ; h++)
                	for(unsigned int w = 0; w < width ; w++)
	                {
        	                int base_in = w + h * width;
                	        int base_out = w + (height - h - 1) * width;

	                        base_in *= 3;
        	                base_out *= 3;

	                        *(data_out+base_out) = *(data_in+base_in+2);
        	                *(data_out+base_out+1) = *(data_in+base_in+1);
                	        *(data_out+base_out+2) = *(data_in+base_in);
			}	
	}
}

void yuv_to_rgb24(unsigned char *data_in, unsigned char *data_out, unsigned int width, unsigned int height)
{
   	unsigned char y, u, v;
   	int size;
   	unsigned char *src;
   	unsigned char *d;
   	int i;	

   	size = height * width;
	src = data_in;
	d = data_out;
	
	for(i = 0 ; i < size ; i++)
	{	
	   	y = src[0];
	   	u = src[1];
		v = src[2];
		
		*d++ = y;
		*d++ = u;
		*d++ = v;

		src += 3;
	}	
}

void yuv422_to_rgb24(unsigned char *data_in, unsigned char *data_out, unsigned int width, unsigned int height)
{
   	unsigned char y1, y2, u, v;
   	int size;
   	unsigned char *src;
   	unsigned char *d;
   	int i;	

   	size = height * width / 2;
	src = data_in;
	d = data_out;
	
	for(i = 0 ; i < size ; i++)
	{
	   	y1 = src[0];
	   	u = src[1];
	   	y2 = src[2];
		v = src[3];
		
		*d++ = y1;
		*d++ = u;
		*d++ = y2;
		*d++ = v;
		
		src += 4;
	}	
}


/// Y Cb Cr -> RGB conversion
int y_cb_cr_offset[3] = { 16, 128, 128 };

static float ycc2rgb219[3][3]={
 { 1.0 ,  0.   ,  1.371 },
 { 1.0 , -0.336, -0.698 },
 { 1.0 ,  1.732,  0. }};

unsigned char limitx_(double x) 
{
	int y = (int) ((x<0) ? (x-0.5) : (x+0.5));
	if(y>255) return 255;
	if(y<0) return 0;
	return y;
}

void ycbcr2rgb(int y, int cb, int cr, unsigned char *rgb)
{
	y  -= y_cb_cr_offset[0];
	cb -= y_cb_cr_offset[1];
	cr -= y_cb_cr_offset[2]; 
 
    	rgb[0] = limitx_(ycc2rgb219[0][0]*y + ycc2rgb219[0][1]*cb + ycc2rgb219[0][2]*cr);
    	rgb[1] = limitx_(ycc2rgb219[1][0]*y + ycc2rgb219[1][1]*cb + ycc2rgb219[1][2]*cr);
    	rgb[2] = limitx_(ycc2rgb219[2][0]*y + ycc2rgb219[2][1]*cb + ycc2rgb219[2][2]*cr);
}

//#define VERBOSE_CONVERT

bool YUV422P_to_YUV(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	int i;
	unsigned char *p_dest;
	unsigned char y1, u, y2, v;

#ifdef VERBOSE_CONVERT
	print("YUV422_to_YUV\n");
    	print("   w   = %d\n", width);
    	print("   h   = %d\n", height);
 	print("   bpl src = %d\n", bytesperline_src);
    	print("   bpl dst = %d\n", bytesperline_dst);
#endif

	p_dest = d;

	int size = height * (width / 2);
	
  	for(i = 0 ; i < size ; i++)
	{
	   	y1 = *s++;
		u  = *s++;
	   	y2 = *s++;
		v  = *s++;

		p_dest[0] = y1;
		p_dest[1] = u;
		p_dest[2] = v;

		p_dest += 3;
	   	
		p_dest[0] = y2;
		p_dest[1] = u;
		p_dest[2] = v;

		p_dest += 3;
	}
	
	return true;
}


bool YUV422P_to_RGB24(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	int i;
	unsigned char *p_dest;
	unsigned char y1, u, y2, v;
	int Y1, Y2, U, V;
	unsigned char r, g, b;

#ifdef VERBOSE_CONVERT
	print("YUV422_to_RGB24\n");
	print("   w   = %d\n", width);
    	print("   h   = %d\n", height);
	print("   bpl src = %d\n", bytesperline_src);
    	print("   bpl dst = %d\n", bytesperline_dst);
#endif

	p_dest = d;

	int size = height * (width / 2);
	
  	for(i = 0 ; i < size ; i++)
	{
	   	y1 = *s++;
		u  = *s++;
	   	y2 = *s++;
		v  = *s++;

		Y1 = ICCIRY(y1);
	   	U = ICCIRUV(u - 128);
	 	Y2 = ICCIRY(y2);
	   	V = ICCIRUV(v - 128);

		r = CLIP(GET_R_FROM_YUV(Y1, U, V));
		g = CLIP(GET_G_FROM_YUV(Y1, U, V));
		b = CLIP(GET_B_FROM_YUV(Y1, U, V));

		p_dest[0] = r;
		p_dest[1] = g;
		p_dest[2] = b;

		p_dest += 3;
	   	
		r = CLIP(GET_R_FROM_YUV(Y2, U, V));
		g = CLIP(GET_G_FROM_YUV(Y2, U, V));
		b = CLIP(GET_B_FROM_YUV(Y2, U, V));

		p_dest[0] = r;
		p_dest[1] = g;
		p_dest[2] = b;

		p_dest += 3;
	}
	
	return true;
}

bool YUV420P_to_GREY(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	unsigned char *y, *u, *v, *d_;
    	unsigned char *us,*vs;
    	unsigned char *dp;
    	int i,j;

#ifdef VERBOSE_CONVERT
	print("YUV420P_to_GREY\n");
	print("   w   = %d\n", width);
    	print("   h   = %d\n", height);
	print("   bpl src = %d\n", bytesperline_src);
    	print("   bpl dst = %d\n", bytesperline_dst);
#endif

    	dp = d;
    	y  = s;
    	u  = y + width * height;
    	v  = u + width * height / 4;
    
    	for (i = 0; i < height; i++) 
	{
		d_ = dp;
		us = u; vs = v;

		unsigned char yy;	

		for (j = 0; j < width; j+= 2) 
		{
			yy = *y;

			//Y = LUT_ICCIRY[yy];
			*(d_++) = yy;

	    		y++;

			yy = *y;
			*(d_++) = yy;

	    		y++; u++; v++;
		}

		if (0 == (i % 2)) 
		{
	    		u = us; v = vs;
		}
	
		dp += bytesperline_dst;
	}
	
	return true;
}


bool YUV420P_to_YUV(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	unsigned char *y, *u, *v, *d_;
    	unsigned char *us,*vs;
    	unsigned char *dp;
    	int i,j;

#ifdef VERBOSE_CONVERT
	print("YUV420P_to_YUV\n");
	print("   w   = %d\n", width);
    	print("   h   = %d\n", height);
    	print("   bpl src = %d\n", bytesperline_src);
   	print("   bpl dst = %d\n", bytesperline_dst);
#endif

    	dp = d;
    	y  = s;
    	u  = y + width * height;
    	v  = u + width * height / 4;
    
    	for (i = 0; i < height; i++) 
	{
		d_ = dp;
		us = u; vs = v;

		unsigned char yy, uu, vv;	

		for (j = 0; j < width; j+= 2) 
		{
			yy = *y;
			uu = *u;
			vv = *v;

			*(d_++) = yy;
			*(d_++) = uu;
			*(d_++) = vv;

	    		y++;

			yy = *y;

			*(d_++) = yy;
			*(d_++) = uu;
			*(d_++) = vv;

	    		y++; u++; v++;
		}

		if (0 == (i % 2)) 
		{
	    		u = us; v = vs;
		}
	
		dp += bytesperline_dst;
	}
	
	return true;
}


bool YUV420P_to_RGB24(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	unsigned char *y, *u, *v, *d_;
    	unsigned char *us,*vs;
    	unsigned char *dp;
    	int i,j;

#ifdef VERBOSE_CONVERT
	print("YUV420P_to_RGB24\n");
	print("   w   = %d\n", width);
    	print("   h   = %d\n", height);
    	print("   bpl src = %d\n", bytesperline_src);
   	print("   bpl dst = %d\n", bytesperline_dst);
#endif

    	dp = d;
    	y  = s;
    	u  = y + width * height;
    	v  = u + width * height / 4;
    
    	for (i = 0; i < height; i++) 
	{
		d_ = dp;
		us = u; vs = v;

		unsigned char yy, uu, vv;	
		int Y, U, V;

		for (j = 0; j < width; j+= 2) 
		{
			yy = *y;
			uu = *u;
			vv = *v;

#ifdef FAST_ICCIRYUV
#ifdef FAST_ICCIRYUV_RGB
			*(d_++) = LUT_ICCIRYUV_R[yy][uu][vv];
			*(d_++) = LUT_ICCIRYUV_G[yy][uu][vv];
			*(d_++) = LUT_ICCIRYUV_B[yy][uu][vv];
#else
			Y = LUT_ICCIRY[yy];
	   		U = LUT_ICCIRUV_128[uu];
	   		V = LUT_ICCIRUV_128[vv];

			*(d_++) = CLIP(GET_R_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_G_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_B_FROM_YUV(Y, U, V));
#endif		
#else
			Y = ICCIRY(yy);
	   		U = ICCIRUV(uu - 128);
	   		V = ICCIRUV(vv - 128);

			*(d_++) = CLIP(GET_R_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_G_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_B_FROM_YUV(Y, U, V));
#endif

	    		y++;

			yy = *y;
#ifdef FAST_ICCIRYUV
#ifdef FAST_ICCIRYUV_RGB
			*(d_++) = LUT_ICCIRYUV_R[yy][uu][vv];
			*(d_++) = LUT_ICCIRYUV_G[yy][uu][vv];
			*(d_++) = LUT_ICCIRYUV_B[yy][uu][vv];
#else
			Y = LUT_ICCIRY[yy];

			*(d_++) = CLIP(GET_R_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_G_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_B_FROM_YUV(Y, U, V));
#endif
#else
			Y = ICCIRY(yy);

			*(d_++) = CLIP(GET_R_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_G_FROM_YUV(Y, U, V));
			*(d_++) = CLIP(GET_B_FROM_YUV(Y, U, V));
#endif

	    		y++; u++; v++;
		}

		if (0 == (i % 2)) 
		{
	    		u = us; v = vs;
		}
	
		dp += bytesperline_dst;
	}
	
	return true;
}


bool RGB24_to_RGB24(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	int i;
	unsigned char *p_dest;

#ifdef VERBOSE_CONVERT
	print("RGB24_to_RGB24\n");
    	print("   w   = %d\n", width);
   	print("   h   = %d\n", height);
    	print("   bpl src = %d\n", bytesperline_src);
 	print("   bpl dst = %d\n", bytesperline_dst);
#endif

	p_dest = d;
	
	int size = width * height;

  	for(i = 0 ; i < size ; i++)
	{
		p_dest[0] = *s++;
		p_dest[1] = *s++;
		p_dest[2] = *s++;

		p_dest += 3;
	}
	
	return true;
}

bool RGB32_to_RGB24(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	int i;
	unsigned char *p_src;
	unsigned char *p_dest;

#ifdef VERBOSE_CONVERT
	print("RGB32_to_RGB24\n");
    	print("   w   = %d\n", width);
   	print("   h   = %d\n", height);
    	print("   bpl src = %d\n", bytesperline_src);
 	print("   bpl dst = %d\n", bytesperline_dst);
#endif

	p_src = s;
	p_dest = d;
	
	int size = width * height;

  	for(i = 0 ; i < size ; i++)
	{
		p_dest[0] = p_src[0];
		p_dest[1] = p_src[1];
		p_dest[2] = p_src[2];

		p_src += 4;
		p_dest += 3;
	}
	
	return true;
}


bool RGB32_to_YUV(int width, int height, unsigned char *s, int bytesperline_src, unsigned char *d, int bytesperline_dst)
{
	int i;
	unsigned char *p_src;
	unsigned char *p_dest;

#ifdef VERBOSE_CONVERT
	print("RGB32_to_YUV\n");
    	print("   w   = %d\n", width);
   	print("   h   = %d\n", height);
    	print("   bpl src = %d\n", bytesperline_src);
 	print("   bpl dst = %d\n", bytesperline_dst);
#endif

	p_src = s;
	p_dest = d;
	
	int size = width * height;

  	for(i = 0 ; i < size ; i++)
	{
		p_dest[0] = CLIP(GET_Y_FROM_RGB(p_src[0], p_src[1], p_src[2]));
		p_dest[1] = CLIP(GET_U_FROM_RGB(p_src[0], p_src[1], p_src[2]));
		p_dest[2] = CLIP(GET_V_FROM_RGB(p_src[0], p_src[1], p_src[2]));

		p_src += 3;
		p_dest += 3;
	}
	
	return true;
}


/** Lookup tables for fast RGB -> luminance calculation.
*/
int times77[256] = {
  0,    77,   154,   231,   308,   385,   462,   539,
  616,   693,   770,   847,   924,  1001,  1078,  1155,
  1232,  1309,  1386,  1463,  1540,  1617,  1694,  1771,
  1848,  1925,  2002,  2079,  2156,  2233,  2310,  2387,
  2464,  2541,  2618,  2695,  2772,  2849,  2926,  3003,
  3080,  3157,  3234,  3311,  3388,  3465,  3542,  3619,
  3696,  3773,  3850,  3927,  4004,  4081,  4158,  4235,
  4312,  4389,  4466,  4543,  4620,  4697,  4774,  4851,
  4928,  5005,  5082,  5159,  5236,  5313,  5390,  5467,
  5544,  5621,  5698,  5775,  5852,  5929,  6006,  6083,
  6160,  6237,  6314,  6391,  6468,  6545,  6622,  6699,
  6776,  6853,  6930,  7007,  7084,  7161,  7238,  7315,
  7392,  7469,  7546,  7623,  7700,  7777,  7854,  7931,
  8008,  8085,  8162,  8239,  8316,  8393,  8470,  8547,
  8624,  8701,  8778,  8855,  8932,  9009,  9086,  9163,
  9240,  9317,  9394,  9471,  9548,  9625,  9702,  9779,
  9856,  9933, 10010, 10087, 10164, 10241, 10318, 10395,
  10472, 10549, 10626, 10703, 10780, 10857, 10934, 11011,
  11088, 11165, 11242, 11319, 11396, 11473, 11550, 11627,
  11704, 11781, 11858, 11935, 12012, 12089, 12166, 12243,
  12320, 12397, 12474, 12551, 12628, 12705, 12782, 12859,
  12936, 13013, 13090, 13167, 13244, 13321, 13398, 13475,
  13552, 13629, 13706, 13783, 13860, 13937, 14014, 14091,
  14168, 14245, 14322, 14399, 14476, 14553, 14630, 14707,
  14784, 14861, 14938, 15015, 15092, 15169, 15246, 15323,
  15400, 15477, 15554, 15631, 15708, 15785, 15862, 15939,
  16016, 16093, 16170, 16247, 16324, 16401, 16478, 16555,
  16632, 16709, 16786, 16863, 16940, 17017, 17094, 17171,
  17248, 17325, 17402, 17479, 17556, 17633, 17710, 17787,
  17864, 17941, 18018, 18095, 18172, 18249, 18326, 18403,
  18480, 18557, 18634, 18711, 18788, 18865, 18942, 19019,
  19096, 19173, 19250, 19327, 19404, 19481, 19558, 19635 };

int times150[256] = {
  0,   150,   300,   450,   600,   750,   900,  1050,
  1200,  1350,  1500,  1650,  1800,  1950,  2100,  2250,
  2400,  2550,  2700,  2850,  3000,  3150,  3300,  3450,
  3600,  3750,  3900,  4050,  4200,  4350,  4500,  4650,
  4800,  4950,  5100,  5250,  5400,  5550,  5700,  5850,
  6000,  6150,  6300,  6450,  6600,  6750,  6900,  7050,
  7200,  7350,  7500,  7650,  7800,  7950,  8100,  8250,
  8400,  8550,  8700,  8850,  9000,  9150,  9300,  9450,
  9600,  9750,  9900, 10050, 10200, 10350, 10500, 10650,
  10800, 10950, 11100, 11250, 11400, 11550, 11700, 11850,
  12000, 12150, 12300, 12450, 12600, 12750, 12900, 13050,
  13200, 13350, 13500, 13650, 13800, 13950, 14100, 14250,
  14400, 14550, 14700, 14850, 15000, 15150, 15300, 15450,
  15600, 15750, 15900, 16050, 16200, 16350, 16500, 16650,
  16800, 16950, 17100, 17250, 17400, 17550, 17700, 17850,
  18000, 18150, 18300, 18450, 18600, 18750, 18900, 19050,
  19200, 19350, 19500, 19650, 19800, 19950, 20100, 20250,
  20400, 20550, 20700, 20850, 21000, 21150, 21300, 21450,
  21600, 21750, 21900, 22050, 22200, 22350, 22500, 22650,
  22800, 22950, 23100, 23250, 23400, 23550, 23700, 23850,
  24000, 24150, 24300, 24450, 24600, 24750, 24900, 25050,
  25200, 25350, 25500, 25650, 25800, 25950, 26100, 26250,
  26400, 26550, 26700, 26850, 27000, 27150, 27300, 27450,
  27600, 27750, 27900, 28050, 28200, 28350, 28500, 28650,
  28800, 28950, 29100, 29250, 29400, 29550, 29700, 29850,
  30000, 30150, 30300, 30450, 30600, 30750, 30900, 31050,
  31200, 31350, 31500, 31650, 31800, 31950, 32100, 32250,
  32400, 32550, 32700, 32850, 33000, 33150, 33300, 33450,
  33600, 33750, 33900, 34050, 34200, 34350, 34500, 34650,
  34800, 34950, 35100, 35250, 35400, 35550, 35700, 35850,
  36000, 36150, 36300, 36450, 36600, 36750, 36900, 37050,
  37200, 37350, 37500, 37650, 37800, 37950, 38100, 38250 };

int times29[256] = {
  0,    29,    58,    87,   116,   145,   174,   203,
  232,   261,   290,   319,   348,   377,   406,   435,
  464,   493,   522,   551,   580,   609,   638,   667,
  696,   725,   754,   783,   812,   841,   870,   899,
  928,   957,   986,  1015,  1044,  1073,  1102,  1131,
  1160,  1189,  1218,  1247,  1276,  1305,  1334,  1363,
  1392,  1421,  1450,  1479,  1508,  1537,  1566,  1595,
  1624,  1653,  1682,  1711,  1740,  1769,  1798,  1827,
  1856,  1885,  1914,  1943,  1972,  2001,  2030,  2059,
  2088,  2117,  2146,  2175,  2204,  2233,  2262,  2291,
  2320,  2349,  2378,  2407,  2436,  2465,  2494,  2523,
  2552,  2581,  2610,  2639,  2668,  2697,  2726,  2755,
  2784,  2813,  2842,  2871,  2900,  2929,  2958,  2987,
  3016,  3045,  3074,  3103,  3132,  3161,  3190,  3219,
  3248,  3277,  3306,  3335,  3364,  3393,  3422,  3451,
  3480,  3509,  3538,  3567,  3596,  3625,  3654,  3683,
  3712,  3741,  3770,  3799,  3828,  3857,  3886,  3915,
  3944,  3973,  4002,  4031,  4060,  4089,  4118,  4147,
  4176,  4205,  4234,  4263,  4292,  4321,  4350,  4379,
  4408,  4437,  4466,  4495,  4524,  4553,  4582,  4611,
  4640,  4669,  4698,  4727,  4756,  4785,  4814,  4843,
  4872,  4901,  4930,  4959,  4988,  5017,  5046,  5075,
  5104,  5133,  5162,  5191,  5220,  5249,  5278,  5307,
  5336,  5365,  5394,  5423,  5452,  5481,  5510,  5539,
  5568,  5597,  5626,  5655,  5684,  5713,  5742,  5771,
  5800,  5829,  5858,  5887,  5916,  5945,  5974,  6003,
  6032,  6061,  6090,  6119,  6148,  6177,  6206,  6235,
  6264,  6293,  6322,  6351,  6380,  6409,  6438,  6467,
  6496,  6525,  6554,  6583,  6612,  6641,  6670,  6699,
  6728,  6757,  6786,  6815,  6844,  6873,  6902,  6931,
  6960,  6989,  7018,  7047,  7076,  7105,  7134,  7163,
  7192,  7221,  7250,  7279,  7308,  7337,  7366,  7395 };

/* Lookup tables for fast ICCIRY and ICCIRUV */
int LUT_ICCIRY[256] = {
                      -18,
                      -17,
                      -16,
                      -15,
                      -14,
                      -12,
                      -11,
                      -10,
                      -9,
                      -8,
                      -7,
                      -5,
                      -4,
                      -3,
                      -2,
                      -1,
                      0,
                      1,
                      2,
                      3,
                      4,
                      5,
                      7,
                      8,
                      9,
                      10,
                      11,
                      12,
                      14,
                      15,
                      16,
                      17,
                      18,
                      19,
                      21,
                      22,
                      23,
                      24,
                      25,
                      26,
                      28,
                      29,
                      30,
                      31,
                      32,
                      33,
                      35,
                      36,
                      37,
                      38,
                      39,
                      40,
                      42,
                      43,
                      44,
                      45,
                      46,
                      47,
                      49,
                      50,
                      51,
                      52,
                      53,
                      54,
                      56,
                      57,
                      58,
                      59,
                      60,
                      61,
                      63,
                      64,
                      65,
                      66,
                      67,
                      68,
                      70,
                      71,
                      72,
                      73,
                      74,
                      75,
                      77,
                      78,
                      79,
                      80,
                      81,
                      82,
                      84,
                      85,
                      86,
                      87,
                      88,
                      90,
                      91,
                      92,
                      93,
                      94,
                      95,
                      97,
                      98,
                      99,
                      100,
                      101,
                      102,
                      104,
                      105,
                      106,
                      107,
                      108,
                      109,
                      111,
                      112,
                      113,
                      114,
                      115,
                      116,
                      118,
                      119,
                      120,
                      121,
                      122,
                      123,
                      125,
                      126,
                      127,
                      128,
                      129,
                      130,
                      132,
                      133,
                      134,
                      135,
                      136,
                      137,
                      139,
                      140,
                      141,
                      142,
                      143,
                      144,
                      146,
                      147,
                      148,
                      149,
                      150,
                      151,
                      153,
                      154,
                      155,
                      156,
                      157,
                      158,
                      160,
                      161,
                      162,
                      163,
                      164,
                      165,
                      167,
                      168,
                      169,
                      170,
                      171,
                      173,
                      174,
                      175,
                      176,
                      177,
                      178,
                      180,
                      181,
                      182,
                      183,
                      184,
                      185,
                      187,
                      188,
                      189,
                      190,
                      191,
                      192,
                      194,
                      195,
                      196,
                      197,
                      198,
                      199,
                      201,
                      202,
                      203,
                      204,
                      205,
                      206,
                      208,
                      209,
                      210,
                      211,
                      212,
                      213,
                      215,
                      216,
                      217,
                      218,
                      219,
                      220,
                      222,
                      223,
                      224,
                      225,
                      226,
                      227,
                      229,
                      230,
                      231,
                      232,
                      233,
                      234,
                      236,
                      237,
                      238,
                      239,
                      240,
                      241,
                      243,
                      244,
                      245,
                      246,
                      247,
                      248,
                      250,
                      251,
                      252,
                      253,
                      254,
                      256,
                      257,
                      258,
                      259,
                      260,
                      261,
                      263,
                      264,
                      265,
                      266,
                      267,
                      268,
                      270,
                      271,
                      272,
                      273,
                      274,
                      275,
                      277,
                      278,
                      279};

int LUT_ICCIRUV_128[256] = {
                      -168,
                      -167,
                      -165,
                      -164,
                      -163,
                      -162,
                      -161,
                      -160,
                      -158,
                      -157,
                      -156,
                      -155,
                      -154,
                      -153,
                      -151,
                      -150,
                      -149,
                      -148,
                      -147,
                      -146,
                      -144,
                      -143,
                      -142,
                      -141,
                      -140,
                      -139,
                      -137,
                      -136,
                      -135,
                      -134,
                      -133,
                      -132,
                      -130,
                      -129,
                      -128,
                      -127,
                      -126,
                      -125,
                      -123,
                      -122,
                      -121,
                      -120,
                      -119,
                      -118,
                      -116,
                      -115,
                      -114,
                      -113,
                      -112,
                      -111,
                      -109,
                      -108,
                      -107,
                      -106,
                      -105,
                      -104,
                      -102,
                      -101,
                      -100,
                      -99,
                      -98,
                      -97,
                      -95,
                      -94,
                      -93,
                      -92,
                      -91,
                      -90,
                      -88,
                      -87,
                      -86,
                      -85,
                      -84,
                      -82,
                      -81,
                      -80,
                      -79,
                      -78,
                      -77,
                      -75,
                      -74,
                      -73,
                      -72,
                      -71,
                      -70,
                      -68,
                      -67,
                      -66,
                      -65,
                      -64,
                      -63,
                      -61,
                      -60,
                      -59,
                      -58,
                      -57,
                      -56,
                      -54,
                      -53,
                      -52,
                      -51,
                      -50,
                      -49,
                      -47,
                      -46,
                      -45,
                      -44,
                      -43,
                      -42,
                      -40,
                      -39,
                      -38,
                      -37,
                      -36,
                      -35,
                      -33,
                      -32,
                      -31,
                      -30,
                      -29,
                      -28,
                      -26,
                      -25,
                      -24,
                      -23,
                      -22,
                      -21,
                      -19,
                      -18,
                      -17,
                      -16,
                      -15,
                      -14,
                      -12,
                      -11,
                      -10,
                      -9,
                      -8,
                      -7,
                      -5,
                      -4,
                      -3,
                      -2,
                      -1,
                      0,
                      1,
                      2,
                      3,
                      4,
                      5,
                      7,
                      8,
                      9,
                      10,
                      11,
                      12,
                      14,
                      15,
                      16,
                      17,
                      18,
                      19,
                      21,
                      22,
                      23,
                      24,
                      25,
                      26,
                      28,
                      29,
                      30,
                      31,
                      32,
                      33,
                      35,
                      36,
                      37,
                      38,
                      39,
                      40,
                      42,
                      43,
                      44,
                      45,
                      46,
                      47,
                      49,
                      50,
                      51,
                      52,
                      53,
                      54,
                      56,
                      57,
                      58,
                      59,
                      60,
                      61,
                      63,
                      64,
                      65,
                      66,
                      67,
                      68,
                      70,
                      71,
                      72,
                      73,
                      74,
                      75,
                      77,
                      78,
                      79,
                      80,
                      81,
                      82,
                      84,
                      85,
                      86,
                      87,
                      88,
                      90,
                      91,
                      92,
                      93,
                      94,
                      95,
                      97,
                      98,
                      99,
                      100,
                      101,
                      102,
                      104,
                      105,
                      106,
                      107,
                      108,
                      109,
                      111,
                      112,
                      113,
                      114,
                      115,
                      116,
                      118,
                      119,
                      120,
                      121,
                      122,
                      123,
                      125,
                      126,
                      127,
                      128,
                      145};

unsigned char LUT_ICCIRYUV_R[256][256][256];
unsigned char LUT_ICCIRYUV_G[256][256][256];
unsigned char LUT_ICCIRYUV_B[256][256][256];

void initLUT_ICCIRYUV_RGB()
{
	int x, Y, U, V;
	int R, G, B;

	for(int y = 0 ; y < 256 ; y++)
	{
		Y = ((((y)-16)<<8)/219);
	
		for(int u = 0 ; u < 256 ; u++)
		{
		   	x = u - 128;
			U = (((x)<<8)/224);
			
			for(int v = 0 ; v < 256 ; v++)
			{
		   		x = v - 128;
				V = (((x)<<8)/224);
			
				R = CLIP(GET_R_FROM_YUV(Y, U, V));
				G = CLIP(GET_G_FROM_YUV(Y, U, V));
				B = CLIP(GET_B_FROM_YUV(Y, U, V));

				LUT_ICCIRYUV_R[y][u][v] = R;
				LUT_ICCIRYUV_G[y][u][v] = G;
				LUT_ICCIRYUV_B[y][u][v] = B;
			}
		}
	}
}

}
