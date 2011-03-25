/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 19 Mar 15:58:02 2011 
 *
 * @brief Implements many sorts of color transformations using standards
 */

#include <cmath>
#include <boost/format.hpp>
#include "ip/color.h"

namespace ip = Torch::ip;

ip::UnsupportedTypeForColorConversion::UnsupportedTypeForColorConversion(Torch::core::array::ElementType eltype) throw(): m_eltype(eltype) {
}

ip::UnsupportedTypeForColorConversion::~UnsupportedTypeForColorConversion() throw() {
}

const char* ip::UnsupportedTypeForColorConversion::what() const throw() {
  try {
    boost::format message("Color conversion for type '%s' is not supported");
    message % Torch::core::array::stringize(m_eltype);
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "ip::UnsupportedTypeForColorConversion: cannot format, exception raised";
    return emergency;
  }
}

ip::UnsupportedRowExtent::UnsupportedRowExtent(int expected, int got) throw(): 
  m_expected(expected),
  m_got(got)
{
}

ip::UnsupportedRowExtent::~UnsupportedRowExtent() throw() {
}

const char* ip::UnsupportedRowExtent::what() const throw() {
  try {
    boost::format message("Color conversion requires an array with %d rows, but I got %d instead");
    message % m_expected % m_got;
    m_message = message.str();
    return m_message.c_str();
  } catch (...) {
    static const char* emergency = "ip::UnsupportedRowExtent: cannot format, exception raised";
    return emergency;
  }
}

/**
 * This method will round and cast to uint8_t a single float value, using the
 * standard library
 */
static inline uint8_t round (float value) {
  return static_cast<uint8_t>(rintf(value));
}

/**
 * This method calculates the greatest value on the input 3-tuple
 */
template <typename T>
static T tmax (T c1, T c2, T c3) {
  if (c2 >= c3) {
    if (c1 >= c2) return c1;
    return c2;
  }
  if (c1 >= c3) return c1;
  return c3;
}

/**
 * This method calculates the lowest value on the input 3-tuple
 */
template <typename T>
static T tmin (T c1, T c2, T c3) {
  if (c2 <= c3) {
    if (c1 <= c2) return c1;
    return c2;
  }
  if (c1 <= c3) return c1;
  return c3;
}

template <> void ip::rgb_to_hsv_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& h, uint8_t& s, uint8_t& v) {
  float H, S, V;
  rgb_to_hsv_one(r/255.f, g/255.f, b/255.f, H, S, V);
  h = round(255*H); s = round(255*S); v = round(255*V);
}

template <> void ip::rgb_to_hsv_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& h, uint16_t& s, uint16_t& v) {
  float H, S, V;
  rgb_to_hsv_one(r/65535.f, g/65535.f, b/65535.f, H, S, V);
  h = round(65535*H); s = round(65535*S); v = round(65535*V);
}

template <> void ip::rgb_to_hsv_one (float r, float g, float b,
    float& h, float& s, float& v) {
  v = tmax(r, g, b); //value
  
  //if the Value is 0, then we also set the other values to zero
  if (v == 0.0f) { 
    h = s = v;
    return;
  }

  //computing the saturation
  float C = v - tmin(r, g, b);
  s = C / v;

  //if the Saturation value is zero, set Hue to zero and return
  if (s == 0.0f) {
    h = s;
    return;
  }

  /**
   * 1) Hue' = (Green - Blue) / C, if Value == Red
   * 2) Hue' = 2 + (Blue - Red) / C, if Value == Green
   * 3) Hue' = 4 + (Red - Green) / C, if Value == Blue
   *
   * Note: Hue' varies between 0 and 6 using the above formulation, we multiply
   * C by 6 to make sure it varies between 0 and 1 (normalized).
   */
  C *= 6;
  if (v == r) {
    //When V == Red, we need to be careful because the Hue will wrap
    if (g >= b) h = (g - b)/C; //first sextant
    else h = 1 - ((b - g)/C); //sextant 6
  }
  else if (v == g) h = 1.0f/3 + (b - r)/C; //sextants 2/3
  else h = 2.0f/3 + (r - g)/C; //sextants 4/5
}

template <> void ip::hsv_to_rgb_one (uint8_t h, uint8_t s, uint8_t v,
    uint8_t& r, uint8_t& g, uint8_t& b) {
  float R, G, B;
  hsv_to_rgb_one(h/255.f, s/255.f, v/255.f, R, G, B);
  r = round(255*R); g = round(255*G); b = round(255*B);
}
	
template <> void ip::hsv_to_rgb_one (uint16_t h, uint16_t s, uint16_t v,
    uint16_t& r, uint16_t& g, uint16_t& b) {
  float R, G, B;
  hsv_to_rgb_one(h/65535.f, s/65535.f, v/65535.f, R, G, B);
  r = round(65535*R); g = round(65535*G); b = round(65535*B);
}
	
template <> void ip::hsv_to_rgb_one (float h, float s, float v,
    float& r, float& g, float& b) {
	
  if(s == 0) { // achromatic (gray)
		r = g = b = v;
		return;
	}

  const float Hp = 6*h;
  const uint8_t sextant = static_cast<uint8_t>(Hp);
  const float Hpmod2 = Hp - (2 * static_cast<uint8_t>(Hp/2)); //Hp%2
  float C = v * s;
  const float m = v - C;
  const float X = C * (1 - fabsf(Hpmod2 - 1)) + m;
  C += m;

	switch(sextant) {
		case 0: //Hue is between red and yellow (red + green)
			r = v;
			g = X;
			b = m;
			break;
		case 1: //Hue is between yellow (red + green) and green
			r = X;
			g = v;
			b = m;
			break;
		case 2: //Hue is between green and cyan (green + blue)
			r = m;
			g = v;
			b = X;
			break;
		case 3: //Hue is between cyan (green + blue) and blue
			r = m;
			g = X;
			b = v;
			break;
		case 4: //Hue is between blue and magenta (blue + red)
			r = X;
			g = m;
			b = v;
			break;
		default: //Hue is between magenta (blue + red) and red
			r = v;
			g = m;
			b = X;
			break;
	}
}

template <> void ip::rgb_to_hsl_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& h, uint8_t& s, uint8_t& l) {
  float H, S, L;
  rgb_to_hsl_one(r/255.f, g/255.f, b/255.f, H, S, L);
  h = round(255*H); s = round(255*S); l = round(255*L);
}

template <> void ip::rgb_to_hsl_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& h, uint16_t& s, uint16_t& l) {
  float H, S, L;
  rgb_to_hsl_one(r/65535.f, g/65535.f, b/65535.f, H, S, L);
  h = round(65535*H); s = round(65535*S); l = round(65535*L);
}

template <> void ip::rgb_to_hsl_one (float r, float g, float b,
    float& h, float& s, float& l) {
  //lightness calculation: L = (M + m)/2
  const float M = tmax(r, g, b);
  const float m = tmin(r, g, b);
  l = 0.5 * (M+m);
  
  //if the lightness is 0, then we also set the other values to zero
  if (l == 0) { 
    h = s = l;
    return;
  }

  //computing the saturation based on the lightness:
  //S = 255 * C / (1 - |2*L -1|)
  float C = M - m; //chroma
  s = C / (1-fabsf(2*l - 1));

  //if the Saturation value is zero, set Hue to zero and return
  if (s == 0) {
    h = s;
    return;
  }

  /**
   * 1) Hue' = (Green - Blue) / C, if Value == Red
   * 2) Hue' = 2 + (Blue - Red) / C, if Value == Green
   * 3) Hue' = 4 + (Red - Green) / C, if Value == Blue
   *
   * Note: Hue' varies between 0 and 6 using the above formulation, we multiply
   * C by 6 to make sure it varies between 0 and 1 (normalized).
   */
  C *= 6;
  if (M == r) {
    //When V == Red, we need to be careful because the Hue will wrap
    if (g >= b) h = (g - b)/C; //first sextant
    else h = 1 - ((b - g)/C); //sextant 6
  }
  else if (M == g) h = 1.0f/3 + (b - r)/C; //sextants 2/3
  else h = 2.0f/3 + (r - g)/C; //sextants 4/5
}

template <> void ip::hsl_to_rgb_one (uint8_t h, uint8_t s, uint8_t l,
    uint8_t& r, uint8_t& g, uint8_t& b) {
  float R, G, B;
  hsl_to_rgb_one(h/255.f, s/255.f, l/255.f, R, G, B);
  r = round(255*R); g = round(255*G); b = round(255*B);
}
	
template <> void ip::hsl_to_rgb_one (uint16_t h, uint16_t s, uint16_t l,
    uint16_t& r, uint16_t& g, uint16_t& b) {
  float R, G, B;
  hsl_to_rgb_one(h/65535.f, s/65535.f, l/65535.f, R, G, B);
  r = round(65535*R); g = round(65535*G); b = round(65535*B);
}
	
template <> void ip::hsl_to_rgb_one (float h, float s, float l,
    float& r, float& g, float& b) {
  
  float C = s*(1-fabsf(2*l - 1)); //Chroma (0-255)
  const float v = (2*l + C)/2; //Value (0-255)
  
  if(v == 0.f) { // achromatic (gray)
		r = g = b = round(v); //Value
		return;
	}

  const float Hp = 6*h;
  const uint8_t sextant = static_cast<uint8_t>(Hp);
  const float Hpmod2 = Hp - (2 * static_cast<uint8_t>(Hp/2)); //Hp%2
  const float m = l - C/2;
  const float X = C * (1 - fabsf(Hpmod2 - 1)) + m;
  C += m;

	switch(sextant) {
		case 0: //Hue is between red and yellow (red + green)
			r = v;
			g = X;
			b = m;
			break;
		case 1: //Hue is between yellow (red + green) and green
			r = X;
			g = v;
			b = m;
			break;
		case 2: //Hue is between green and cyan (green + blue)
			r = m;
			g = v;
			b = X;
			break;
		case 3: //Hue is between cyan (green + blue) and blue
			r = m;
			g = X;
			b = v;
			break;
		case 4: //Hue is between blue and magenta (blue + red)
			r = X;
			g = m;
			b = v;
			break;
		default: //Hue is between magenta (blue + red) and red
			r = v;
			g = m;
			b = X;
			break;
	}
}

template <> void ip::rgb_to_yuv_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& y, uint8_t& u, uint8_t& v) {
  float Y, U, V;
  rgb_to_yuv_one(r/255.f, g/255.f, b/255.f, Y, U, V);
  y = round(255*Y); u = round(255*U); v = round(255*V);
}

template <> void ip::rgb_to_yuv_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& y, uint16_t& u, uint16_t& v) {
  float Y, U, V;
  rgb_to_yuv_one(r/65535.f, g/65535.f, b/65535.f, Y, U, V);
  y = round(65535*Y); u = round(65535*U); v = round(65535*V);
}

/**
 * Using the JPEG YUV conversion scheme
 */
template <> void ip::rgb_to_yuv_one (float r, float g, float b,
    float& y, float& u, float& v) {
  ip::rgb_to_gray_one(r, g, b, y); //Y'
  u = 128 - 0.168736f*r - 0.331264f*g + 0.5f*b; //Cb
  v = 128 + 0.5f*r - 0.418688f*g - 0.081312f*b; //Cr
}

template <> void ip::yuv_to_rgb_one (uint8_t y, uint8_t u, uint8_t v,
    uint8_t& r, uint8_t& g, uint8_t& b) {
  float R, G, B;
  yuv_to_rgb_one(y/255.f, u/255.f, v/255.f, R, G, B);
  r = round(255*R); g = round(255*G); b = round(255*B);
}
	
template <> void ip::yuv_to_rgb_one (uint16_t y, uint16_t u, uint16_t v,
    uint16_t& r, uint16_t& g, uint16_t& b) {
  float R, G, B;
  yuv_to_rgb_one(y/65535.f, u/65535.f, v/65535.f, R, G, B);
  r = round(65535*R); g = round(65535*G); b = round(65535*B);
}
	
/**
 * We are doing the inverse of the rgb_to_yuv_one() method above
 */
template <> void ip::yuv_to_rgb_one (float y, float u, float v,
    float& r, float& g, float& b) {
  r = y + 1.40199959f*(v-128);
  b = y + 1.772000066f*(u-128);
  g =  - 0.71413616f*(v-128);
}

template <> void ip::rgb_to_gray_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& y) {
  float Y;
  rgb_to_gray_one(r/255.f, g/255.f, b/255.f, Y);
  y = round(255.f*Y);
}

template <> void ip::rgb_to_gray_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& y) {
  float Y;
  rgb_to_gray_one(r/65535.f, g/65535.f, b/65535.f, Y);
  y = round(65535.f*Y);
}

/**
 * Y = 0.299R+0.587G+0.114B
 */
template <> void ip::rgb_to_gray_one (float r, float g, float b, 
    float& gray) {
  gray = 0.299f*r + 0.587f*g + 0.114f*b;
}
