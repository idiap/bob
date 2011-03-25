/**
 * @author <a href="mailto:andre.dos.anjos@gmail.com">Andre Anjos</a> 
 * @date Sat 19 Mar 15:58:02 2011 
 *
 * @brief Implements many sorts of color transformations using standards
 */

#include <cmath>
#include <limits>
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
 * This method will scale and cast to integer a single float value, using the
 * standard library
 */
template <typename T>
static inline T scale (float value) {
  return static_cast<T>(rintf(std::numeric_limits<T>::max()*value));
}

/**
 * This method will scale and cast to float a single integer value, using the
 * standard library
 */
template <typename T>
static inline float normalize (T value) {
  return static_cast<float>(value)/static_cast<float>(std::numeric_limits<T>::max());
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

/**
 * This method clamps the float value between 0 and 1
 */
static float clamp (float f) {
  return (f<0)? 0.f : (f>1)? 1.f: f;
}

template <> void ip::rgb_to_hsv_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& h, uint8_t& s, uint8_t& v) {
  float H, S, V;
  rgb_to_hsv_one(normalize(r), normalize(g), normalize(b), H, S, V);
  h = scale<uint8_t>(H); s = scale<uint8_t>(S); v = scale<uint8_t>(V);
}

template <> void ip::rgb_to_hsv_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& h, uint16_t& s, uint16_t& v) {
  float H, S, V;
  rgb_to_hsv_one(normalize(r), normalize(g), normalize(b), H, S, V);
  h = scale<uint16_t>(H); s = scale<uint16_t>(S); v = scale<uint16_t>(V);
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
    if (g >= b) h = clamp((g - b)/C); //first sextant
    else h = clamp(1 - ((b - g)/C)); //sextant 6
  }
  else if (v == g) h = clamp(1.0f/3 + (b - r)/C); //sextants 2/3
  else h = clamp(2.0f/3 + (r - g)/C); //sextants 4/5
}

template <> void ip::hsv_to_rgb_one (uint8_t h, uint8_t s, uint8_t v,
    uint8_t& r, uint8_t& g, uint8_t& b) {
  float R, G, B;
  hsv_to_rgb_one(normalize(h), normalize(s), normalize(v), R, G, B);
  r = scale<uint8_t>(R); g = scale<uint8_t>(G); b = scale<uint8_t>(B);
}
	
template <> void ip::hsv_to_rgb_one (uint16_t h, uint16_t s, uint16_t v,
    uint16_t& r, uint16_t& g, uint16_t& b) {
  float R, G, B;
  hsv_to_rgb_one(normalize(h), normalize(s), normalize(v), R, G, B);
  r = scale<uint16_t>(R); g = scale<uint16_t>(G); b = scale<uint16_t>(B);
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
			r = clamp(v);
			g = clamp(X);
			b = clamp(m);
			break;
		case 1: //Hue is between yellow (red + green) and green
			r = clamp(X);
			g = clamp(v);
			b = clamp(m);
			break;
		case 2: //Hue is between green and cyan (green + blue)
			r = clamp(m);
			g = clamp(v);
			b = clamp(X);
			break;
		case 3: //Hue is between cyan (green + blue) and blue
			r = clamp(m);
			g = clamp(X);
			b = clamp(v);
			break;
		case 4: //Hue is between blue and magenta (blue + red)
			r = clamp(X);
			g = clamp(m);
			b = clamp(v);
			break;
		default: //Hue is between magenta (blue + red) and red
			r = clamp(v);
			g = clamp(m);
			b = clamp(X);
			break;
	}
}

template <> void ip::rgb_to_hsl_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& h, uint8_t& s, uint8_t& l) {
  float H, S, L;
  rgb_to_hsl_one(normalize(r), normalize(g), normalize(b), H, S, L);
  h = scale<uint8_t>(H); s = scale<uint8_t>(S); l = scale<uint8_t>(L);
}

template <> void ip::rgb_to_hsl_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& h, uint16_t& s, uint16_t& l) {
  float H, S, L;
  rgb_to_hsl_one(normalize(r), normalize(g), normalize(b), H, S, L);
  h = scale<uint16_t>(H); s = scale<uint16_t>(S); l = scale<uint16_t>(L);
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
  //S = C / (1 - |2*L -1|)
  float C = M - m; //chroma
  s = clamp(C / (1-fabsf(2*l - 1)));

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
    if (g >= b) h = clamp((g - b)/C); //first sextant
    else h = clamp(1 - ((b - g)/C)); //sextant 6
  }
  else if (M == g) h = clamp(1.0f/3 + (b - r)/C); //sextants 2/3
  else h = clamp(2.0f/3 + (r - g)/C); //sextants 4/5
}

template <> void ip::hsl_to_rgb_one (uint8_t h, uint8_t s, uint8_t l,
    uint8_t& r, uint8_t& g, uint8_t& b) {
  float R, G, B;
  hsl_to_rgb_one(normalize(h), normalize(s), normalize(l), R, G, B);
  r = scale<uint8_t>(R); g = scale<uint8_t>(G); b = scale<uint8_t>(B);
}
	
template <> void ip::hsl_to_rgb_one (uint16_t h, uint16_t s, uint16_t l,
    uint16_t& r, uint16_t& g, uint16_t& b) {
  float R, G, B;
  hsl_to_rgb_one(normalize(h), normalize(s), normalize(l), R, G, B);
  r = scale<uint16_t>(R); g = scale<uint16_t>(G); b = scale<uint16_t>(B);
}
	
template <> void ip::hsl_to_rgb_one (float h, float s, float l,
    float& r, float& g, float& b) {
  
  float C = s*(1-fabsf(2*l - 1)); //Chroma [0,1]
  const float v = (2*l + C)/2; //Value [0,1]
  
  if(v == 0.f) { // achromatic (gray)
		r = g = b = v; //Value
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
			r = clamp(v);
			g = clamp(X);
			b = clamp(m);
			break;
		case 1: //Hue is between yellow (red + green) and green
			r = clamp(X);
			g = clamp(v);
			b = clamp(m);
			break;
		case 2: //Hue is between green and cyan (green + blue)
			r = clamp(m);
			g = clamp(v);
			b = clamp(X);
			break;
		case 3: //Hue is between cyan (green + blue) and blue
			r = clamp(m);
			g = clamp(X);
			b = clamp(v);
			break;
		case 4: //Hue is between blue and magenta (blue + red)
			r = clamp(X);
			g = clamp(m);
			b = clamp(v);
			break;
		default: //Hue is between magenta (blue + red) and red
			r = clamp(v);
			g = clamp(m);
			b = clamp(X);
			break;
	}
}

template <> void ip::rgb_to_yuv_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& y, uint8_t& u, uint8_t& v) {
  float Y, U, V;
  rgb_to_yuv_one(normalize(r), normalize(g), normalize(b), Y, U, V);
  y = scale<uint8_t>(Y); u = scale<uint8_t>(U); v = scale<uint8_t>(V);
}

template <> void ip::rgb_to_yuv_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& y, uint16_t& u, uint16_t& v) {
  float Y, U, V;
  rgb_to_yuv_one(normalize(r), normalize(g), normalize(b), Y, U, V);
  y = scale<uint16_t>(Y); u = scale<uint16_t>(U); v = scale<uint16_t>(V);
}

/**
 * Using the JPEG YUV conversion scheme
 */
template <> void ip::rgb_to_yuv_one (float r, float g, float b,
    float& y, float& u, float& v) {
  ip::rgb_to_gray_one(r, g, b, y); //Y'
  u = clamp(0.5f - 0.168736f*r - 0.331264f*g + 0.5f*b); //Cb [0, 1]
  v = clamp(0.5f + 0.5f*r - 0.418688f*g - 0.081312*b); //Cr [0, 1]
}

template <> void ip::yuv_to_rgb_one (uint8_t y, uint8_t u, uint8_t v,
    uint8_t& r, uint8_t& g, uint8_t& b) {
  float R, G, B;
  yuv_to_rgb_one(normalize(y), normalize(u), normalize(v), R, G, B);
  r = scale<uint8_t>(R); g = scale<uint8_t>(G); b = scale<uint8_t>(B);
}
	
template <> void ip::yuv_to_rgb_one (uint16_t y, uint16_t u, uint16_t v,
    uint16_t& r, uint16_t& g, uint16_t& b) {
  float R, G, B;
  yuv_to_rgb_one(normalize(y), normalize(u), normalize(v), R, G, B);
  r = scale<uint16_t>(R); g = scale<uint16_t>(G); b = scale<uint16_t>(B);
}
	
/**
 * We are doing the inverse of the rgb_to_yuv_one() method above
 */
template <> void ip::yuv_to_rgb_one (float y, float u, float v,
    float& r, float& g, float& b) {
  r = clamp(y + 1.40199959f*(v-0.5f));
  b = clamp(y + 1.772000066f*(u-0.5f));
  g = clamp(y - 0.344135678f*(u-0.5f) - 0.714136156f*(v-0.5f));
}

template <> void ip::rgb_to_gray_one (uint8_t r, uint8_t g, uint8_t b,
    uint8_t& y) {
  float Y;
  rgb_to_gray_one(normalize(r), normalize(g), normalize(b), Y);
  y = scale<uint8_t>(Y);
}

template <> void ip::rgb_to_gray_one (uint16_t r, uint16_t g, uint16_t b,
    uint16_t& y) {
  float Y;
  rgb_to_gray_one(normalize(r), normalize(g), normalize(b), Y);
  y = scale<uint16_t>(Y);
}

/**
 * Y = 0.299R+0.587G+0.114B
 */
template <> void ip::rgb_to_gray_one (float r, float g, float b, 
    float& gray) {
  gray = clamp(0.299f*r + 0.587f*g + 0.114f*b);
}
