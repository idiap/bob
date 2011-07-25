/**
 * @author Andre Anjos <andre.anjos@idiap.ch>
 * @date Sun 24 Jul 20:39:52 2011
 *
 * @brief Binds simple drawing primitives
 */

#include <boost/python.hpp>
#include "ip/drawing.h"

using namespace boost::python;
namespace ip = Torch::ip;

template <typename T>
struct drawing_binder {

  void bind_grayscale() {
    def("draw_point_", (void (*)(blitz::Array<T,2>& image, int, int, T))&ip::draw_point_<T>, (arg("image"), arg("x"), arg("y"), arg("gray")), "Draws a point in the grayscale (2D) image. No checks, if you try to access an area outside the image using this method, you may trigger a segmentation fault.");
    def("draw_point", (void (*)(blitz::Array<T,2>& image, int, int, T))&ip::draw_point<T>, (arg("image"), arg("x"), arg("y"), arg("gray")), "Draws a point in the given grayscale (2D) image. Trying to access outside the image range will trigger an IndexError.");
    def("try_draw_point", (void (*)(blitz::Array<T,2>& image, int, int, T))&ip::try_draw_point<T>, (arg("image"), arg("x"), arg("y"), arg("gray")), "Tries to draw a point at the given grayscale (2D) image. If the point is out of range, just ignores the request. This is what is used for drawing lines.");
    def("draw_line", &ip::draw_line<blitz::Array<T,2>, T>, (arg("image"), arg("x1"), arg("y1"), arg("x2"), arg("y2"), arg("gray")), "Draws a line between two points p1(x1,y1) and p2(x2,y2).  This function is based on the Bresenham's line algorithm and is highly optimized to be able to draw lines very quickly. There is no floating point arithmetic nor multiplications and divisions involved. Only addition, subtraction and bit shifting are used.\n\nThe line may go out of the image bounds in which case such points (lying outside the image boundary are ignored).\n\nReferences: http://en.wikipedia.org/wiki/Bresenham's_line_algorithm");
    def("draw_cross", &ip::draw_cross<blitz::Array<T,2>, T>, (arg("image"), arg("x"), arg("y"), arg("radius"), arg("gray")), "Draws a cross with a given radius and gray level at the image. Uses the draw_line() primitive above. The cross will look like an 'x' and not like a '+'. To get a '+' sign, use the draw_cross_plus() variant.");
    def("draw_cross_plus", &ip::draw_cross_plus<blitz::Array<T,2>, T>, (arg("image"), arg("x"), arg("y"), arg("radius"), arg("gray")), "Draws a cross with a given radius and gray level at the image. Uses the draw_line() primitive above. The cross will look like an '+' and not like a 'x'. To get a 'x' sign, use the draw_cross() variant.");
    def("draw_box", &ip::draw_box<blitz::Array<T,2>, T>, (arg("image"), arg("x"), arg("y"), arg("width"), arg("height"), arg("gray")), "Draws a box at the image using the draw_line() primitive.");
  }

  void bind_color() {
    def("draw_point_", (void (*)(blitz::Array<T,3>& image, int, int, const boost::tuple<T,T,T>&))&ip::draw_point_<T>, (arg("image"), arg("x"), arg("y"), arg("color")), "Draws a point in the color (3D) image. No checks, if you try to access an area outside the image using this method, you may trigger a segmentation fault.");
    def("draw_point", (void (*)(blitz::Array<T,3>& image, int, int, const boost::tuple<T,T,T>&))&ip::draw_point<T>, (arg("image"), arg("x"), arg("y"), arg("color")), "Draws a point in the given color (3D) image. Trying to access outside the image range will trigger an IndexError.");
    def("try_draw_point", (void (*)(blitz::Array<T,3>& image, int, int, const boost::tuple<T,T,T>&))&ip::try_draw_point<T>, (arg("image"), arg("x"), arg("y"), arg("color")), "Tries to draw a point at the given color (3D) image. If the point is out of range, just ignores the request. This is what is used for drawing lines.");
    def("draw_line", &ip::draw_line<blitz::Array<T,3>, boost::tuple<T,T,T> >, (arg("image"), arg("x1"), arg("y1"), arg("x2"), arg("y2"), arg("color")), "Draws a line between two points p1(x1,y1) and p2(x2,y2).  This function is based on the Bresenham's line algorithm and is highly optimized to be able to draw lines very quickly. There is no floating point arithmetic nor multiplications and divisions involved. Only addition, subtraction and bit shifting are used.\n\nThe line may go out of the image bounds in which case such points (lying outside the image boundary are ignored).\n\nReferences: http://en.wikipedia.org/wiki/Bresenham's_line_algorithm");
    def("draw_cross", &ip::draw_cross<blitz::Array<T,3>, boost::tuple<T,T,T> >, (arg("image"), arg("x"), arg("y"), arg("radius"), arg("color")), "Draws a cross with a given radius and color at the image. Uses the draw_line() primitive above. The cross will look like an 'x' and not like a '+'. To get a '+' sign, use the draw_cross_plus() variant.");
    def("draw_cross_plus", &ip::draw_cross_plus<blitz::Array<T,3>, boost::tuple<T,T,T> >, (arg("image"), arg("x"), arg("y"), arg("radius"), arg("color")), "Draws a cross with a given radius and color at the image. Uses the draw_line() primitive above. The cross will look like an '+' and not like a 'x'. To get a 'x' sign, use the draw_cross() variant.");
    def("draw_box", &ip::draw_box<blitz::Array<T,3>, boost::tuple<T,T,T> >, (arg("image"), arg("x"), arg("y"), arg("width"), arg("height"), arg("color")), "Draws a box at the image using the draw_line() primitive.");
  }

  drawing_binder() {
    bind_grayscale();
    bind_color();
  }

};

void bind_ip_drawing() {
  drawing_binder<uint8_t>();
  drawing_binder<uint16_t>();
  drawing_binder<float>();
}
