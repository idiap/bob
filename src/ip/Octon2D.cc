#include "ip/Octon2D.h"
#include "ip/Color.h"

namespace Torch {

Octon2D::~Octon2D()
{
}

void Octon2D::draw(Image *image, Color color)
{
	int y_min, y_max;
        int x_min, x_max;
        int ypx_min, ypx_max;
        int ymx_min, ymx_max;

        y_min = octon.y_min;
        y_max = octon.y_max;
        x_min = octon.x_min;
        x_max = octon.x_max;
        ypx_min = octon.ypx_min;
        ypx_max = octon.ypx_max;
        ymx_min = octon.ymx_min;
        ymx_max = octon.ymx_max;

        // Display in anti-clockwise direction
        image->drawLine(ypx_max - y_max, y_max, x_max, ypx_max - x_max, color);
        image->drawLine(x_max, ypx_max - x_max, x_max, ymx_min + x_max, color);
        image->drawLine(x_max, ymx_min + x_max, y_min - ymx_min, y_min, color);
        image->drawLine(y_min - ymx_min, y_min, ypx_min - y_min, y_min, color);
        image->drawLine(ypx_min - y_min, y_min, x_min, ypx_min - x_min, color);
        image->drawLine(x_min, ypx_min - x_min, x_min, ymx_max + x_min, color);
        image->drawLine(x_min, ymx_max + x_min, y_max - ymx_max, y_max, color);
        image->drawLine(y_max - ymx_max, y_max, ypx_max - y_max, y_max, color);
}

#ifdef HAVE_X11
void Octon2D::xdraw(Display *pDisplay, Pixmap pixmap, GC gc, unsigned long color, int line_width, int line_style, bool fill, int fill_style)
{
	int y_min, y_max;
        int x_min, x_max;
        int ypx_min, ypx_max;
        int ymx_min, ymx_max;

        y_min = octon.y_min;
        y_max = octon.y_max;
        x_min = octon.x_min;
        x_max = octon.x_max;
        ypx_min = octon.ypx_min;
        ypx_max = octon.ypx_max;
        ymx_min = octon.ymx_min;
        ymx_max = octon.ymx_max;

	XSetForeground(pDisplay, gc, color);

        // Display in anti-clockwise direction
        XdrawLine(pDisplay, pixmap, gc, ypx_max - y_max, y_max, x_max, ypx_max - x_max);
        XdrawLine(pDisplay, pixmap, gc, x_max, ypx_max - x_max, x_max, ymx_min + x_max);
        XdrawLine(pDisplay, pixmap, gc, x_max, ymx_min + x_max, y_min - ymx_min, y_min);
        XdrawLine(pDisplay, pixmap, gc, y_min - ymx_min, y_min, ypx_min - y_min, y_min);
        XdrawLine(pDisplay, pixmap, gc, ypx_min - y_min, y_min, x_min, ypx_min - x_min);
        XdrawLine(pDisplay, pixmap, gc, x_min, ypx_min - x_min, x_min, ymx_max + x_min);
        XdrawLine(pDisplay, pixmap, gc, x_min, ymx_max + x_min, y_max - ymx_max, y_max);
        XdrawLine(pDisplay, pixmap, gc, y_max - ymx_max, y_max, ypx_max - y_max, y_max);
}
#endif

bool Octon2D::inside(const Point2D& P_)
{
	bool bInOcton;
        int ypx, ymx;

        bInOcton = false;

        if(	(P_.get(1) >= octon.y_min) && (P_.get(1) <= octon.y_max) &&
		(P_.get(0) >= octon.x_min) && (P_.get(1) <= octon.x_max))
        {
                ypx = (int) (P_.get(1) + P_.get(0));
                ymx = (int) (P_.get(1) - P_.get(0));

                bInOcton = true;

                if(ypx > octon.ypx_max) bInOcton = false;
                else if(ypx < octon.ypx_min) bInOcton = false;
                if(ymx > octon.ymx_max) bInOcton = false;
                else if(ymx < octon.ymx_min) bInOcton = false;
        }

        return bInOcton;

	return false;
}

}
