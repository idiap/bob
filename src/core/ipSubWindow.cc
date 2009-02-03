#include "ipSubWindow.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

ipSubWindow::ipSubWindow()
	: 	ipCore(),
		m_sw_x(0), m_sw_y(0), m_sw_w(0), m_sw_h(0)
{
}

//////////////////////////////////////////////////////////////////////////
// Destructor

ipSubWindow::~ipSubWindow()
{
}

//////////////////////////////////////////////////////////////////////////
// Change the sub-window to process in

bool ipSubWindow::setSubWindow(int sw_x, int sw_y, int sw_w, int sw_h)
{
	if (	sw_x < 0 || sw_y < 0 || sw_w <= 0 || sw_h <= 0)
	{
		return false;
	}

	m_sw_x = sw_x;
	m_sw_y = sw_y;
	m_sw_w = sw_w;
	m_sw_h = sw_h;
	return true;
}

//////////////////////////////////////////////////////////////////////////

}

