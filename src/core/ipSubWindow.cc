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
	if (	sw_x < 0 || sw_y < 0 || sw_w <= 0 || sw_h <= 0 ||
		sw_x + sw_w > getInputWidth() ||
		sw_y + sw_h > getInputHeight())
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
// Retrieve the sub-window to process in

int ipSubWindow::getSubWindowX() const
{
	return m_sw_x;
}

int ipSubWindow::getSubWindowY() const
{
	return m_sw_y;
}

int ipSubWindow::getSubWindowW() const
{
	return m_sw_w;
}

int ipSubWindow::getSubWindowH() const
{
	return m_sw_h;
}

//////////////////////////////////////////////////////////////////////////
// Change the input image size

bool ipSubWindow::setInputSize(const sSize& new_size)
{
	// Reset the sub-window, it may have no sense with the new window size
	m_sw_x = 0;
	m_sw_y = 0;
	m_sw_w = 0;
	m_sw_h = 0;

	return ipCore::setInputSize(new_size);
}

bool ipSubWindow::setInputSize(int new_w, int new_h)
{
	// Reset the sub-window, it may have no sense with the new window size
	m_sw_x = 0;
	m_sw_y = 0;
	m_sw_w = 0;
	m_sw_h = 0;

	return ipCore::setInputSize(new_w, new_h);
}

//////////////////////////////////////////////////////////////////////////

}

