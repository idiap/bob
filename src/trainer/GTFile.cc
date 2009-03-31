#include "GTFile.h"

namespace Torch {

///////////////////////////////////////////////////////////////////////////////
// Constructor

GTFile::GTFile(int n_points_)
	:	m_n_points(n_points_),
		m_points(new sPoint2D[m_n_points]),
		m_labels(new char*[m_n_points])
{
	addBOption("verbose", false, "verbose");

	// All the labels are by default <undefined>
	static const char* label_undef = "undefined";
	for (int i = 0; i < m_n_points; i ++)
	{
		m_labels[i] = new char[strlen(label_undef) + 1];
		strcpy(m_labels[i], label_undef);
	}
}

///////////////////////////////////////////////////////////////////////////////
// Destructor

GTFile::~GTFile()
{
	delete[] m_points;
	for (int i = 0; i < m_n_points; i ++)
	{
		delete[] m_labels[i];
	}
	delete[] m_labels;
}

///////////////////////////////////////////////////////////////////////////////
// Access functions (if <index> is invalid, NULL/0 will be returned)

const sPoint2D* GTFile::getPoint(int index) const
{
	return isIndex(index, m_n_points) == true ? &m_points[index] : 0;
}

const sPoint2D*	GTFile::getPoint(const char* label) const
{
	for (int i = 0; i < m_n_points; i ++)
	{
		if (strcasecmp(m_labels[i], label) == 0)
		{
			return &m_points[i];
		}
	}

	return 0;
}

const char* GTFile::getLabel(int index) const
{
	return isIndex(index, m_n_points) == true ? m_labels[index] : 0;
}

///////////////////////////////////////////////////////////////////////////////
// Check if some label is defined

bool GTFile::hasLabel(const char* label) const
{
	return getIndex(label) >= 0;
}

///////////////////////////////////////////////////////////////////////////////
// Get the index for some label (or -1, if the label does not exist)

int GTFile::getIndex(const char* label) const
{
	for (int i = 0; i < m_n_points; i ++)
	{
		if (strcasecmp(m_labels[i], label) == 0)
		{
			return i;
		}
	}

	return -1;
}

///////////////////////////////////////////////////////////////////////////////
// Set a label at some point index

bool GTFile::setLabel(int index, const char* label)
{
	if (isIndex(index, m_n_points) == true)
	{
		delete[] m_labels[index];
		m_labels[index] = new char[strlen(label) + 1];
		strcpy(m_labels[index], label);
		return true;
	}

	return false;
}

///////////////////////////////////////////////////////////////////////////////

}
