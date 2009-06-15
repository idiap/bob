#include "Object.h"

namespace Torch
{

// Constructor
Object::Object()
	: 	m_optionImpl(new VariableCollector())
{
	addBOption("verbose", false, "verbose flag");
}

// Destructor
Object::~Object()
{
	delete m_optionImpl;
}

//////////////////////////////////////////////////////////////////////////////////////
// Option management functions - just wrappers over OptionImpl

// addXOption
bool Object::addBOption(const char* name, bool init_value, const char* help)
{
	return m_optionImpl->addB(name, init_value, help);
}
bool Object::addIOption(const char* name, int init_value, const char* help)
{
	return m_optionImpl->addI(name, init_value, help);
}
bool Object::addFOption(const char* name, float init_value, const char* help)
{
	return m_optionImpl->addF(name, init_value, help);
}
bool Object::addDOption(const char* name, double init_value, const char* help)
{
	return m_optionImpl->addD(name, init_value, help);
}
bool Object::addSOption(const char* name, const char* init_value, const char* help)
{
	return m_optionImpl->addS(name, init_value, help);
}

// setXOption
bool Object::setBOption(const char* name, bool new_value)
{
	if (m_optionImpl->setB(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Object::setIOption(const char* name, int new_value)
{
	if (m_optionImpl->setI(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Object::setFOption(const char* name, float new_value)
{
	if (m_optionImpl->setF(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Object::setDOption(const char* name, double new_value)
{
	if (m_optionImpl->setD(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Object::setSOption(const char* name, const char* new_value)
{
	if (m_optionImpl->setS(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}

// getXOption
bool Object::getBOption(const char* name, bool* ok)
{
	return m_optionImpl->getB(name, ok);
}
int Object::getIOption(const char* name, bool* ok)
{
	return m_optionImpl->getI(name, ok);
}
float Object::getFOption(const char* name, bool* ok)
{
	return m_optionImpl->getF(name, ok);
}
double Object::getDOption(const char* name, bool* ok)
{
	return m_optionImpl->getD(name, ok);
}
const char* Object::getSOption(const char* name, bool* ok)
{
	return m_optionImpl->getS(name, ok);
}

}
