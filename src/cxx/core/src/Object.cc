/**
 * @file cxx/core/src/Object.cc
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
#include "core/Object.h"

namespace bob
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
