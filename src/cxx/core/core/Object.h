/**
 * @file cxx/core/core/Object.h
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
#ifndef _BOB5SPRO_OBJECT_H_
#define _BOB5SPRO_OBJECT_H_

#include "core/general.h"
#include "core/VariableCollector.h"

namespace bob
{
   	class VariableCollector;

	class Object
	{
	public:

		/// Constructor
		Object();

	    	/// Destructor
	    	virtual ~Object();

		///////////////////////////////////////////////////////////
		/// Option management functions - adding new ones

		/// returns \c false if the name is already taken
		bool			addBOption(const char* name, bool init_value, const char* help = "");
		bool			addIOption(const char* name, int init_value, const char* help = "");
		bool			addFOption(const char* name, float init_value, const char* help = "");
		bool			addDOption(const char* name, double init_value, const char* help = "");
		bool			addSOption(const char* name, const char* init_value, const char* help = "");

		///////////////////////////////////////////////////////////
		/// Option management functions - changing their values

		/// returns \c false if the name is wrong or the option has a different type
		bool			setBOption(const char* name, bool new_value);
		bool			setIOption(const char* name, int new_value);
		bool			setFOption(const char* name, float new_value);
		bool			setDOption(const char* name, double new_value);
		bool			setSOption(const char* name, const char* new_value);

		///////////////////////////////////////////////////////////
		/// Option management functions - retrieving alues

		/// it the name is wrong of the option has a different type, the return value is invalid
		///	check the \c ok value
		bool			getBOption(const char* name, bool* ok = 0);
		int			getIOption(const char* name, bool* ok = 0);
		float			getFOption(const char* name, bool* ok = 0);
		double			getDOption(const char* name, bool* ok = 0);
		const char*		getSOption(const char* name, bool* ok = 0);

    /**
     * Returns a handle to all variables registered.
     */
		inline int getNvariables() { return m_optionImpl->getNvariables(); }
		inline const Variable* getVariables() { return m_optionImpl->getVariables(); }

	protected:

		/// called when some option was changed
		virtual void		optionChanged(const char* name) { }

	private:

		///////////////////////////////////////////////////////////
		/// Attributes

		VariableCollector*	m_optionImpl;		// Implementation details (not exposed in the interface)
	};
}

#endif
