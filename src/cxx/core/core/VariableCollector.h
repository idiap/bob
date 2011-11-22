/**
 * @file cxx/core/core/VariableCollector.h
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
#ifndef _TORCH5SPRO_VARIABLE_COLLECTOR_H_
#define _TORCH5SPRO_VARIABLE_COLLECTOR_H_

#include "core/general.h"

namespace Torch
{
	class File;

	struct Variable
	{
		enum Type
		{
			TypeNothing,
			TypeBool,
			TypeInt,
			TypeFloat,
			TypeDouble,
			TypeString,
			TypeIntArray,
			TypeFloatArray,
			TypeDoubleArray
		};

		// Constructor - default
		Variable();

		// Destructor
		virtual ~Variable();

		// Initializes the object
		void init(const char* name, const char* help, bool init_value);
		void init(const char* name, const char* help, int init_value);
		void init(const char* name, const char* help, float init_value);
		void init(const char* name, const char* help, double init_value);
		void init(const char* name, const char* help, const char* init_value);
		void init(const char* name, const char* help, int n_values, int init_value);
		void init(const char* name, const char* help, int n_values, float init_value);
		void init(const char* name, const char* help, int n_values, double init_value);
		void init(const Variable& other);

		// Delete the allocated memory
		void cleanup();

		Type  m_type;
		char* m_name;
		char* m_help;
		void* m_value;
		int   m_n_values;
	};

	class VariableCollector
	{
	public:

		// Constructor
		VariableCollector();

		// Destructor
		~VariableCollector();

		// Variable management functions - adding new ones
		bool		addB(const char* name, bool init_value, const char* help = "");
		bool		addI(const char* name, int init_value, const char* help = "");
		bool		addF(const char* name, float init_value, const char* help = "");
		bool		addD(const char* name, double init_value, const char* help = "");
		bool		addS(const char* name, const char* init_value, const char* help = "");

		bool		addIarray(const char* name, int n_values, int init_value, const char* help = "");
		bool		addFarray(const char* name, int n_values, float init_value, const char* help = "");
		bool		addDarray(const char* name, int n_values, double init_value, const char* help = "");

		// Variable management functions - changing their values
		bool		setB(const char* name, bool new_value);
		bool		setI(const char* name, int new_value);
		bool		setF(const char* name, float new_value);
		bool		setD(const char* name, double new_value);
		bool		setS(const char* name, const char* new_value);
		bool		setIarray(const char* name, int n_values);
		bool		setFarray(const char* name, int n_values);
		bool		setDarray(const char* name, int n_values);

		// Variable management functions - retrieving alues
		bool		getB(const char* name, bool* ok = 0);
		int		getI(const char* name, bool* ok = 0);
		float		getF(const char* name, bool* ok = 0);
		double		getD(const char* name, bool* ok = 0);
		const char*	getS(const char* name, bool* ok = 0);
		int*		getIarray(const char* name, bool* ok = 0);
		float*		getFarray(const char* name, bool* ok = 0);
		double*		getDarray(const char* name, bool* ok = 0);

		///
		void print();

		/// Loading/Saving the content from files
		bool		loadFile(File& file);
		bool		saveFile(File& file) const;

		///
		int getNvariables() { return m_size; };
		
		///
		Variable* getVariables() { return m_variables; };

		///
		bool copy(VariableCollector *variables_);
		
	private:

		//////////////////////////////////////////////////////////////////////

		// Resize the variable array as to accomodate new variables
		void		resize();

		// Deallocate allocated variables
		void		cleanup();

		// Search an variable given its name and returns its index (or <0 if not found)
		int		search(const char* name) const;

		// Sets an \c ok flag to the given value
		void		setOK(bool* ok, bool value) const;

	private:

		///////////////////////////////////////////////////////////
		// Attributes

		Variable*	m_variables;
		int		m_size;			// No of variables used
		int		m_capacity;		// No of variables allocated (>= m_size)
		int		m_resizeDelta;		// No of variables to pre-allocate
	};

}

#endif

