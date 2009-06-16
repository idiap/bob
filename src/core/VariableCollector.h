#ifndef _TORCH5SPRO_VARIABLE_COLLECTOR_H_
#define _TORCH5SPRO_VARIABLE_COLLECTOR_H_

#include "general.h"

namespace Torch
{
	class File;
	struct Variable;

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

	private:

		//////////////////////////////////////////////////////////////////////

		// Resize the variable array as to accomodate new variables
		void		resize();

		// Deallocate allocated variables
		void		cleanup();

		// Search an variable given its name and returns its index (or <0 if not found)
		int		search(const char* name) const;

		// Sets an <ok> flag to the given value
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

