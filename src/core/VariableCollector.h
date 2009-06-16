#ifndef _TORCH5SPRO_VARIABLE_COLLECTOR_H_
#define _TORCH5SPRO_VARIABLE_COLLECTOR_H_

#include "general.h"

namespace Torch
{
	class File;

	class VariableCollector
	{
	public:
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
			Variable()
				:	m_type(TypeNothing),
					m_name(0), m_help(0),
					m_value(0), m_n_values(0)
			{
			}

			// Destructor
			~Variable()
			{
				cleanup();
			}

			// Initialize the object - bool
			void			init(const char* name, const char* help, bool init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeBool;
				m_value = new bool(init_value);
				m_n_values = 1;
			}

			// Initialize the object - integer
			void			init(const char* name, const char* help, int init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeInt;
				m_value = new int(init_value);
				m_n_values = 1;
			}

			// Initialize the object - float
			void			init(const char* name, const char* help, float init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeFloat;
				m_value = new float(init_value);
				m_n_values = 1;
			}

			// Initialize the object - double
			void			init(const char* name, const char* help, double init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeDouble;
				m_value = new double(init_value);
				m_n_values = 1;
			}

			// Initialize the object - string
			void			init(const char* name, const char* help, const char* init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeString;
				m_value = new char[strlen(init_value) + 1];
				strcpy((char*)m_value, init_value);
				m_n_values = strlen(init_value);
			}

			// Initialize the object - integer array
			void			init(const char* name, const char* help, int n_values, int init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeIntArray;
				if(n_values <= 0)
				{
					m_value = NULL;
					m_n_values = 0;
				}
				else
				{
					int *value_ = new int[n_values];
					for(int i = 0 ; i < n_values ; i++) value_[i] = init_value;
					m_value = value_;
					m_n_values = n_values;
				}
			}

			// Initialize the object - float array
			void			init(const char* name, const char* help, int n_values, float init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeFloatArray;
				if(n_values <= 0)
				{
					m_value = NULL;
					m_n_values = 0;
				}
				else
				{
					float *value_ = new float[n_values];
					for(int i = 0 ; i < n_values ; i++) value_[i] = init_value;
					m_value = value_;
					m_n_values = n_values;
				}
			}

			// Initialize the object - double array
			void			init(const char* name, const char* help, int n_values, double init_value)
			{
				cleanup();

				m_name = new char[strlen(name) + 1];
				m_help = new char[strlen(help) + 1];
				strcpy(m_name, name);
				strcpy(m_help, help);

				m_type = TypeDoubleArray;
				if(n_values <= 0)
				{
					m_value = NULL;
					m_n_values = 0;
				}
				else
				{
					double *value_ = new double[n_values];
					for(int i = 0 ; i < n_values ; i++) value_[i] = init_value;
					m_value = value_;
					m_n_values = n_values;
				}
			}

			// Initialize the object - from another
			void			init(const Variable& other)
			{
				cleanup();

				m_name = new char[strlen(other.m_name) + 1];
				m_help = new char[strlen(other.m_help) + 1];
				strcpy(m_name, other.m_name);
				strcpy(m_help, other.m_help);

				m_type = other.m_type;
				m_n_values = other.m_n_values;

				switch(other.m_type)
				{
				case TypeBool:
					m_value = new bool(*((bool*)other.m_value));
					break;

				case TypeInt:
					m_value = new int(*((int*)other.m_value));
					break;

				case TypeFloat:
					m_value = new float(*((float*)other.m_value));
					break;

				case TypeDouble:
					m_value = new double(*((double*)other.m_value));
					break;

				case TypeString:
					m_value = new char[strlen((const char*)other.m_value) + 1];
					strcpy((char*)m_value, (const char*)other.m_value);
					break;

				case TypeIntArray:
					{
					int *ivalue_src_ = (int *) other.m_value;
					int *ivalue_dst_ = new int[m_n_values];
					for(int i = 0 ; i < m_n_values ; i++) ivalue_dst_[i] = ivalue_src_[i];
					m_value = ivalue_dst_;
					}
					break;

				case TypeFloatArray:
					{
					float *fvalue_src_ = (float *) other.m_value;
					float *fvalue_dst_ = new float[m_n_values];
					for(int i = 0 ; i < m_n_values ; i++) fvalue_dst_[i] = fvalue_src_[i];
					m_value = fvalue_dst_;
					}
					break;

				case TypeDoubleArray:
					{
					double *dvalue_src_ = (double *) other.m_value;
					double *dvalue_dst_ = new double[m_n_values];
					for(int i = 0 ; i < m_n_values ; i++) dvalue_dst_[i] = dvalue_src_[i];
					m_value = dvalue_dst_;
					}
					break;

				case TypeNothing:
				default:
					break;
				}
			}

			// Delete the allocated memory
			void			cleanup()
			{
				delete[] m_name;
				delete[] m_help;

				switch(m_type)
				{
				case TypeBool:
					delete (bool*)m_value;
					break;

				case TypeInt:
					delete (int*)m_value;
					break;

				case TypeFloat:
					delete (float*)m_value;
					break;

				case TypeDouble:
					delete (double*)m_value;
					break;

				case TypeString:
					delete[] (char*)m_value;
					break;

				case TypeIntArray:
					delete[] (int*)m_value;
					break;

				case TypeFloatArray:
					delete[] (float*)m_value;
					break;

				case TypeDoubleArray:
					delete[] (double*)m_value;
					break;

				case TypeNothing:
				default:
					break;
				}
				m_value = 0;
				m_n_values = 0;
			}

			/////////////////////////////////////////////////
			// Attributes

			Type 			m_type;
			char* 			m_name;
			char* 			m_help;
			void*			m_value;
			int			m_n_values;
		};
		//////////////////////////////////////////////////////////////////////

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

