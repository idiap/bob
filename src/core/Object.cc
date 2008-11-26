#include "Object.h"
#include "File.h"

namespace Torch
{
	///////////////////////////////////////////////////////////
	// Implements to option management for <Object> type
	///////////////////////////////////////////////////////////
	class OptionImpl
	{
	public:
		///////////////////////////////////////////////////////////
		// Option - implements different type of options

		struct Option
		{
			enum Type
			{
				TypeNothing,
				TypeBool,
				TypeInt,
				TypeFloat,
				TypeDouble
			};

			// Constructor - default
			Option()
				:	m_type(TypeNothing),
					m_name(0), m_help(0),
					m_value(0)
			{
			}

			// Destructor
			~Option()
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
			}

			// Initialize the object - from another
			void			init(const Option& other)
			{
				cleanup();

				m_name = new char[strlen(other.m_name) + 1];
				m_help = new char[strlen(other.m_help) + 1];
				strcpy(m_name, other.m_name);
				strcpy(m_help, other.m_help);

				m_type = other.m_type;

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

				case TypeNothing:
				default:
					break;
				}
				m_value = 0;
			}

			/////////////////////////////////////////////////
			// Attributes

			Type 			m_type;
			char* 			m_name;
			char* 			m_help;
			void*			m_value;
		};
		//////////////////////////////////////////////////////////////////////

		// Constructor
		OptionImpl()
			:	m_options(0),
				m_size(0), m_capacity(0), m_resizeDelta(12)
		{
		}

		// Destructor
		~OptionImpl()
		{
			cleanup();
		}

		///////////////////////////////////////////////////////////
		// Option management functions - adding new ones

		bool				addBOption(const char* name, bool init_value, const char* help = "")
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_options[m_size ++].init(name, help, init_value);
			return true;
		}
		bool				addIOption(const char* name, int init_value, const char* help = "")
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_options[m_size ++].init(name, help, init_value);
			return true;
		}
		bool				addFOption(const char* name, float init_value, const char* help = "")
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_options[m_size ++].init(name, help, init_value);
			return true;
		}
		bool				addDOption(const char* name, double init_value, const char* help = "")
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_options[m_size ++].init(name, help, init_value);
			return true;
		}

		///////////////////////////////////////////////////////////
		// Option management functions - changing their values

		bool				setBOption(const char* name, bool new_value)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeBool)
			{
				return false;
			}
			*((bool*)m_options[index].m_value) = new_value;
			return true;
		}
		bool				setIOption(const char* name, int new_value)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeInt)
			{
				return false;
			}
			*((int*)m_options[index].m_value) = new_value;
			return true;
		}
		bool				setFOption(const char* name, float new_value)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeFloat)
			{
				return false;
			}
			*((float*)m_options[index].m_value) = new_value;
			return true;
		}
		bool				setDOption(const char* name, double new_value)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeDouble)
			{
				return false;
			}
			*((double*)m_options[index].m_value) = new_value;
			return true;
		}

		///////////////////////////////////////////////////////////
		// Option management functions - retrieving alues

		bool				getBOption(const char* name, bool* ok = 0)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeBool)
			{
				setOK(ok, false);
				return false;
			}

			setOK(ok, true);
			return *((bool*)m_options[index].m_value);
		}
		int				getIOption(const char* name, bool* ok = 0)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeInt)
			{
				setOK(ok, false);
				return 0;
			}

			setOK(ok, true);
			return *((int*)m_options[index].m_value);
		}
		float				getFOption(const char* name, bool* ok = 0)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeFloat)
			{
				setOK(ok, false);
				return 0.0f;
			}

			setOK(ok, true);
			return *((float*)m_options[index].m_value);
		}
		double				getDOption(const char* name, bool* ok = 0)
		{
			const int index = search(name);
			if (index < 0 || m_options[index].m_type != Option::TypeDouble)
			{
				setOK(ok, false);
				return 0.0;
			}

			setOK(ok, true);
			return *((double*)m_options[index].m_value);
		}

	private:

		//////////////////////////////////////////////////////////////////////

		// Resize the option array as to accomodate new options
		void				resize()
		{
			// Check if the option array need resize
			if (m_size < m_capacity)
			{
				return;
			}

			// Create a new array - as a copy
			Option* new_options = new Option[m_capacity + m_resizeDelta];
			for (int i = 0; i < m_size; i ++)
			{
				new_options[i].init(m_options[i]);
			}

			// Delete the old one and point to the new one
			cleanup();
			m_capacity += m_resizeDelta;
			m_options = new_options;
		}

		// Deallocate allocated options
		void				cleanup()
		{
			delete[] m_options;
		}

		// Search an option given its name and returns its index (or <0 if not found)
		int				search(const char* name) const
		{
			for (int i = 0; i < m_size; i ++)
			{
				if (strcmp(m_options[i].m_name, name) == 0)
				{
					return i;
				}
			}
			return -1;
		}

		// Sets an <ok> flag to the given value
		void				setOK(bool* ok, bool value) const
		{
			if (ok != 0)
			{
				*ok = value;
			}
		}

	private:

		///////////////////////////////////////////////////////////
		// Attributes

		Option*				m_options;
		int				m_size;			// No of options used
		int				m_capacity;		// No of options allocated (>= m_size)
		int				m_resizeDelta;		// No of options to pre-allocate
	};
}

//////////////////////////////////////////////////////////////////////////////////////

// Constructor
Torch::Object::Object()
	: 	m_optionImpl(new OptionImpl())
{
	addBOption("verbose", false, "verbose flag");
}

// Destructor
Torch::Object::~Object()
{
	delete m_optionImpl;
}

//////////////////////////////////////////////////////////////////////////////////////
// Option management functions - just wrappers over OptionImpl

// addXOption
bool Torch::Object::addBOption(const char* name, bool init_value, const char* help)
{
	return m_optionImpl->addBOption(name, init_value, help);
}
bool Torch::Object::addIOption(const char* name, int init_value, const char* help)
{
	return m_optionImpl->addIOption(name, init_value, help);
}
bool Torch::Object::addFOption(const char* name, float init_value, const char* help)
{
	return m_optionImpl->addFOption(name, init_value, help);
}
bool Torch::Object::addDOption(const char* name, double init_value, const char* help)
{
	return m_optionImpl->addDOption(name, init_value, help);
}

// setXOption
bool Torch::Object::setBOption(const char* name, bool new_value)
{
	if (m_optionImpl->setBOption(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Torch::Object::setIOption(const char* name, int new_value)
{
	if (m_optionImpl->setIOption(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Torch::Object::setFOption(const char* name, float new_value)
{
	if (m_optionImpl->setFOption(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}
bool Torch::Object::setDOption(const char* name, double new_value)
{
	if (m_optionImpl->setDOption(name, new_value) == true)
	{
		optionChanged(name);
		return true;
	}
	return false;
}

// getXOption
bool Torch::Object::getBOption(const char* name, bool* ok)
{
	return m_optionImpl->getBOption(name, ok);
}
int Torch::Object::getIOption(const char* name, bool* ok)
{
	return m_optionImpl->getIOption(name, ok);
}
float Torch::Object::getFOption(const char* name, bool* ok)
{
	return m_optionImpl->getFOption(name, ok);
}
double Torch::Object::getDOption(const char* name, bool* ok)
{
	return m_optionImpl->getDOption(name, ok);
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options})

bool Torch::Object::load(const char* filename)
{
	File file;
	return file.open(filename, "r") && loadFile(file);
}

bool Torch::Object::save(const char* filename) const
{
	File file;
	return file.open(filename, "w+") && saveFile(file);
}

bool Torch::Object::loadFile(File& file)
{
	// Nothing to do!
	return true;
}

bool Torch::Object::saveFile(File& file) const
{
	// Nothing to do!
	return true;
}

///////////////////////////////////////////////////////////
