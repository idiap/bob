#include "VariableCollector.h"
#include "File.h"

namespace Torch
{
		// Constructor
		VariableCollector::VariableCollector()
			:	m_variables(0),
				m_size(0), m_capacity(0), m_resizeDelta(12)
		{
		}

		// Destructor
		VariableCollector::~VariableCollector()
		{
			cleanup();
		}

		///////////////////////////////////////////////////////////
		// Variable management functions - adding new ones

		bool	VariableCollector::addB(const char* name, bool init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, init_value);
			return true;
		}
		bool	VariableCollector::addI(const char* name, int init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, init_value);
			return true;
		}
		bool	VariableCollector::addF(const char* name, float init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, init_value);
			return true;
		}
		bool	VariableCollector::addD(const char* name, double init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, init_value);
			return true;
		}
		bool	VariableCollector::addS(const char* name, const char* init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, init_value);
			return true;
		}
		bool	VariableCollector::addIarray(const char* name, int n_values, int init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, n_values, init_value);
			return true;
		}
		bool	VariableCollector::addFarray(const char* name, int n_values, float init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, n_values, init_value);
			return true;
		}
		bool	VariableCollector::addDarray(const char* name, int n_values, double init_value, const char* help)
		{
			if (search(name) >= 0)
			{
				return false; 	// The name is already taken
			}
			resize();
			m_variables[m_size ++].init(name, help, n_values, init_value);
			return true;
		}

		///////////////////////////////////////////////////////////
		// Variable management functions - changing their values

		bool	VariableCollector::setB(const char* name, bool new_value)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeBool)
			{
				return false;
			}
			*((bool*)m_variables[index].m_value) = new_value;
			return true;
		}
		bool	VariableCollector::setI(const char* name, int new_value)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeInt)
			{
				return false;
			}
			*((int*)m_variables[index].m_value) = new_value;
			return true;
		}
		bool	VariableCollector::setF(const char* name, float new_value)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeFloat)
			{
				return false;
			}
			*((float*)m_variables[index].m_value) = new_value;
			return true;
		}
		bool	VariableCollector::setD(const char* name, double new_value)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeDouble)
			{
				return false;
			}
			*((double*)m_variables[index].m_value) = new_value;
			return true;
		}
		bool	VariableCollector::setS(const char* name, const char* new_value)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeString)
			{
				return false;
			}
			delete[] (char*)m_variables[index].m_value;
			m_variables[index].m_value = new char[strlen(new_value) + 1];
			strcpy((char*)m_variables[index].m_value, new_value);
			return true;
		}
		bool	VariableCollector::setIarray(const char* name, int n_values)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeIntArray)
			{
				return false;
			}
			if(m_variables[index].m_value != NULL) delete[] (int*)m_variables[index].m_value;
			m_variables[index].m_value = new int[n_values];
			m_variables[index].m_n_values = n_values;
			return true;
		}
		bool	VariableCollector::setFarray(const char* name, int n_values)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeFloatArray)
			{
				return false;
			}
			if(m_variables[index].m_value != NULL) delete[] (float*)m_variables[index].m_value;
			m_variables[index].m_value = new float[n_values];
			m_variables[index].m_n_values = n_values;
			return true;
		}
		bool	VariableCollector::setDarray(const char* name, int n_values)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeDoubleArray)
			{
				return false;
			}
			if(m_variables[index].m_value != NULL) delete[] (double*)m_variables[index].m_value;
			m_variables[index].m_value = new double[n_values];
			m_variables[index].m_n_values = n_values;
			return true;
		}

		///////////////////////////////////////////////////////////
		// Variable management functions - retrieving alues

		bool	VariableCollector::getB(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeBool)
			{
				setOK(ok, false);
				return false;
			}

			setOK(ok, true);
			return *((bool*)m_variables[index].m_value);
		}
		int	VariableCollector::getI(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeInt)
			{
				setOK(ok, false);
				return 0;
			}

			setOK(ok, true);
			return *((int*)m_variables[index].m_value);
		}
		float	VariableCollector::getF(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeFloat)
			{
				setOK(ok, false);
				return 0.0f;
			}

			setOK(ok, true);
			return *((float*)m_variables[index].m_value);
		}
		double	VariableCollector::getD(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeDouble)
			{
				setOK(ok, false);
				return 0.0;
			}

			setOK(ok, true);
			return *((double*)m_variables[index].m_value);
		}
		const char*	VariableCollector::getS(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeString)
			{
				setOK(ok, false);
				return 0;
			}

			setOK(ok, true);
			return (const char*)m_variables[index].m_value;
		}
		int*	VariableCollector::getIarray(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeIntArray)
			{
				setOK(ok, false);
				return 0;
			}

			setOK(ok, true);
			return (int*)m_variables[index].m_value;
		}
		float*	VariableCollector::getFarray(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeFloatArray)
			{
				setOK(ok, false);
				return 0;
			}

			setOK(ok, true);
			return (float*)m_variables[index].m_value;
		}
		double*	VariableCollector::getDarray(const char* name, bool* ok)
		{
			const int index = search(name);
			if (index < 0 || m_variables[index].m_type != Variable::TypeDoubleArray)
			{
				setOK(ok, false);
				return 0;
			}

			setOK(ok, true);
			return (double*)m_variables[index].m_value;
		}

		void VariableCollector::print()
		{
			for (int i = 0; i < m_size; i ++)
			{
				Torch::print("PARAMETER [%d] \"%s\" ", i, m_variables[i].m_name);

				switch(m_variables[i].m_type)
				{
				case Variable::TypeInt:
					Torch::print("<INT>"); 
					break;
				case Variable::TypeFloat:
					Torch::print("<FLOAT>"); 
					break;
				case Variable::TypeDouble:
					Torch::print("<DOUBLE>"); 
					break;
				case Variable::TypeIntArray:
					Torch::print("<INT*>"); 
					break;
				case Variable::TypeFloatArray:
					Torch::print("<FLOAT*>"); 
					break;
				case Variable::TypeDoubleArray:
					Torch::print("<DOUBLE*>"); 
					break;
				default:
					Torch::print("<?>");
					break;
				}
				Torch::print(" (%d) { ", m_variables[i].m_n_values);
				switch(m_variables[i].m_type)
				{
				case Variable::TypeInt:
					Torch::print("%d ", *((int *) m_variables[i].m_value));
					break;
				case Variable::TypeFloat:
					Torch::print("%g ", *((float *) m_variables[i].m_value));
					break;
				case Variable::TypeDouble:
					Torch::print("%g ", *((double *) m_variables[i].m_value));
					break;
				case Variable::TypeIntArray:
					if(m_variables[i].m_n_values == 0) Torch::print("empty ");
					else
					{
						int *ivalues_ = (int *) m_variables[i].m_value;
						for(int j = 0 ; j < m_variables[i].m_n_values ; j++)
							Torch::print("%d ", ivalues_[j]);
					}
					break;
				case Variable::TypeFloatArray:
					if(m_variables[i].m_n_values == 0) Torch::print("empty ");
					else
					{
						float *fvalues_ = (float *) m_variables[i].m_value;
						for(int j = 0 ; j < m_variables[i].m_n_values ; j++)
							Torch::print("%g ", fvalues_[j]);
					}
					break;
				case Variable::TypeDoubleArray:
					if(m_variables[i].m_n_values == 0) Torch::print("empty ");
					else
					{
						double *dvalues_ = (double *) m_variables[i].m_value;
						for(int j = 0 ; j < m_variables[i].m_n_values ; j++)
							Torch::print("%g ", dvalues_[j]);
					}
					break;
				default:
					Torch::print("?");
					break;
				}
				Torch::print("}");
				if(strlen(m_variables[i].m_help) <= 1)
					Torch::print(" \"no description\"\n");
				else Torch::print(" \"%s\"\n", m_variables[i].m_help);
			}
		}

		// Loading/Saving the content from files (\emph{not the options}) - overriden
		bool VariableCollector::loadFile(File& file)
		{

		   	// Loads the parameters by assuming they have been prepared with method.
			// So we need to make sure that the variable exists
		   
		  	int n_params_;
			if (file.taggedRead(&n_params_, sizeof(int), 1, "N_PARAMS") != 1)
			{
				Torch::message("VariableCollector::load - failed to read the number of parameters!\n");
				return false;
			}
			if(n_params_ != m_size)
			{
				Torch::message("VariableCollector::load - incorrect number of parameters!\n");
				return false;
			}

			for (int i = 0; i < m_size; i ++)
			{
				/*
			   	int type_;
				if (file.taggedRead(&type_, sizeof(int), 1, "TYPE") != 1)
				{
					Torch::message("VariableCollector::load - failed to load the type of the parameter <%s>!\n", m_variables[i].m_name);
					Torch::print("> %d\n", type_);
					return false;
				}
				Variable::Type type__ = (Variable::Type) type_;
				if(type__ != m_variables[i].m_type)
				{
					Torch::message("VariableCollector::load - incorrect type of the parameter <%s>!\n", m_variables[i].m_name);
					return false;
				}
				*/
				//
				char *str_ = new char [strlen(m_variables[i].m_name)+1];
				int n_values_;
				switch(m_variables[i].m_type)
				{
				case Variable::TypeInt:
					if (file.taggedRead((int *) m_variables[i].m_value, sizeof(int), 1, m_variables[i].m_name) != 1)
					{
						Torch::message("VariableCollector::load - failed to read INT <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeFloat:
					if (file.taggedRead((float *) m_variables[i].m_value, sizeof(float), 1, m_variables[i].m_name) != 1)
					{
						Torch::message("VariableCollector::load - failed to read FLOAT <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeDouble:
					if (file.taggedRead((double *) m_variables[i].m_value, sizeof(double), 1, m_variables[i].m_name) != 1)
					{
						Torch::message("VariableCollector::load - failed to read DOUBLE <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeIntArray:
					sprintf(str_, "N_%s", m_variables[i].m_name); 
					if (file.taggedRead(&n_values_, sizeof(int), 1, str_) != 1)
					{
						Torch::message("VariableCollector::load - failed to read the number of values for the INT* <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					setIarray(m_variables[i].m_name, n_values_);
					//if(m_variables[i].m_value != NULL) delete[] (int*)m_variables[i].m_value;
					//m_variables[i].m_value = new int[n_values_];
					//m_variables[i].m_n_values = n_values_;
					if (file.taggedRead((int *) m_variables[i].m_value, sizeof(int), m_variables[i].m_n_values, m_variables[i].m_name) != m_variables[i].m_n_values)
					{
						Torch::message("VariableCollector::load - failed to read <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeFloatArray:
					sprintf(str_, "N_%s", m_variables[i].m_name); 
					if (file.taggedRead(&n_values_, sizeof(int), 1, str_) != 1)
					{
						Torch::message("VariableCollector::load - failed to read the number of values for the FLOAT* <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					setFarray(m_variables[i].m_name, n_values_);
					//if(m_variables[i].m_value != NULL) delete[] (float*)m_variables[i].m_value;
					//m_variables[i].m_value = new float[n_values_];
					//m_variables[i].m_n_values = n_values_;
					if (file.taggedRead((float *) m_variables[i].m_value, sizeof(float), m_variables[i].m_n_values, m_variables[i].m_name) != m_variables[i].m_n_values)
					{
						Torch::message("VariableCollector::load - failed to read <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeDoubleArray:
					sprintf(str_, "N_%s", m_variables[i].m_name); 
					if (file.taggedRead(&n_values_, sizeof(int), 1, str_) != 1)
					{
						Torch::message("VariableCollector::load - failed to read the number of values for the DOUBLE* <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					setDarray(m_variables[i].m_name, n_values_);
					//if(m_variables[i].m_value != NULL) delete[] (double*)m_variables[i].m_value;
					//m_variables[i].m_value = new double[n_values_];
					//m_variables[i].m_n_values = n_values_;
					if (file.taggedRead((double *) m_variables[i].m_value, sizeof(double), m_variables[i].m_n_values, m_variables[i].m_name) != m_variables[i].m_n_values)
					{
						Torch::message("VariableCollector::load - failed to read <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				}
				delete [] str_;
			}
			// OK
			return true;
		}
		
		bool VariableCollector::saveFile(File& file) const
		{
			if (file.taggedWrite(&m_size, sizeof(int), 1, "N_PARAMS") != 1)
			{
				Torch::message("VariableCollector::save - failed to write the number of parameters!\n");
				return false;
			}
			for (int i = 0; i < m_size; i ++)
			{
			   	/*
			   	int type_ = (int) m_variables[i].m_type;
				Torch::print("> %d\n", type_);
				if (file.taggedWrite(&type_, sizeof(int), 1, "TYPE") != 1)
				{
					Torch::message("VariableCollector::save - failed to write the type of the parameter <%s>!\n", m_variables[i].m_name);
					return false;
				}
				*/
				char *str_ = new char [strlen(m_variables[i].m_name)+1];
				switch(m_variables[i].m_type)
				{
				case Variable::TypeInt:
					if (file.taggedWrite((int *) m_variables[i].m_value, sizeof(int), 1, m_variables[i].m_name) != 1)
					{
						Torch::message("VariableCollector::save - failed to write <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeFloat:
					if (file.taggedWrite((float *) m_variables[i].m_value, sizeof(float), 1, m_variables[i].m_name) != 1)
					{
						Torch::message("VariableCollector::save - failed to write <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeDouble:
					if (file.taggedWrite((double *) m_variables[i].m_value, sizeof(double), 1, m_variables[i].m_name) != 1)
					{
						Torch::message("VariableCollector::save - failed to write <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeIntArray:
					sprintf(str_, "N_%s", m_variables[i].m_name); 
					if (file.taggedWrite(&m_variables[i].m_n_values, sizeof(int), 1, str_) != 1)
					{
						Torch::message("VariableCollector::save - failed to write the number of values for the <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					if (file.taggedWrite((int *) m_variables[i].m_value, sizeof(int), m_variables[i].m_n_values, m_variables[i].m_name) != m_variables[i].m_n_values)
					{
						Torch::message("VariableCollector::save - failed to write <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeFloatArray:
					sprintf(str_, "N_%s", m_variables[i].m_name); 
					if (file.taggedWrite(&m_variables[i].m_n_values, sizeof(int), 1, str_) != 1)
					{
						Torch::message("VariableCollector::save - failed to write the number of values for the <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					if (file.taggedWrite((float *) m_variables[i].m_value, sizeof(float), m_variables[i].m_n_values, m_variables[i].m_name) != m_variables[i].m_n_values)
					{
						Torch::message("VariableCollector::save - failed to write <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				case Variable::TypeDoubleArray:
					sprintf(str_, "N_%s", m_variables[i].m_name); 
					if (file.taggedWrite(&m_variables[i].m_n_values, sizeof(int), 1, str_) != 1)
					{
						Torch::message("VariableCollector::save - failed to write the number of values for the <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					if (file.taggedWrite((double *) m_variables[i].m_value, sizeof(double), m_variables[i].m_n_values, m_variables[i].m_name) != m_variables[i].m_n_values)
					{
						Torch::message("VariableCollector::save - failed to write <%s> field!\n", m_variables[i].m_name);
						return false;
					}
					break;
				}
				delete []str_;
			}

			// OK
			return true;
		}

		// Resize the variable array as to accomodate new variables
		void	VariableCollector::resize()
		{
			// Check if the variable array need resize
			if (m_size < m_capacity)
			{
				return;
			}

			// Create a new array - as a copy
			Variable* new_variables = new Variable[m_capacity + m_resizeDelta];
			for (int i = 0; i < m_size; i ++)
			{
				new_variables[i].init(m_variables[i]);
			}

			// Delete the old one and point to the new one
			cleanup();
			m_capacity += m_resizeDelta;
			m_variables = new_variables;
		}

		// Deallocate allocated variables
		void	VariableCollector::cleanup()
		{
		   	/* Warning !!

				Are void* pointers in the structure deleted as well ?
			*/

			delete[] m_variables;
		}

		// Search a variable given its name and returns its index (or <0 if not found)
		int	VariableCollector::search(const char* name) const
		{
			for (int i = 0; i < m_size; i ++)
			{
				if (strcmp(m_variables[i].m_name, name) == 0)
				{
					return i;
				}
			}
			return -1;
		}

		// Sets an <ok> flag to the given value
		void	VariableCollector::setOK(bool* ok, bool value) const
		{
			if (ok != 0)
			{
				*ok = value;
			}
		}

}

