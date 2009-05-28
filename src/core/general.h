#ifndef GENERAL_INC
#define GENERAL_INC

#define _USE_MATH_DEFINES

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <limits.h>
#include <stdarg.h>
#include <time.h>
#include <float.h>

// Old systems need that to define FLT_MAX and DBL_MAX
#ifndef DBL_MAX
#include <values.h>
#endif

namespace Torch {

//-----------------------------------

/// Print an error message. The program will exit.
void fatalerror(const char* fmt, ...);

/// Print an error message. The program will NOT exit.
void error(const char* fmt, ...);

/// Print a warning message.
void warning(const char* fmt, ...);

/// Print a message.
void message(const char* fmt, ...);

/// Like printf.
void print(const char* fmt, ...);

/// Functions used for sorting arrays with <qsort>
int compare_floats(const void* a, const void* b);
int compare_doubles(const void* a, const void* b);

//-----------------------------------

#ifndef min
/// The min function
#define	min(a,b) ((a) > (b) ? (b) : (a))
#endif

#ifndef max
/// The max function
#define	max(a,b) ((a) > (b) ? (a) : (b))
#endif

#ifndef getInRange
/// The getInRange function (force some value to be in the [m, M] range)
#define	getInRange(v,m,M) ((v) < (m) ? (m) : ((v) > (M) ? (M) : (v)))
#endif

#ifndef isInRange
/// The isInRange function (checks if some value is in the [m, M] range)
#define	isInRange(v,m,M) ((v) >= (m) && (v) <= (M))
#endif

#ifndef isIndex
/// The isIndex function checks if some value is an index
#define	isIndex(v,N) ((v) >= 0 && (v) < (N))
#endif

#define IS_NEAR(var, value, delta) ((var >= (value - delta)) && (var <= (value + delta)))

#define FixI(v) (int) (v+0.5)

/// Macros to check for errors (fatal -> force exit, error -> return false)
#define CHECK_FATAL(expression)					\
{								\
	const bool condition = (expression);			\
	if (condition == false)					\
	{							\
		fatalerror("Error: in file [%s] at line [%d]!\n",	\
			__FILE__, __LINE__);			\
	}							\
}

#define CHECK_ERROR(expression)					\
{								\
	const bool condition = (expression);			\
	if (condition == false)					\
	{							\
		message("Error: in file [%s] at line [%d]!\n",	\
			__FILE__, __LINE__);			\
		return false;					\
	}							\
}

	////////////////////////////////////////////////////////////////////////////////////
	// Memory management
	////////////////////////////////////////////////////////////////////////////////////



	// Memory management
	namespace Private
	{
		template <typename T>
		class MemManager
		{
		public:
			// Access to the single instance
			static MemManager<T>&	getInstance()
			{
				static MemManager<T> instance;
				return instance;
			}

			// Destructor
			~MemManager()
			{
				deallocate();
			}

			// Automatic deallocation
			T*		manage(T* data)
			{
				add(m_n_objects, m_size_objects, m_objects, data);
				return data;
			}
			T*		manage_array(T* data)
			{
				add(m_n_object_arrays1D, m_size_object_arrays1D, m_object_arrays1D, data);
				return data;
			}
			T**		manage_array(T** data)
			{
				add(m_n_object_arrays2D, m_size_object_arrays2D, m_object_arrays2D, data);
				return data;
			}
			T***		manage_array(T*** data)
			{
				add(m_n_object_arrays3D, m_size_object_arrays3D, m_object_arrays3D, data);
				return data;
			}

			// Except some from automatic deallocation
			void		unmanage(T* data)
			{
				remove(m_n_objects, m_objects, data);
				remove(m_n_object_arrays1D, m_object_arrays1D, data);
			}
			void		unmanage(T** data)
			{
				remove(m_n_object_arrays2D, m_object_arrays2D, data);
			}
			void		unmanage(T*** data)
			{
				remove(m_n_object_arrays3D, m_object_arrays3D, data);
			}

		private:

			// Constructor
			MemManager()
				:	m_objects(0),
					m_object_arrays1D(0),
					m_object_arrays2D(0),
					m_object_arrays3D(0),

					m_n_objects(0), m_size_objects(0),
					m_n_object_arrays1D(0), m_size_object_arrays1D(0),
					m_n_object_arrays2D(0), m_size_object_arrays2D(0),
					m_n_object_arrays3D(0), m_size_object_arrays3D(0)
			{
			}

			// Copy constructor and assignment operator - declared & !defined
			MemManager(const MemManager&);
			MemManager& operator=(const MemManager&);

			// Deallocate stored tensors
			void			deallocate()
			{
				for (int i = 0; i < m_n_objects; i ++)
				{
					delete m_objects[i];
				}
				for (int i = 0; i < m_n_object_arrays1D; i ++)
				{
					delete[] m_object_arrays1D[i];
				}
				for (int i = 0; i < m_n_object_arrays2D; i ++)
				{
					delete[] m_object_arrays2D[i];
				}
				for (int i = 0; i < m_n_object_arrays3D; i ++)
				{
					delete[] m_object_arrays3D[i];
				}

				// Deallocate the pointers
				delete[] m_objects;
				delete[] m_object_arrays1D;
				delete[] m_object_arrays2D;
				delete[] m_object_arrays3D;
			}

			// Store pointers
			template <typename U>
			static void		add(int& n, int& size, U*& pointers, U data)
			{
				// Check if this pointer was already added
				int i = 0;
				for ( ; i < n && pointers[i] != data; i ++)
					;
				if (i >= n)	// It wasn't!
				{
					if (i >= size)
					{
						// Need to make room for it
						U* tmp_pointers = new U[size * 2 + 1];
						for (int j = 0; j < n; j ++)
						{
							tmp_pointers[j] = pointers[j];
						}
						delete[] pointers;
						pointers = tmp_pointers;
						size = size * 2 + 1;
					}

					pointers[n ++] = data;
				}
			}

			// Remove some stored pointer
			template <typename U>
			static void		remove(int& n, U*& pointers, U data)
			{
				// Check if this pointer is indeed already added
				int i = 0;
				for ( ; i < n && pointers[i] != data; i ++)
					;
				if (i < n)
				{
					// Move the pointers after this one to the left with one position
					while (i + 1 < n)
					{
						pointers[i] = pointers[i + 1];
						i ++;
					}
					n --;
				}
			}

		private:

			//////////////////////////////////////////////////////
			// Attributes

			T**		m_objects;
			T**		m_object_arrays1D;
			T***		m_object_arrays2D;
			T****		m_object_arrays3D;

			int		m_n_objects, m_size_objects;
			int		m_n_object_arrays1D, m_size_object_arrays1D;
			int		m_n_object_arrays2D, m_size_object_arrays2D;
			int		m_n_object_arrays3D, m_size_object_arrays3D;
		};
	}

	// Automatic deallocation
	template <typename T>
	T*		manage(T* data)
	{
		return Private::MemManager<T>::getInstance().manage(data);
	}
	template <typename T>
	T*		manage_array(T* data)
	{
		return Private::MemManager<T>::getInstance().manage_array(data);
	}
	template <typename T>
	T**		manage_array(T** data)
	{
		return Private::MemManager<T>::getInstance().manage_array(data);
	}
	template <typename T>
	T***		manage_array(T*** data)
	{
		return Private::MemManager<T>::getInstance().manage_array(data);
	}

	// Except some from automatic deallocation
	template <typename T>
	void		unmanage(T* data)
	{
		return Private::MemManager<T>::getInstance().unmanage(data);
	}
	template <typename T>
	void		unmanage(T** data)
	{
		return Private::MemManager<T>::getInstance().unmanage(data);
	}
	template <typename T>
	void		unmanage(T*** data)
	{
		return Private::MemManager<T>::getInstance().unmanage(data);
	}
}

#endif
