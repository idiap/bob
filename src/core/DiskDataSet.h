#ifndef DISK_DATA_SET_INC
#define DISK_DATA_SET_INC

#include "DataSet.h"
#include "TensorFile.h"

namespace Torch
{
	class DiskDataSet : public DataSet
	{
	public:

		// Constructor
		DiskDataSet(Tensor::Type example_type_ = Tensor::Double);

		// Destructor
		virtual ~DiskDataSet();

		// Do _not_ try to modify them, they are readonly!
		virtual Tensor* getExample(long index);
		virtual Tensor&	operator()(long index);

		// Access targets
		virtual Tensor* getTarget(long index);
		virtual void	setTarget(long index, Tensor* target);

		// Load a tensor data file
		bool		load(const char* file_name);

	private:

		/// Delete the allocated tensors
		void 		cleanup();

		//////////////////////////////////////////////////////
		// Attributes

		Tensor*		m_buffer;	// Buffer for the current example
		Tensor**	m_targets;	// Array of pointers to external tensors

		// Files to load tensors from
		TensorFile**	m_files;
		int		m_n_files;
	};
}

#endif

