#ifndef DISK_DATA_SET_INC
#define DISK_DATA_SET_INC

#include "DataSet.h"

namespace Torch
{
	class DiskDataSet : public DataSet
	{
	public:

		// Constructor
		DiskDataSet(	Tensor::Type example_type_ = Tensor::Double,
				Tensor::Type target_type_ = Tensor::Short);

		// Destructor
		virtual ~DiskDataSet();

		// Access examples
		virtual Tensor* getExample(long index);
		virtual Tensor&	operator()(long index);

		// Access targets
		virtual Tensor* getTarget(long index) = 0;
		virtual void	setTarget(long index, Tensor* target) = 0;

		// Load a data file
		bool		load(consts char* file_name);

	private:

		/// Delete the allocated tensors
		void 		cleanup();

		//////////////////////////////////////////////////////
		// Attributes

		// Cache
		static const int CacheSize = 1024;
		MemoryDataSet	m_cache;
		int		m_cache_index[CacheSize];

		// File loaded
		TensorFile**	m_files;
		int		m_n_files;
	};
}

#endif
