#ifndef DISK_DATA_SET_INC
#define DISK_DATA_SET_INC

#include "DataSet.h"
#include "TargetSet.h"
#include "TensorFile.h"
//#include "MemoryDataSet.h"
//#include "TensorFile.h"

namespace Torch
{
	class DiskDataSet : public DataSet
	{
		public:

			// Constructor
			DiskDataSet(Tensor::Type example_type_ = Tensor::Double);

			// Destructor
			virtual ~DiskDataSet();

			// Unable to modify tensors, do _not_ use !
			virtual Tensor* getExample(long index);
			virtual Tensor&	operator()(long index);

			// Access targets
			virtual Tensor* getTarget(long index);
			virtual void	setTarget(long index, Tensor* target);

			// Load a data file
			bool		load(const char* file_name);

		private:
			/// Delete the allocated tensors
			void 		cleanup();

			//////////////////////////////////////////////////////
			// Attributes

			Tensor*		current;	// the only tensor that is in memory
							// at any given time

			TargetSet* 	targetset;	// the set of all targets. need to
							// change since when we load.


			//////////////////////////////////////////////////////
			// Class IndexMapper
			//
			// \brief 	map  [ global index ] : [ File*, Offset ] 
			class IndexMapper {
				private:
					//////////////////////////////////////////////////////////////
					// Attributes 

					TensorFile**	file_list;	// array to where the files are

					long*		map_index;	// index to the files and offset 

					long		m_file_cnt;	// keep track of # of files	

					long		m_top_num;	// the highest index

					long		m_limit;	// actual size

					// starting limit of mapper
					static const long M_START_LIMIT = 1000; 
					static const long M_RESCALE_FAC	= 2;

					//////////////////////////////////////////////////////////////
					// Methodes

					// used to non-destructive resize
					void 	resize(long size);

					// clean
					void 	clean();	// used to dealloc all


				public:
					//////////////////////////////////////////////////////////////
					// Methodes

					IndexMapper();

					~IndexMapper();

					// get a pointer to the correct file
					TensorFile* 	getFile(long global_index);

					// get the offset in the the file
					long		getOffset(long global_index);

					// store match of [ global index ] : [ File*, Offset ] 
					bool		map(const char* filename);
			};

			IndexMapper*	map;		// handle mappin between global index
	};
}

#endif

