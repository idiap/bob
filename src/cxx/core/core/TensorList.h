#ifndef _TENSOR_LIST_INC
#define _TENSOR_LIST_INC

#include "core/DataSet.h"
#include "core/MemoryDataSet.h"
#include "core/TensorFile.h"
#include "core/FileListCmdOption.h"
#include "core/Tensor.h"

namespace Torch
{
    class TensorList //: public DataSet
	{
	public:
	TensorList();
	~TensorList();
	bool process(FileListCmdOption *tensorList_files,ShortTensor *target,Tensor::Type mType);
	DataSet *getOutput();

	private:
        int n_files;
        MemoryDataSet *m_data;
        int n_examples;
        Tensor::Type m_type;
        int dimension;
        int d1,d2,d3,d4;
         Tensor *tensor;
	};
}

#endif
