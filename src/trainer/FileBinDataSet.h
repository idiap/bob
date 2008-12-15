#ifndef FILE_BIN_DATA_SET_INC
#define FILE_BIN_DATA_SET_INC

#include "DataSet.h"
#include "File.h"

namespace Torch
{
   	class Tensor;

   	class FileBinDataSet : public DataSet
	{
	public:
		///
		FileBinDataSet();

		void setData(char *filename, Tensor::Type , Tensor::Type, int width, int height);

		virtual Tensor* getExample(long);
		virtual Tensor& operator()(long);
		virtual Tensor* getTarget(long);

		///
		virtual ~FileBinDataSet();

		Tensor *examples;
		Tensor *target;

        	ShortTensor* short_example;
		ShortTensor* short_target;

	protected:
		Tensor *current_example;
		Tensor *current_target;

		ShortTensor* short_currentT;
	};

}

#endif
