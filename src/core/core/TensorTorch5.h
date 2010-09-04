#ifndef TORCH5SPRO_TENSOR_TORCH5_H
#define TORCH5SPRO_TENSOR_TORCH5_H

#include "core/general.h"

#define METHOD_NOT_IMPLEMENTED { warning("Not implemented !\n"); }

namespace Torch
{
	/////////////////////////////////////////////////////////////////
	// Tensor
	/////////////////////////////////////////////////////////////////
	class Tensor
	{
	public:

		/////////////////////////////////////////////////////////////
		/// Supported tensor types
		enum Type
		{
			Char,
			Short,
			Int,
			Long,
			Float,
			Double
		};
		/////////////////////////////////////////////////////////////

	private:

		short datatype;

	protected:

		mutable bool m_isReference;

	public:

		Tensor(short datatype_) : datatype(datatype_) { m_isReference = false; };
		Tensor(Tensor::Type datatype_) : datatype(datatype_ ) { m_isReference = false; };

		Tensor::Type getDatatype() const { return (Tensor::Type)datatype; };


		//--- pure virtual methods

		// get the dimension of the tensor
		virtual int nDimension() const = 0;

		// get the size of a specific dimension
		virtual long size(int dimension_) const = 0;

		// Get the number of elements (over all dimensions)
		long sizeAll() const;

		// set the tensor from another tensor (same type) - this will create a reference
		virtual void setTensor(const Tensor *src) = 0;

		// copy the tensor from another tensor (copy of any type) - this will make a real copy of the tensor values
		virtual void copy(const Tensor *src) = 0;

		// transpose 2 dimensions of a tensor
		virtual void transpose(const Tensor *src, int dimension1_, int dimension2_) = 0;

		// narrow a tensor along dimension #dimension_# starting at slice #firstIndex_# and of #size_# slices
		virtual void narrow(const Tensor *src, int dimension_, long firstIndex_, long size_) = 0;

		// select a tensor along dimension #dimension_# at slice #sliceIndex_#
		virtual void select(const Tensor *src, int dimension_, long sliceIndex_) = 0;

		// select a new tensor along dimension #dimension_# at slice #sliceIndex_#
		virtual Tensor* select(int dimension_, long sliceIndex_) const = 0;

		// print the tensor
		virtual void print(const char *name = NULL) const = 0;

		// print the tensor
		virtual void sprint(const char *name, ...) const = 0;

		// Access to the raw data
		virtual void* dataW() = 0;
		virtual const void* dataR() const = 0;
    inline virtual void resetFromData() {}

		// Get the size of an element
		virtual int typeSize() const = 0;

		// Test if the tensor is a reference tensor
		virtual bool isReference() { return m_isReference; };

		//---

		//
		virtual void resize(long dim0_) = 0;

		//
		virtual void resize(long dim0_, long dim1_) = 0;

		//
		virtual void resize(long dim0_, long dim1_, long dim2_) = 0;

		//
		virtual void resize(long dim0_, long dim1_, long dim2_, long dim3_) = 0;

		/////////////////////////////////////////////////////////////////////
		// SET functions for each possible type

		virtual void set(long, char) = 0;
                virtual void set(long, long, char) = 0;
                virtual void set(long, long, long, char) = 0;
                virtual void set(long, long, long, long, char) = 0;

                virtual void set(long, short) = 0;
                virtual void set(long, long, short) = 0;
                virtual void set(long, long, long, short) = 0;
                virtual void set(long, long, long, long, short) = 0;

                virtual void set(long, int) = 0;
                virtual void set(long, long, int) = 0;
                virtual void set(long, long, long, int) = 0;
                virtual void set(long, long, long, long, int) = 0;

                virtual void set(long, long) = 0;
                virtual void set(long, long, long) = 0;
                virtual void set(long, long, long, long) = 0;
                virtual void set(long, long, long, long, long) = 0;

                virtual void set(long, float) = 0;
                virtual void set(long, long, float) = 0;
                virtual void set(long, long, long, float) = 0;
                virtual void set(long, long, long, long, float) = 0;

                virtual void set(long, double) = 0;
                virtual void set(long, long, double) = 0;
                virtual void set(long, long, long, double) = 0;
                virtual void set(long, long, long, long, double) = 0;

                /////////////////////////////////////////////////////////////////////

		virtual ~Tensor() {};
	};

extern const char *str_TensorTypeName[];

extern "C"
{
#include "th/TH.h"
}

#define TYPE char
#define CAP_TYPE Char
#define TYPE_FORMAT "%d"
#include "core/TensorGen.h"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE

#define TYPE short
#define CAP_TYPE Short
#define TYPE_FORMAT "%d"
#include "core/TensorGen.h"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE

#define TYPE int
#define CAP_TYPE Int
#define TYPE_FORMAT "%d"
#include "core/TensorGen.h"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE

#define TYPE long
#define CAP_TYPE Long
#define TYPE_FORMAT "%ld"
#include "core/TensorGen.h"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE

#define TYPE float
#define CAP_TYPE Float
#define TYPE_FORMAT "%f"
#include "core/TensorGen.h"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE

#define TYPE double
#define CAP_TYPE Double
#define TYPE_FORMAT "%g"
#define DEFAULT_TENSOR
#include "core/TensorGen.h"
#undef TYPE_FORMAT
#undef TYPE
#undef CAP_TYPE
#undef DEFAULT_TENSOR

}

#endif
