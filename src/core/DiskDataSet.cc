#include <cassert>
#include "DiskDataSet.h"
#include "Tensor.h"

namespace Torch {

DiskDataSet::DiskDataSet(Tensor::Type example_type_)
	: DataSet(example_type_)
{
	//
	map = new IndexMapper();

	// create a target set
	targetset = new TargetSet(0);
}

DiskDataSet::~DiskDataSet()
{
	// IMP LATER
}

Tensor& Torch::DiskDataSet::operator()(long) 
{
	assert(true);
}

Tensor* Torch::DiskDataSet::getTarget(long) 
{
	assert(true);
}

void Torch::DiskDataSet::setTarget(long, Tensor*) 
{
	assert(true);
}

Tensor* DiskDataSet::getExample(long index)
{
	// get the correct file
	TensorFile* tf = map->getFile(index);

	// get the correct offset
	int offset = map->getOffset(index);

	// temp tensor 
	Tensor* match = NULL;

	for (int cnt = 0; cnt < offset; cnt++) {
		match = tf->load();
	}

	// return the <index> tensor
	return match;
}

bool DiskDataSet::load(const char* filename)
{
	map->map(filename);
}

/*
Tensor* DiskDataSet::getTarget(long t)
{
}

void DiskDataSet::cleanup()
{
}

DiskDataSet::~DiskDataSet()
{
}

*/

DiskDataSet::IndexMapper::IndexMapper()
{	
	// initialize the counters
	m_file_cnt	= 0;
	m_top_num	= 0;

	// initialize the pointers
	map_index	= NULL;	
	file_list 	= NULL;

	// allocate and initialize using resize
	// 	make it the default size
	resize(M_START_LIMIT);
}

void DiskDataSet::IndexMapper::resize(long size)
{	

	// if we dont have an array, just create an new 
	if (NULL == map_index || NULL == file_list ) {
		map_index 	= new long  [size];
		file_list 	= new TensorFile* [size];
	}

	// non-destructive creation of new array
	else {

		// create new array
		long* 		tmp_map_index 	= new long  	  [size];
		TensorFile** 	tmp_file_list 	= new TensorFile* [size];


		// copy over as many old targets as possible
		const long copy_end = ( size < m_limit ) ? size : m_limit;

		for (int cnt = 0; cnt < copy_end; cnt++) {
			tmp_map_index[cnt] = map_index[cnt];
			tmp_file_list[cnt] = file_list[cnt];
		}


		// initialize the rest to zero (if any)
		const long mid          = copy_end;
		const long rest_end     = (size < m_limit ) ? m_limit : size;

		for (int cnt = mid; cnt < rest_end; cnt++) {
			tmp_map_index[cnt] = 0;
			tmp_file_list[cnt] = NULL;
		}

		// de-alloc old memory
		delete	 map_index; 	
		delete[] file_list;

		// update array
		map_index = tmp_map_index;
		file_list = tmp_file_list;
	}

	//update the number of targets
	m_limit = size;
}

///////////////////////////////////////////////////////////////////////////////
// \brief 	map
// \param[in]	file	FileTensor including tensora
// \param[out]		Success or Fail

bool DiskDataSet::IndexMapper::map(const char* filename)
{
	// create and open file
	TensorFile *tf = new TensorFile();
	tf->openRead(filename);

	// get the amount of tensors in file
	long samples = tf->getNumberOfSamples();

	// make sure we fit, otherwise double size
	if ((m_top_num + samples) > m_limit) {
		resize(m_limit * M_RESCALE_FAC);
	}

	map_index[m_file_cnt] = samples;
	file_list[m_file_cnt] = tf;

	// adjust the highest index so far
	m_top_num += samples;

	// increase files cnt
	m_file_cnt++;

	// do _not_ close the file since 
	// we cannot open it again without a filename
}

TensorFile* DiskDataSet::IndexMapper::getFile(long index)
{
	// pointer to match
	TensorFile* match = NULL;
	
	long acc = 0;

	long cnt = 0;
	// loop over the files to find the correct on
	do {
		// add the number of samples in file cnt
		acc += map_index[cnt];

		// if acc is higher or equal than index we found or match
		if (acc >= index) {
			match = file_list[cnt];
		}

	} while (NULL == match);

	return match;
}

long DiskDataSet::IndexMapper::getOffset(long index) 
{
	// make sure index fits
	if (index > m_top_num) {
		return -1;
	}

	// pointer to match
	long match = -1;
	
	long acc = 0;

	long cnt = 0;
	// loop over the files to find the correct on
	do {
		// add the number of samples in file cnt
		acc += map_index[cnt];

		// if acc is higher or equal than index we found or match
		if (acc >= index) {
			match = map_index[cnt] - (acc - index);
		}

	} while (0 > match);

	return match;
}

} // namespace torch
