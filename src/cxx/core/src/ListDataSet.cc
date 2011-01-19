#include <vector>
#include <iostream>
#include <algorithm>

#include "core/Tensor.h"
#include "core/TensorFile.h"

#include "core/ListDataSet.h"

// torch things

using namespace std;
using namespace Torch;

/**
 * Append to the list,
 *
 * \param[in] name_ the name of the file to read in into dataset
 * \return the number of tensor that are appended
 */
int ListDataSet::append(std::string fname)
{
	// if we are empty we need to store the type later
	bool first_tensor = false;
	if (0 == m_n_examples)
		first_tensor = true;

	// open the tensor file for reading
	TensorFile tf;
	tf.openRead(fname.c_str());

	// one by one,, read up the tensors
	Tensor* tensor = tf.load();
	int cnt = 1;
	while (NULL != tensor) {

		// we want to store the type of tensors
		// weeknes in this algo, is that we assume that
		// they are all the same.
		if (true == first_tensor) {
			m_example_type = tensor->getDatatype();
			first_tensor = false;

			// register the size of the vector
			// n_inputs = tensor->sizeAll(); WRONG IF BLOCKS
			n_inputs = tensor->size(0);
		}

		// save in the list
		tensors->push_back(tensor);

		// get next tensor
		tensor = tf.load();
		cnt++;
	}
	cnt--; // we have always read one too much

	// update the number of tensors in the dataset
	m_n_examples += cnt;

	// return the number of samples that are read from file
	return cnt;
}

/**
 * To load from multiple files
 *
 */
int ListDataSet::load(vector<string> &filenames)
{
	int total = 0;

	vector<string>::iterator it = filenames.begin();
	while ( it != filenames.end() ) {
		string fname = *it++;
		total += append(fname.c_str());
	}

	return total;
}



/**
 * Turn the whole data set to floats
 *
 */
void ListDataSet::turn_to_floats()
{
	vector<Tensor *> *new_vec = new vector<Tensor *>;

	vector<Tensor *>::iterator it = tensors->begin();
	while ( it != tensors->end()) {

		// old
		Tensor *old =*it++;

		// new tensors
		FloatTensor *flt = new FloatTensor();
		flt->copy(old);

		new_vec->push_back(flt);

		// remove the old
		delete old;
	}

	// erase old vector
	delete tensors;

	// assign the new vector
	m_example_type = Tensor::Float;
	tensors = new_vec;
}

/**
 * Turn the whole data set to doubles
 *
 */
void ListDataSet::turn_to_doubles()
{
	vector<Tensor *> *new_vec = new vector<Tensor *>;

	vector<Tensor *>::iterator it = tensors->begin();
	while ( it != tensors->end()) {

		// old
		Tensor *old =*it++;

		// new tensors
		DoubleTensor *dbl = new DoubleTensor();
		dbl->copy(old);

		new_vec->push_back(dbl);

		// remove the old
		delete old;
	}

	// erase old vector
	delete tensors;

	// assign the new vector
	m_example_type = Tensor::Double;
	tensors = new_vec;
}

/**
 * Get memory data set with <n> samples (uniformly sampling)
 *
 */
MemoryDataSet *ListDataSet::get_subset(const int size)
{
	// create a memory data set to return
	// MemoryDataSet *subset = new MemoryDataSet(size,
	//					  m_example_type,
	//					  false,
	MemoryDataSet *subset = new MemoryDataSet(size,
						  m_example_type,
						  false);
	// get 10 samples
	for (int s = 0; s < size; s++) {

		// pick out a tensor, uniformly randomly, double checked :)
		const int index = rand() % m_n_examples;

		// copy it over
		Tensor *pick = getExample(index);
		Tensor *push = subset->getExample(s);

		push->copy(pick);
	}

	// return the memory data set
	return subset;
}


/**
 * Get the whole dataset but as a MemoryDataSet
 *
 */
MemoryDataSet *ListDataSet::get_subset_all()
{
	MemoryDataSet *subset = new MemoryDataSet(getSize(),
						  m_example_type,
						  false);
	for (int s = 0; s < getSize(); s++) {

		// copy it over
		Tensor *pick = getExample(s);
		Tensor *push = subset->getExample(s);

		push->copy(pick);
	}

	// return the memory data set
	return subset;
}


/**
 * Save the listdataset as a  tensor file (on disk)
 *
 */
int ListDataSet::save(const char *fname)
{
	// get the first tensor, this will work as the blueprint of all
	// the tensors
	Tensor *curr = getExample(0);

	const Tensor::Type ttype = curr->getDatatype();
	const int tdims = curr->nDimension();
	const int tsize0 = curr->size(0);
	const int tsize1 = curr->size(1);
	const int tsize2 = curr->size(2);
	const int tsize3 = curr->size(3);

	TensorFile tf;
	tf.openWrite(fname, ttype, tdims, tsize0, tsize1, tsize2, tsize3);

	vector<Tensor *>::iterator it = tensors->begin();
	while ( it != tensors->end()) {

		curr =*it++;
		tf.save(*curr);
	}

	// close file
	tf.close();

	//
	return 1;
}
