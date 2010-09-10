
#ifndef LIST_DATA_SET_H
#define LIST_DATA_SET_H

#include <vector>
#include <string>
#include <iostream>

#include "core/DataSet.h"
#include "core/MemoryDataSet.h"

namespace Torch {

	class ListDataSet : public DataSet {

	private:
		std::vector<Tensor *> *tensors;
		long n_inputs; ///< the size of the vectors

	public:
		ListDataSet() : DataSet () {
			tensors = new std::vector<Tensor *>;
			n_inputs = 0;
		}

		/**
		 *
		 */
		~ListDataSet() {
			std::vector<Tensor *>::iterator it = tensors->begin();
			while (it != tensors->end() ) {
				Tensor *tensor = *it++;
				delete tensor;
			}

			delete tensors;
		}

		void addTensor(Tensor *tensor) {
			tensors->push_back(tensor);
			n_inputs++;
		}

		/**
		 *
		 *\return the number of loaded tensors
		 */
		int load(std::string fname) {
			return append(fname);
		};

		/**
		 * Load with multiple files
		 */
		int load(std::vector<std::string> &filenames);

		/**
		 * Even if list is not empty, just append
		 *
		 *\return the number of loaded tensors
		 *
		 */
		int append(std::string fname);

		/**
		 *
		 * \brief save
		 */
		int save(const char *fname);

		int getSize() {
			return tensors->size();
		}

		long getVectorDim() {
			return n_inputs;
		}

		Tensor *getNextExample() {
			std::cerr << "NOT IMPLEMENTED" << std::endl;
			exit(-1);
		}

		Tensor *getExample(long int index) {
			Tensor *example = tensors->at(index);
			if (NULL == example) {
				std::cerr << "NULL == example" << std::endl;
			}

			return example;
		}

		Tensor &operator() (long int index) {
			std::cerr << "NOT IMPLEMENTED" << std::endl;
			exit(-1);
		}

		Tensor *getTarget(long int) { return NULL; }

		void setTarget(long int, Tensor*) { }

		/**
		 * Turn the whole data set to floats (double), since we are 
		 * not allowd to always use doubles (floats)
		 */
		void turn_to_floats();
		void turn_to_doubles();

		/**
		 * Get memory data set with example 10 uniformly samples
		 *
		 */
		MemoryDataSet *get_subset(const int size);
		MemoryDataSet *get_subset_all();
	};
}

#endif
