#include "Tensor.h"
#include "MemoryDataSet.h"

using namespace Torch;

void setDigitTensor(short digit[5][4], DoubleTensor &t)
{
	for(int i = 0 ; i < 5 ; i++)
	{
		for(int j = 0 ; j < 4 ; j++)
		{
			//print("%d ", digit[i][j]);

			t(i,j) = digit[i][j];
		}
		//print("\n");
	}
}

short zero[5][4] = {	{0,1,1,0},
   			{1,0,0,1},
   			{1,0,0,1},
   			{1,0,0,1},
   			{0,1,1,0}};

short one[5][4] = {	{0,0,1,1},
   			{0,1,0,1},
   			{0,0,0,1},
   			{0,0,0,1},
   			{0,0,0,1}};

short two[5][4] = {	{1,1,1,1},
   			{0,0,1,0},
   			{0,1,0,0},
   			{1,0,0,0},
   			{1,1,1,1}};

short three[5][4] = {	{1,1,1,1},
   			{0,0,0,1},
   			{0,0,1,1},
   			{0,0,0,1},
   			{1,1,1,1}};

short four[5][4] = {	{1,0,0,0},
   			{1,0,0,0},
   			{1,1,1,1},
   			{0,0,1,0},
   			{0,0,1,0}};

short five[5][4] = {	{1,1,1,1},
   			{1,0,0,0},
   			{1,1,1,1},
   			{0,0,0,1},
   			{1,1,1,1}};

short six[5][4] = {	{1,1,1,1},
   			{1,0,0,0},
   			{1,1,1,1},
   			{1,0,0,1},
   			{1,1,1,1}};

short seven[5][4] = {	{1,1,1,1},
   			{0,0,0,1},
   			{0,0,1,1},
   			{0,1,0,0},
   			{0,1,0,0}};

short eight[5][4] = {	{1,1,1,1},
   			{1,0,0,1},
   			{0,1,1,0},
   			{1,0,0,1},
   			{1,1,1,1}};

short nine[5][4] = {	{1,1,1,1},
   			{1,0,0,1},
   			{1,1,1,1},
   			{0,0,0,1},
   			{1,1,1,1}};

int main()
{
   	//---
	{
		// Handle an empty dataset
		MemoryDataSet empty_dataset;

		print("Empty DataSet:\n");
		print("   n_examples = %d\n", empty_dataset.getNumberOfExamples());

	}

	{
		//
		MemoryDataSet m0(10, Tensor::Double);

		print("DataSet m0:\n");
		print("   n_examples = %d\n", m0.getNumberOfExamples());

		// loop on examples to print them
		for(int t = 0 ; t < m0.getNumberOfExamples() ; t++)
		{
			DoubleTensor *input = (DoubleTensor *) m0.getExample(t);
			input->resize(10);
			input->fill(1);
			input->sprint("input %d", t); 
		}

		print("DataSet m0 done.\n");
	}

	//---
	{
		// Handle a dataset of 10 examples of 1D input and target tensors
		MemoryDataSet m1(10, Tensor::Double, true, Tensor::Short);

		print("DataSet m1:\n");
		print("   n_examples = %d\n", m1.getNumberOfExamples());

		// loop on examples to allocate them
		for(int t = 0 ; t < m1.getNumberOfExamples() ; t++)
		{
			DoubleTensor *input = (DoubleTensor *) m1.getExample(t);
			input->resize(2,4);
			input->fill(t);

			ShortTensor *target = (ShortTensor *) m1.getTarget(t);
			target->resize(1);
			target->fill(t);
		}

		// loop on examples to print them again
		for(int t = 0 ; t < m1.getNumberOfExamples() ; t++)
		{
			m1.getExample(t)->sprint("input %d", t); 
			m1.getTarget(t)->sprint("target %d", t); 
		}

		print("DataSet m1 done.\n");
	}
/*
	//---
	{
		// Handle the XOR dataset: 4 examples of 1D input tensor of size 2 and 1D target tensor of size 1
		//
		//   	input	target
		//   	0 0	0
		//	0 1	1
		//	1 0	1
		//	1 1	1
		//
		//
		
		MemoryDataSet m2(4, Tensor::Double, true, Tensor::Short);

		print("DataSet m2:\n");
		print("   n_examples = %d\n", m2.getNumberOfExamples());

		// this dataset will have 2 possible targets
		ShortTensor *target0 = new ShortTensor(1); target0->fill(0);
		ShortTensor *target1 = new ShortTensor(1); target1->fill(1);

		// set the example 0
		DoubleTensor &input0 = *((DoubleTensor*)(m2.getExample(0)));
		input0(0) = 0; input0(1) = 0;
		example0.target = target0;

		// set the example 1
		DoubleTensor input1(2);
		input1(0) = 0; input1(1) = 1;
		TensorPair &example1 = m2(1);
		example1.input = &input1;
		example1.target = target1;

		// set the example 2
		DoubleTensor input2(2);
		input2(0) = 1; input2(1) = 0;
		TensorPair &example2 = m2(2);
		example2.input = &input2;
		example2.target = target1;

		// set the example 3
		DoubleTensor input3(2);
		input3(0) = 1; input3(1) = 1;
		TensorPair &example3 = m2(3);
		example3.input = &input3;
		example3.target = target0;


		// loop on examples to print them
		for(int t = 0 ; t < m2.getNumberOfExamples() ; t++)
		{
			TensorPair example = m2(t);
		
			input = example.input;
			target = example.target;

			if(input != NULL) input->sprint("input %d", t); 
			if(target != NULL) target->sprint("target %d", t); 
		}

		// delete the two target Tensors
		delete target0;
		delete target1;
		
		// delete the only input Tensor allocated
		delete example0.input;

	}

	//---
	{
		// Handle a dataset of digits (2D input tensors of size 5x4) for classification (1D target tensor of size 1)
		//		
		//		   	.oo.  ..oo  oooo  oooo  o...  oooo  oooo  oooo  oooo  oooo   
		//			o  o  .o.o  ...o  ...o  o.o.  o...  o...  ...o  o..o  o..o
		//			o  o  ...o  .oo.  .ooo  oooo  oooo  oooo  .oo.  .oo.  oooo
		//			o  o  ...o  o...  ...o  ..o.  ...o  o..o  ..o.  o..o  ...o
		//			.oo.  ...o  oooo  oooo  ..o.  oooo  oooo  .o..  oooo  ...o
		//
		//

	   	int n_targets = 10;
		int n_examples_per_digit = 10;
		int n_examples = n_targets * n_examples_per_digit;

		// the dataset of digits
		MemoryDataSet digitDataSet(n_examples);

		// the targets
		ShortTensor **target = new ShortTensor*[n_targets];
		DoubleTensor **digit = new DoubleTensor*[n_targets];
		for(int i = 0 ; i < n_targets ; i++)
		{
			target[i] = new ShortTensor(1);
			target[i]->fill(i);
		}

		// the original digits
		DoubleTensor digit0(5,4);
		setDigitTensor(zero, digit0);
		digit0.print("Digit 0");
		digit[0] = &digit0;

		DoubleTensor digit1(5,4);
		setDigitTensor(one, digit1);
		digit1.print("Digit 1");
		digit[1] = &digit1;

		DoubleTensor digit2(5,4);
		setDigitTensor(two, digit2);
		digit2.print("Digit 2");
		digit[2] = &digit2;

		DoubleTensor digit3(5,4);
		setDigitTensor(three, digit3);
		digit3.print("Digit 3");
		digit[3] = &digit3;

		DoubleTensor digit4(5,4);
		setDigitTensor(four, digit4);
		digit4.print("Digit 4");
		digit[4] = &digit4;

		DoubleTensor digit5(5,4);
		setDigitTensor(five, digit5);
		digit5.print("Digit 5");
		digit[5] = &digit5;

		DoubleTensor digit6(5,4);
		setDigitTensor(six, digit6);
		digit6.print("Digit 6");
		digit[6] = &digit6;

		DoubleTensor digit7(5,4);
		setDigitTensor(seven, digit7);
		digit7.print("Digit 7");
		digit[7] = &digit7;

		DoubleTensor digit8(5,4);
		setDigitTensor(eight, digit8);
		digit8.print("Digit 8");
		digit[8] = &digit8;

		DoubleTensor digit9(5,4);
		setDigitTensor(nine, digit9);
		digit9.print("Digit 9");
		digit[9] = &digit9;

		int T = 0;
		for(int d = 0 ; d < n_targets ; d++)
			for(int t = 0 ; t < n_examples_per_digit ; t++)
			{
				TensorPair &example = digitDataSet(T);

				example.input = new DoubleTensor(5,4);
				example.input->copy(digit[d]);

				// ideally we should add some noise on those input examples

				example.target = target[d];

				T++;
			}

		for(int t = 0 ; t < T ; t++)
		{
			TensorPair example = digitDataSet(t);

			DoubleTensor *tensor_ = (DoubleTensor *) example.input;
			delete tensor_;
		}

		// delete the targets
		for(int i = 0 ; i < n_targets ; i++) delete target[i];
		delete [] target;
		delete [] digit;
	}

	//---
	{
		// Handle a dataset of points for distribution modelling
	}
*/

	return 0;
}

