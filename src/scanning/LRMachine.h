#ifndef _TORCHVISION_LR_MACHINE_H_
#define _TORCHVISION_LR_MACHINE_H_

#include "core/Machine.h"			// <LRMachine> is a <Machine>

namespace Torch
{
	#define LR_MACHINE_ID	10002

	/////////////////////////////////////////////////////////////////////////
	// Torch::LRMachine:
	//	- implements Logistic Regression Linear Discriminant Analysis
	//	- <getOutput> will return a 1x1D DoubleTensor with the score
	//	- it processes only DoubleTensors having the same size as set with <resize>
	//
	//      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class LRMachine : public Torch::Machine
	{
	public:

		// Constructor
		LRMachine(int size = 1);

		// Destructor
		virtual ~LRMachine();

		// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine* 	getAnInstance() const { return manage(new LRMachine); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return LR_MACHINE_ID; }

		// Loading/Saving the content from files (\em{not the options})
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		// Resize
		bool			resize(int size);

		// Set machine's parameters
		void			setThreshold(double threshold);
		void			setWeights(const double* weights);

		// Access functions
		double			getThreshold() const { return m_threshold; }
		const double*		getWeights() const { return m_weights; }

		// Apply the sigmoid function on some data
		static double		sigmoid(const double* data, const double* weights, int size);
		static double		sigmoidEps(const double* data, const double* weights, int size, double eps = 0.1);

		/////////////////////////////////////////////////////////////////

        private:

		/////////////////////////////////////////////////////////////////
                // Attributes

                double*			m_poutput;	// Direct access to the machine's output

                int			m_size;		// Number of dimensions
		double*			m_weights;	// [N+1]-dimensional weights
		double			m_threshold;	// Tunned threshold (default 0.5)
	};
}

#endif
