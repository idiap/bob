#ifndef _TORCHVISION_FLDA_MACHINE_H_
#define _TORCHVISION_FLDA_MACHINE_H_

#include "Machine.h"			// <FLDAMachine> is a <Machine>
#include "Machines.h"

namespace Torch
{
	/////////////////////////////////////////////////////////////////////////
	// Torch::FLDAMachine:
	//	- implements Fisher Linear Discriminant Analysis
	//	- <getOutput> will return a 1x1D DoubleTensor with the score
	//	- it processes only DoubleTensors having the same size as set with <resize>
	//
	//      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class FLDAMachine : public Torch::Machine
	{
	public:

		// Constructor
		FLDAMachine(int size = 1);

		// Destructor
		virtual ~FLDAMachine();

		// Process the input tensor
		virtual bool 	forward(const Tensor& input);

		// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine* getAnInstance() const { return manage(new FLDAMachine); }

		// Get the ID specific to each Machine
		virtual int	getID() const { return FLDA_MACHINE_ID; }

		// Loading/Saving the content from files (\emph{not the options})
		virtual bool	loadFile(File& file);
		virtual bool	saveFile(File& file) const;

		// Resize
		bool		resize(int size);

		// Set machine's parameters
		void		setThreshold(double threshold);
		void		setProjection(const double* proj, double proj_avg);

		// Access functions
		double		getThreshold() const { return m_threshold; }

	private:

		/////////////////////////////////////////////////////////////
		// Attributes

		double*		m_poutput;	// Direct access to the machine's output

		int		m_size;		// Number of dimensions
		double*		m_proj;		// N-dimensional projection vector
		double		m_proj_avg;	// 1-dimensional projected average vector
		double		m_threshold;	// Tunned threshold (default 0.0)
	};
}

#endif
