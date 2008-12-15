#ifndef _TORCH5SPRO_GRADIENT_MACHINE_H_
#define _TORCH5SPRO_GRADIENT_MACHINE_H_

#include "Machine.h"	// GradientMachine is a <Machine>

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::GradientMachine:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class GradientMachine : public Machine
	{
	public:

		/// Constructor
		GradientMachine(const int n_inputs_, const int n_outputs_, const int n_parameters_ = 0);

		/// Destructor
		virtual ~GradientMachine();

		///////////////////////////////////////////////////////////

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		///////////////////////////////////////////////////////////

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		int n_inputs;
		int n_outputs;

		// Parameters
		int n_parameters;
		double*			m_parameters;		// The parameters of the gradient machine
		double*			m_der_parameters;	// The derivatives of the parameters
	};

}

#endif
