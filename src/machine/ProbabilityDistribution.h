#ifndef _TORCH5SPRO_PROBABILITY_DISTRIBUTION_MACHINE_H_
#define _TORCH5SPRO_PROBABILITY_DISTRIBUTION_MACHINE_H_

#include "Machine.h"	// ProbabilityDistribution is a <Machine>

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::ProbabilityDistribution:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class ProbabilityDistribution : public Machine
	{
	public:
		/// Constructors
		ProbabilityDistribution();

		ProbabilityDistribution(const int n_inputs_, const int n_parameters_ = 0);
		
		/// Destructor
		virtual ~ProbabilityDistribution();

		///
		virtual bool 		resize(const int n_inputs_, const int n_parameters_ = 0);

		///
		virtual bool 		shuffle() { return true; };

		///
		virtual bool 		prepare();
		
		///
		virtual bool 		EMinit() { return true; };

		///
		virtual bool 		EMupdate() { return true; };
		
		///////////////////////////////////////////////////////////

		///
		virtual bool 		forward(const Tensor& input);
		virtual bool 		forward(const DoubleTensor *input) = 0;

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		///////////////////////////////////////////////////////////

	protected:
		int n_inputs;
		int n_parameters;
		double *parameters;
	};

}

#endif
