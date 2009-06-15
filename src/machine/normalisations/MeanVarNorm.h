#ifndef _TORCH5SPRO_MEANVAR_NORM_H_
#define _TORCH5SPRO_MEANVAR_NORM_H_

#include "Machines.h"
#include "DataSet.h"

namespace Torch
{
        //////////////////////////////////////////////////////////////////////////////////////
	// Torch::Machine:
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class MeanVarNorm : public Machine
	{
	public:

		/// Constructors
		MeanVarNorm();
		
		MeanVarNorm(const int n_inputs_, DataSet *dataset);
		
		/// Destructor
		virtual ~MeanVarNorm();

		///////////////////////////////////////////////////////////

		virtual bool 	resize(const int n_inputs_);

		///
		virtual bool 	forward(const Tensor& input);

		/// Loading/Saving the content from files (\emph{not the options}) - overriden
		virtual bool	loadFile(File& file);
		virtual bool	saveFile(File& file) const;

		/// Constructs an empty Machine of this kind - overriden
		/// (used by <MachineManager>, this object should be deallocated by the user)
		virtual Machine*	getAnInstance() const { return new MeanVarNorm(); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return MEANVAR_NORM_MACHINE_ID; }

		///////////////////////////////////////////////////////////

		int 		n_inputs;
		int 		n_outputs;

		double*		m_mean;
		double*		m_stdv;
	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool meanvar_norm_machine_registered = MachineManager::getInstance().add(new MeanVarNorm(), "MeanVarNorm");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
}

#endif
