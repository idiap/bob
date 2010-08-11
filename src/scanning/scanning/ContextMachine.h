#ifndef _TORCHVISION_SCANNING_CONTEXT_MACHINE_H_
#define _TORCHVISION_SCANNING_CONTEXT_MACHINE_H_

#include "machine/Classifier.h"			// <ContextMachine> is a <Classifier>
#include "scanning/LRMachine.h"
#include "scanning/Context.h"

namespace Torch
{
	#define CONTEXT_MACHINE_ID	10003

	/////////////////////////////////////////////////////////////////////////
	// Torch::ContextMachine:
	//	- implements a combination of small machines for each context feature
	//
	//      - PARAMETERS (name, type, default value, description):
        //		//
	//
	// TODO: doxygen header!
	/////////////////////////////////////////////////////////////////////////

	class ContextMachine : public Torch::Classifier
	{
	public:

		// Constructor
		ContextMachine();

		// Destructor
		virtual ~ContextMachine();

		// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		// Constructs an empty Machine of this kind
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine* 	getAnInstance() const { return manage(new ContextMachine); }

		// Get the ID specific to each Machine
		virtual int		getID() const { return CONTEXT_MACHINE_ID; }

		// Loading/Saving the content from files (<em>not the options</em>)
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		// Access functions
		LRMachine&		getFModel(int f) { return m_fmodels[f]; }
		const LRMachine&	getFModel(int f) const { return m_fmodels[f]; }
		LRMachine&		getCModel() { return m_cmodel; }
		const LRMachine&	getCModel() const  { return m_cmodel; }

		/////////////////////////////////////////////////////////////////

        private:

                /////////////////////////////////////////////////////////////////
                // Attributes

		double*			m_poutput;		// Direct access to the machine's output
		DoubleTensor		m_foutputs;		// Store outputs from context feature models

		LRMachine*		m_fmodels;		// Models for each feature
		LRMachine		m_cmodel;		// Combined model

		Context			m_context;		// Buffered sample for easily access the features
	};
}

#endif
