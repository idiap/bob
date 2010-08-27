#ifndef _TORCH5SPRO_CASCADE_MACHINE_H_
#define _TORCH5SPRO_CASCADE_MACHINE_H_

#include "machine/Classifier.h"	// <CascadeMachine> is a <Classifier>
#include "machine/Machines.h"

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::CascadeMachine:
	//	Implementes a cascade of machines.
	//
	//	FORWARD algorithm:
	//	- for each stage:
	//		- for each Machine on that stage:
	//			- forward
	//		- if the result is above threshold => not pattern, break
	//
	//	Beside the cascade structure, in the model file there is saved (and load):
	//		- the id of the used machine (for creating the appropriate <xxxMachine> objects)
	//
	// 	NB: The Cascade will not deallocate the given machines with <setXXX> functions!
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class CascadeMachine : public Classifier
	{
	public:

		// Constructor
		CascadeMachine();

		// Destructor
		virtual ~CascadeMachine();

		///////////////////////////////////////////////////////////
		// Machine manipulation (add/remove/set Machines or stages)
		// NB: The Cascade will take care of deallocating the SET machines!

		bool			resize(int n_stages);
		bool			resize(int i_stage, int n_machines);
		bool			setMachine(int i_stage, int i_machine, Machine* machine);
		bool			setWeight(int i_stage, int i_machine, double weight);
		bool			setThreshold(int i_stage, double threshold);

		virtual bool            setThreshold(double threshold);

		// Set the model size to use (need to set the model size to each <Machine>) - overriden
		virtual void		setSize(const TensorSize& size);

		///////////////////////////////////////////////////////////

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		// Loading/Saving the content from files (<em>not the options</em>) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Machine of this kind - overriden
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine*	getAnInstance() const { return manage(new CascadeMachine()); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return CASCADE_MACHINE_ID; }

		///////////////////////////////////////////////////////////
		// Access functions

		int			getNoStages() const { return m_n_stages; }
		int			getNoMachines(int i_stage) const;
		Machine*		getMachine(int i_stage, int i_machine);
		const Machine*		getMachine(int i_stage, int i_machine) const;
		double			getWeight(int i_stage, int i_machine) const;
		double			getThreshold(int i_stage) const;

		///////////////////////////////////////////////////////////

	protected:

		// Deallocate the memory
		void			deallocate();

		///////////////////////////////////////////////////////////////
		// One stage in the cascade (a sequence of <Machine>s)

		struct Stage
		{
			// Constructor
			Stage()	:	m_machines(0),
                                        m_weights(0),
					m_n_machines(0),
					m_threshold(0.0)
			{
			}

			// Destructor
			~Stage()
			{
				deallocate();
			}

			// Reset the number of machines
			bool		resize(int n_machines);

			// Deallocate memory
			void		deallocate();

			// Set a new machine
			bool		setMachine(int i_machine, Machine* machine);

			// Set a new weight for some machine
			bool		setWeight(int i_machine, double weight);

			// Access functions
			Machine*	getMachine(int i_machine);
			const Machine*	getMachine(int i_machine) const;
			double		getWeight(int i_machine) const;

			//////////////////////////////////////////////////////////
			// Attributes

			Machine**	m_machines;	// The machines on this stage
			double*		m_weights;
			int		m_n_machines;	// Number of machines on this stage
			double		m_threshold;	// Threshold set after each stage
		};

		///////////////////////////////////////////////////////////////
		/// Attributes

		// The <Stage>s that compose the cascade
		Stage*			m_stages;
		int			m_n_stages;

	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool cascade_machine_registered = MachineManager::getInstance().add(
                new CascadeMachine(), "CascadeMachine");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

}

#endif
