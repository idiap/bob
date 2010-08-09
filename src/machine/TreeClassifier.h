#ifndef _TORCH5SPRO_TREE_CLASSIFIER_H_
#define _TORCH5SPRO_TREE_CLASSIFIER_H_

#include "Classifier.h"	// <TreeClassifier> is a <Classifier>
#include "Machines.h"

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::TreeClassifier:
	//	Implementes a tree of nodes. Each node has N Classifiers and N+1 childs.
	//
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class TreeClassifier : public Classifier
	{
	public:

		// Constructor
		TreeClassifier();

		// Destructor
		virtual ~TreeClassifier();

		///////////////////////////////////////////////////////////
		// Classifier manipulation (add/remove/set Classifiers or Nodes)
		// NB: The Tree will take care of deallocating the SET Classifiers!

		bool			resize(int n_nodes);
		bool			resize(int i_node, int n_classifiers);
		bool			setClassifier(int i_node, int i_classifier, Classifier* classifier, double mu = 0.0, double stdv = 1.0);
		bool			setThreshold(int i_node, int i_classifier, double threshold);
		bool			setChild(int i_node, int i_classifier, int child_node);

		virtual bool            setThreshold(double threshold);

		bool			setClasses(int n_classes);
		int			getClasses() const;

		// Set the model size to use (need to set the model size to each <Classifier>) - overriden
		virtual void		setSize(const TensorSize& size);

		///////////////////////////////////////////////////////////

		/// Process the input tensor
		virtual bool 		forward(const Tensor& input);

		double			normal(double x, double mu, double sigma, double stdv);
			   
		// Loading/Saving the content from files (\em{not the options}) - overriden
		virtual bool		loadFile(File& file);
		virtual bool		saveFile(File& file) const;

		/// Constructs an empty Classifier of this kind - overriden
		// (used by <MachineManager>, this object is automatically deallocated)
		virtual Machine*	getAnInstance() const { return manage(new TreeClassifier()); }

		// Get the ID specific to each Machine - overriden
		virtual int		getID() const { return TREE_CLASSIFIER_ID; }

		///////////////////////////////////////////////////////////
		// Access functions

		int			getNoNodes() const { return m_n_nodes; }
		int			getNoClassifiers(int i_node) const;
		Classifier*		getClassifier(int i_node, int i_classifier);
		const Classifier*	getClassifier(int i_node, int i_classifier) const;
		double			getThreshold(int i_node, int i_classifier) const;
		int			getChild(int i_node, int i_classifier) const;

		///////////////////////////////////////////////////////////

	protected:

		// Deallocate the memory
		void			deallocate();

		///////////////////////////////////////////////////////////////
		// One Node in the tree (a sequence of <Classifier>s)

		struct Node
		{
			// Constructor
			Node()	:	m_classifiers(0),
					m_n_classifiers(0),
					m_thresholds(0),
					m_mu(0),
					m_stdv(0),
					m_sigma(0),
					m_childs(0)
			{
			}

			// Destructor
			~Node()
			{
				deallocate();
			}

			// Reset the number of classifiers
			bool		resize(int n_classifiers);

			// Deallocate memory
			void		deallocate();

			// Set a new classifier
			bool		setClassifier(int i_classifier, Classifier* classifier, double mu, double stdv);

			// Set a new threshold for some classifier
			bool		setThreshold(int i_classifier, double threshold);
				
			// Set a child
			bool		setChild(int i_classifier, int child_node);

			// Access functions
			Classifier*		getClassifier(int i_classifier);
			const Classifier*	getClassifier(int i_classifier) const;
			double			getThreshold(int i_classifier) const;
			int			getChild(int i_classifier) const;

			//////////////////////////////////////////////////////////
			// Attributes

			Classifier**	m_classifiers;	// The classifiers in this node
			int		m_n_classifiers;// Number of classifiers in this node
			double*		m_thresholds;	// Threshold of each classifier
			double*		m_mu;
			double*		m_stdv;
			double*		m_sigma;
			int*		m_childs;	// Childs associated to each classifier
		};

		///////////////////////////////////////////////////////////////
		/// Attributes

		// The <Node>s that compose the tree
		Node*			m_nodes;
		int			m_n_nodes;
		int			m_n_classes;

		// Fast access to the output
		double*			m_fast_output;	// Pointer to the DoubleTensor
	};

        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        // REGISTER this machine to the <MachineManager>
        const bool tree_machine_registered = MachineManager::getInstance().add(
                new TreeClassifier(), "TreeClassifier");
        // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

}

#endif
