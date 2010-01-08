#include "TreeClassifier.h"
#include "Tensor.h"
#include "File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Reset the number of classifiers

bool TreeClassifier::Node::resize(int n_classifiers)
{
	if (n_classifiers < 1)
	{
		return false;
	}

	// OK, allocate the new classifiers
	if (n_classifiers == m_n_classifiers)
	{
	}
	else
	{
		deallocate();
		m_n_classifiers = n_classifiers;
		m_classifiers = new Classifier*[m_n_classifiers];
		m_thresholds = new double[m_n_classifiers];
		for (int i = 0; i < m_n_classifiers; i ++)
		{
			m_classifiers[i] = 0;
			m_thresholds[i] = 0.0;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Deallocate memory

void TreeClassifier::Node::deallocate()
{
	delete[] m_classifiers;
	delete[] m_thresholds;
	m_classifiers = 0;
	m_thresholds = 0;
	m_n_classifiers = 0;
}

//////////////////////////////////////////////////////////////////////////
// Set a new classifier

bool TreeClassifier::Node::setClassifier(int i_classifier, Classifier* classifier)
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers || classifier == 0)
	{
		return false;
	}

	// OK
	m_classifiers[i_classifier] = classifier;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Set a threshold for some classifier

bool TreeClassifier::Node::setThreshold(int i_classifier, double threshold)
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers)
	{
		Torch::error("TreeClassifier::Node::setThreshold - invalid parameters!\n");
		return false;
	}

	// OK
	m_thresholds[i_classifier] = threshold;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Set the model size to use (need to set the model size to each <Classifier>) - overriden

void TreeClassifier::setSize(const TensorSize& size)
{
	Machine::setSize(size);

	// OK, set the model size to each <Classifier>
	for (int s = 0; s < m_n_nodes; s ++)
	{
		Node& node = m_nodes[s];
		for (int n = 0; n < node.m_n_classifiers; n ++)
		{
			Classifier* classifier = node.m_classifiers[n];
			if (classifier != 0)
			{
				classifier->setSize(size);
			}
		}
	}
}

//////////////////////////////////////////////////////////////////////////
// Access functions

double TreeClassifier::Node::getThreshold(int i_classifier) const
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers)
	{
		Torch::error("TreeClassifier::Node::getThreshold - invalid parameters!\n");
	}

	return m_thresholds[i_classifier];
}

Classifier* TreeClassifier::Node::getClassifier(int i_classifier)
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers)
	{
		Torch::error("TreeClassifier::Node::getClassifier - invalid parameters!\n");
	}

	return m_classifiers[i_classifier];
}

const Classifier* TreeClassifier::Node::getClassifier(int i_classifier) const
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers)
	{
		Torch::error("TreeClassifier::Node::getClassifier - invalid parameters!\n");
	}

	return m_classifiers[i_classifier];
}

//////////////////////////////////////////////////////////////////////////
// Constructor

TreeClassifier::TreeClassifier()
	:	Classifier(),
		m_nodes(0), m_n_nodes(0),
                m_fast_output(0)
{
        // Allocate the output
	m_output.resize(1);
	const DoubleTensor* t_output = (DoubleTensor*)&m_output;
	m_fast_output = t_output->t->storage->data + t_output->t->storageOffset;
}

//////////////////////////////////////////////////////////////////////////
// Destructor

TreeClassifier::~TreeClassifier()
{
	deallocate();
}

//////////////////////////////////////////////////////////////////////////
// Deallocate the memory

void TreeClassifier::deallocate()
{
	delete[] m_nodes;
	m_nodes = 0;
	m_n_nodes = 0;
}

//////////////////////////////////////////////////////////////////////////
// Process the input tensor

/*
     !!! TODO !!!
 */

bool TreeClassifier::forward(const Tensor& input)
{
	m_isPattern = true;
	m_patternClass = 0;

	for (int s = 0; s < m_n_nodes; s ++)
	{
		const Node& node = m_nodes[s];

		double output = 0.0;
		for (int n = 0; n < node.m_n_classifiers; n ++)
		{
			Classifier* classifier = node.m_classifiers[n];
			if (classifier == 0)
			{
				Torch::message("TreeClassifier::forward - invalid classifier!\n");
				return false;
			}
			classifier->setRegion(m_region);
			if (classifier->forward(input) == false)
			{
				Torch::message("TreeClassifier::forward - error running the classifier!\n");
				return false;
			}

			// if(classifier->getOutput().get(0) > node.m_thresholds[n]) output += classifier->getOutput().get(0);
		}

		// Check if rejected
		*m_fast_output = output;

		//if (output < stage.m_threshold)
		//{
		//        m_isPattern = false;
		//	break;
		//}
	}

	// OK
	m_confidence = *m_fast_output;
	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool TreeClassifier::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, 1, "ID") != 1)
	{
		Torch::message("TreeClassifier::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("TreeClassifier::load - invalid <ID>, this is not a TreeClassifier model!\n");
		return false;
	}

	// Get the model size
	TensorSize inputSize;
	if (file.taggedRead(&inputSize, "INPUT_SIZE") != 1)
	{
		Torch::message("TreeClassifier::load - failed to read <INPUT_SIZE> field!\n");
		return false;
	}

	// Create the classifier nodes
	int n_nodes;
        if (file.taggedRead(&n_nodes, 1, "N_NODES") != 1)
        {
        	Torch::message("TreeClassifier::load - failed to read <N_NODES> field!\n");
        	return false;
        }
        if (resize(n_nodes) == false)
        {
        	return false;
        }

	// For each node ...
	for (int s = 0; s < n_nodes; s ++)
	{
		// Number of classifiers per node
		int n_trainers;
		if (file.taggedRead(&n_trainers, 1, "N_TRAINERS") != 1)
		{
			Torch::message("TreeClassifier::load - failed to read <N_TRAINERS> field!\n");
		}
		if (resize(s, n_trainers) == false)
		{
			return false;
		}

		// Load each classifier
		for (int n = 0; n < n_trainers; n ++)
		{
		        // Get the classifier's ID
			int id;
			if (file.taggedRead(&id, 1, "ID") != 1)
			{
				Torch::message("TreeClassifier::load - failed to read the Classifier's <ID> field!\n");
				return false;
			}
			Classifier* classifier = (Classifier *) MachineManager::getInstance().get(id);
			if (classifier == 0)
			{
				Torch::message("TreeClassifier::load - invalid classifier ID!\n");
				return false;
			}

			// Load the classifier's parameters
			if (classifier->loadFile(file) == false)
			{
			        Torch::message("TreeClassifier::load - the [%d/%d]-[%d/%d] classifier could not be loaded!\n",
                                        s + 1, n_nodes, n + 1, n_trainers);
			        return false;
			}
			if (setClassifier(s, n, classifier) == false)
			{
				return false;
			}

			// Load the threshold of this classifier
			double threshold;
			if (file.taggedRead(&threshold, 1, "THRESHOLD") != 1)
			{
				Torch::message("TreeClassifier::load - failed to read <THRESHOLD> field!\n");
				return false;
			}
			if (setThreshold(s, n, threshold) == false)
			{
				return false;
			}
		}
	}

	// OK, force the model size to all classifiers
	setSize(inputSize);
	return true;
}

bool TreeClassifier::saveFile(File& file) const
{
	// Write the ID
	const int id = getID();
	if (file.taggedWrite(&id, 1, "ID") != 1)
	{
		Torch::message("TreeClassifier::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the model size
	if (file.taggedWrite(&m_size, "INPUT_SIZE") != 1)
	{
		Torch::message("TreeClassifier::save - failed to write <INPUT_SIZE> field!\n");
		return false;
	}

	// Write the number of nodes
	if (file.taggedWrite(&m_n_nodes, 1, "N_NODES") != 1)
        {
        	Torch::message("TreeClassifier::save - failed to write <N_NODES> field!\n");
        	return false;
        }

	// For each node ...
	for (int s = 0; s < m_n_nodes; s ++)
	{
		const Node& node = m_nodes[s];

		// Number of classifiers per node
		if (file.taggedWrite(&node.m_n_classifiers, 1, "N_TRAINERS") != 1)
		{
			Torch::message("TreeClassifier::save - failed to write <N_TRAINERS> field!\n");
		}

		// Save each classifier
		for (int n = 0; n < node.m_n_classifiers; n ++)
		{
			const Classifier* classifier = node.m_classifiers[n];

			// Write the classifier's ID
			int id = classifier->getID();
			if (file.taggedWrite(&id, 1, "ID") != 1)
			{
				Torch::message("TreeClassifier::save - failed to write the classifier's <ID> field!\n");
				return false;
			}

			// Save the classifier's parameters
			if (classifier->saveFile(file) == false)
			{
			        Torch::message("TreeClassifier::load - the [%d/%d]-[%d/%d] classifier could not be saved!\n",
                                        s + 1, m_n_nodes, n + 1, node.m_n_classifiers);
				return false;
			}

			// Save the threshold of this classifier
			if (file.taggedWrite(&node.m_thresholds[n], 1, "THRESHOLD") != 1)
			{
				Torch::message("TreeClassifier::save - failed to write <THRESHOLD> field!\n");
				return false;
			}
		}
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
// classifier manipulation (add/remove/set classifiers or nodes)
// NB: The Tree will take care of deallocating the SET classifiers!

bool TreeClassifier::resize(int n_nodes)
{
	if (n_nodes < 1)
	{
		Torch::message("TreeClassifier::resize - invalid parameters!\n");
		return false;
	}

	deallocate();

	// OK
	m_n_nodes = n_nodes;
	m_nodes = new Node[m_n_nodes];
	return true;
}

bool TreeClassifier::resize(int i_node, int n_classifiers)
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::message("TreeClassifier::resize(,) - invalid parameters!\n");
		return false;
	}

	// OK
	Node& node = m_nodes[i_node];
	return node.resize(n_classifiers);
}

bool TreeClassifier::setClassifier(int i_node, int i_classifier, Classifier* classifier)
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::message("TreeClassifier::setClassifier - invalid parameters!\n");
		return false;
	}

	// OK
	Node& node = m_nodes[i_node];
	if (node.setClassifier(i_classifier, classifier))
	{
		// Don't forget to set the model size to this classifier too
		classifier->setSize(m_size);
		return true;
	}
	return false;
}

bool TreeClassifier::setThreshold(int i_node, int i_classifier, double threshold)
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::message("TreeClassifier::setThreshold - invalid parameters!\n");
		return false;
	}

	// OK
	Node& node = m_nodes[i_node];
	return node.setThreshold(i_classifier, threshold);
}

//////////////////////////////////////////////////////////////////////////
// Access functions

int TreeClassifier::getNoClassifiers(int i_node) const
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::error("TreeClassifier::getNoClassifiers - invalid parameters!\n");
	}

	return m_nodes[i_node].m_n_classifiers;
}

Classifier* TreeClassifier::getClassifier(int i_node, int i_classifier)
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::error("TreeClassifier::getClassifier - invalid parameters!\n");
	}

	return m_nodes[i_node].getClassifier(i_classifier);
}

const Classifier* TreeClassifier::getClassifier(int i_node, int i_classifier) const
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::error("TreeClassifier::getClassifier - invalid parameters!\n");
	}

	return m_nodes[i_node].getClassifier(i_classifier);
}

double TreeClassifier::getThreshold(int i_node, int i_classifier) const
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::error("TreeClassifier::getThreshold - invalid parameters!\n");
	}

	return m_nodes[i_node].getThreshold(i_classifier);
}


//////////////////////////////////////////////////////////////////////////
}
