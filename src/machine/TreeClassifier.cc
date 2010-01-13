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
		m_mu = new double[m_n_classifiers];
		m_stdv = new double[m_n_classifiers];
		m_sigma = new double[m_n_classifiers];
		m_childs = new int[m_n_classifiers+1];
		for (int i = 0; i < m_n_classifiers; i ++)
		{
			m_classifiers[i] = 0;
			m_thresholds[i] = 0.0;
			m_mu[i] = 0.0;
			m_stdv[i] = 0.0;
			m_sigma[i] = 0.0;
			m_childs[i] = -1;
		}
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Deallocate memory

void TreeClassifier::Node::deallocate()
{
	delete[] m_classifiers;
	delete[] m_sigma;
	delete[] m_stdv;
	delete[] m_mu;
	delete[] m_thresholds;
	delete[] m_childs;
	m_classifiers = 0;
	m_mu = 0;
	m_stdv = 0;
	m_sigma = 0;
	m_thresholds = 0;
	m_childs = 0;
	m_n_classifiers = 0;
}

//////////////////////////////////////////////////////////////////////////
// Set a new classifier

bool TreeClassifier::Node::setClassifier(int i_classifier, Classifier* classifier, double mu, double stdv)
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers || classifier == 0)
	{
		return false;
	}

	m_classifiers[i_classifier] = classifier;
	m_mu[i_classifier] = mu;
	m_stdv[i_classifier] = stdv;
	m_sigma[i_classifier] = sqrt(stdv);

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
			
bool TreeClassifier::Node::setChild(int i_classifier, int child_node)
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers+1)
	{
		Torch::error("TreeClassifier::Node::setChild - invalid parameters!\n");
		return false;
	}

	// OK
	m_childs[i_classifier] = child_node;
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

int TreeClassifier::Node::getChild(int i_classifier) const
{
	if (i_classifier < 0 || i_classifier >= m_n_classifiers+1)
	{
		Torch::error("TreeClassifier::Node::getChild - invalid parameters!\n");
	}

	return m_childs[i_classifier];
}

//////////////////////////////////////////////////////////////////////////
// Constructor

TreeClassifier::TreeClassifier()
	:	Classifier(),
		m_nodes(0), m_n_nodes(0), m_n_classes(0),
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

double TreeClassifier::normal(double x, double mu, double sigma, double stdv)
{
   	if(mu == 0.0 && stdv == 1.0) return x;

	double z = x - mu;

	return sqrt(1.0/(sigma*2.0*M_PI)) * exp(-0.5 * z * z / stdv);
}

//////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool TreeClassifier::forward(const Tensor& input)
{
	m_isPattern = false;
	m_patternClass = 0;
	double output = 0.0;

	int i_node = 0; // start on the root node (alwas the first one)
	bool leaf_node_reached = false;

	while(leaf_node_reached == false)
	{
		// if the node is a leaf node
		if(i_node >= m_n_nodes)
		{
			leaf_node_reached = true;

			int leaf_node_class = i_node - m_n_nodes;
			if(leaf_node_class == 0)
			{
				// the first leaf node is always assumed to be the non-pattern one

				m_isPattern = false;
				m_patternClass = 0;
				*m_fast_output = 0.0;
				m_confidence = 0.0;
			}
			else
			{
				m_isPattern = true;
				m_patternClass = leaf_node_class;
				*m_fast_output = output;
				m_confidence = output;
			}
		}
		else
		{
			const Node& node = m_nodes[i_node];

			//
			int argmax_output = -1;
			double max_output = -1000;

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

				if(classifier->isPattern())
				{
					double z = classifier->getOutput().get(0);
					
					if(z >= node.m_thresholds[n]) 
					{
						// The score should be normalized here
						double znorm = normal(z, node.m_mu[n], node.m_sigma[n], node.m_stdv[n]);

						//print("(%d %d): norm(%g, %g, %g, %g) -> %g\n", i_node, n, z, node.m_mu[n], node.m_sigma[n], node.m_stdv[n], znorm);

						if(znorm > max_output) 
						{
							max_output = znorm;
							argmax_output = n;
						}
					}
				}
			}

			if(argmax_output == -1)
			{
				// if the input is rejected by all the classifiers
				i_node = node.getChild(node.m_n_classifiers);
			}
			else
			{
				i_node = node.getChild(argmax_output);

				output = max_output;
			}
		}
	}

	// OK
	m_confidence = *m_fast_output;

	return true;

	/*
	for (int s = 0; s < m_n_nodes; s ++)
	{
		const Node& node = m_nodes[s];

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

			if(classifier->getOutput().get(0) > node.m_thresholds[n]) 
			{
				output += classifier->getOutput().get(0);
				m_isPattern = true;
			}

			//
			//if(classifier->isPattern() == true) m_isPattern = true;
		}

	}

	// Check if rejected
	*m_fast_output = output;

	// OK
	m_confidence = *m_fast_output;
	return true;
	*/
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

	int n_classes;
        if (file.taggedRead(&n_classes, 1, "N_CLASSES") != 1)
        {
        	Torch::message("TreeClassifier::load - failed to read <N_CLASSES> field!\n");
        	return false;
        }

	setClasses(n_classes);

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

			// Load normalization
			double mu;
			double stdv;
			if (file.taggedRead(&mu, 1, "MU") != 1)
			{
				Torch::message("TreeClassifier::load - failed to read <MU> field!\n");
				return false;
			}
			if (file.taggedRead(&stdv, 1, "STDV") != 1)
			{
				Torch::message("TreeClassifier::load - failed to read <STDV> field!\n");
				return false;
			}

			if (setClassifier(s, n, classifier, mu, stdv) == false)
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

		// Load each classifier
		for (int n = 0; n < n_trainers+1; n ++)
		{
			// Load the child associated to this classifier
			int child;
			if (file.taggedRead(&child, 1, "CHILD") != 1)
			{
				Torch::message("TreeClassifier::load - failed to read <CHILD> field!\n");
				return false;
			}
			if (setChild(s, n, child) == false)
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

	// Write the number of classes
	if (file.taggedWrite(&m_n_classes, 1, "N_CLASSES") != 1)
        {
        	Torch::message("TreeClassifier::save - failed to write <N_CLASSES> field!\n");
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
			        Torch::message("TreeClassifier::save - the [%d/%d]-[%d/%d] classifier could not be saved!\n",
                                        s + 1, m_n_nodes, n + 1, node.m_n_classifiers);
				return false;
			}

			// Save the normalisation for this classifier
			if (file.taggedWrite(&node.m_mu[n], 1, "MU") != 1)
			{
				Torch::message("TreeClassifier::save - failed to write <MU> field!\n");
				return false;
			}
			if (file.taggedWrite(&node.m_stdv[n], 1, "STDV") != 1)
			{
				Torch::message("TreeClassifier::save - failed to write <STDV> field!\n");
				return false;
			}

			// Save the threshold of this classifier
			if (file.taggedWrite(&node.m_thresholds[n], 1, "THRESHOLD") != 1)
			{
				Torch::message("TreeClassifier::save - failed to write <THRESHOLD> field!\n");
				return false;
			}

		}
		for (int n = 0; n < node.m_n_classifiers+1; n ++)
		{
			// Save the child of this classifier
			if (file.taggedWrite(&node.m_childs[n], 1, "CHILD") != 1)
			{
				Torch::message("TreeClassifier::save - failed to write <CHILD> field!\n");
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

int TreeClassifier::getChild(int i_node, int i_classifier) const
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::message("TreeClassifier::setClassifier - invalid parameters!\n");
		return false;
	}

	// OK
	Node& node = m_nodes[i_node];
	return node.getChild(i_classifier);
}

bool TreeClassifier::setClassifier(int i_node, int i_classifier, Classifier* classifier, double mu, double stdv)
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::message("TreeClassifier::setClassifier - invalid parameters!\n");
		return false;
	}

	// OK
	Node& node = m_nodes[i_node];
	if (node.setClassifier(i_classifier, classifier, mu, stdv))
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
		
bool TreeClassifier::setChild(int i_node, int i_classifier, int child_node)
{
	if (i_node < 0 || i_node >= m_n_nodes)
	{
		Torch::message("TreeClassifier::setChild - invalid parameters!\n");
		return false;
	}

	// OK
	Node& node = m_nodes[i_node];
	return node.setChild(i_classifier, child_node);
}

bool TreeClassifier::setClasses(int n_classes)
{
	m_n_classes = n_classes;

	return true; 
}

int TreeClassifier::getClasses() const 
{
	return m_n_classes;
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
