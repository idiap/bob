#include "ipSWEvaluator.h"
#include "Classifier.h"
#include "Machines.h"
#include "Tensor.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ipSWEvaluator::ipSWEvaluator()
	: 	ipCore(),
                m_classifier(0),
                m_input_copy(0)
{
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ipSWEvaluator::~ipSWEvaluator()
{
        delete m_classifier;
}

/////////////////////////////////////////////////////////////////////////
// Set the classifier to load from some file

bool ipSWEvaluator::setClassifier(const char* filename)
{
        // Load the machine
        Machine* machine = Torch::loadMachineFromFile(filename);
        if (machine == 0)
        {
                Torch::message("ipSWEvaluator::setClassifier - invalid model file!\n");
                return false;
        }

        // Check if it's really a classifier
        Classifier* classifier = dynamic_cast<Classifier*>(machine);
        if (classifier == 0)
        {
                delete machine;
                Torch::message("ipSWEvaluator::setClassifier - the loaded model is not a classifier!\n");
                return false;
        }

        // OK
        delete m_classifier;
        m_classifier = classifier;
        return true;
}

/////////////////////////////////////////////////////////////////////////
// Access functions

inline bool ipSWEvaluator::isPattern() const
{
        if (m_classifier == 0)
        {
                Torch::error("ipSWEvaluator::isPattern - no valid classifier specified!\n");
        }
        return m_classifier->isPattern();
}

inline double ipSWEvaluator::getConfidence() const
{
        if (m_classifier == 0)
        {
                Torch::error("ipSWEvaluator::getConfidence - no valid classifier specified!\n");
        }
        return m_classifier->getConfidence();
}

inline int ipSWEvaluator::getModelWidth() const
{
        if (m_classifier == 0)
        {
                Torch::error("ipSWEvaluator::getModelWidth - no valid classifier specified!\n");
        }
        return m_classifier->getSize().size[1];
}

inline int ipSWEvaluator::getModelHeight() const
{
        if (m_classifier == 0)
        {
                Torch::error("ipSWEvaluator::getModelHeight - no valid classifier specified!\n");
        }
        return m_classifier->getSize().size[0];
}

const Classifier& ipSWEvaluator::getClassifier() const
{
	if (m_classifier == 0)
        {
                Torch::error("ipSWEvaluator::getClassifier - no valid classifier specified!\n");
        }
        return *m_classifier;
}

Classifier& ipSWEvaluator::getClassifier()
{
	if (m_classifier == 0)
        {
                Torch::error("ipSWEvaluator::getClassifier - no valid classifier specified!\n");
        }
        return *m_classifier;
}

/////////////////////////////////////////////////////////////////////////
// Check if the input tensor has the right dimensions and type - overriden

inline bool ipSWEvaluator::checkInput(const Tensor& input) const
{
        return  m_classifier != 0 &&
                (input.nDimension() == 2 || input.nDimension() == 3);
}

/////////////////////////////////////////////////////////////////////////
// Allocate (if needed) the output tensors given the input tensor dimensions - overriden

inline bool ipSWEvaluator::allocateOutput(const Tensor& input)
{
        // No output is generated, the Machine has the output!
        return true;
}

/////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated) - overriden

inline bool ipSWEvaluator::processInput(const Tensor& input)
{
        // Keep a copy of the input tensor (to pass to the Classifier)
        m_input_copy = &input;

        // OK
        return true;
}

/////////////////////////////////////////////////////////////////////////
/// Change the region of the input tensor to process

inline void ipSWEvaluator::setRegion(const TensorRegion& region)
{
	m_classifier->setRegion(region);
	if (m_classifier->forward(*m_input_copy) == false)
	{
		error("ipSWEvaluator::setRegion - the classifier cannot process the input tensor!\n");
	}
}

/////////////////////////////////////////////////////////////////////////

}
