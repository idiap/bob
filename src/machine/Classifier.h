#ifndef _TORCH5SPRO_CLASSIFIER_H_
#define _TORCH5SPRO_CLASSIFIER_H_

#include "Machine.h"	// <Classifier> is a <Machine>

namespace Torch
{
	//////////////////////////////////////////////////////////////////////////////////////
	// Torch::Classifier:
	//	Generic machine for classifying some pattern:
	//              - two classes: feature vs. non feature
	//              - multi-class
	//
	// TODO: doxygen header!
	//////////////////////////////////////////////////////////////////////////////////////

	class Classifier : public Machine
	{
	public:

		/// Constructor
		Classifier();

		/// Destructor
		virtual ~Classifier();

		///////////////////////////////////////////////////////////
		// Access functions

                bool                    isPattern() const { return m_isPattern; }
                int                     getPatternClass() const { return m_patternClass; }
                double                  getConfidence() const { return m_confidence; }
		///////////////////////////////////////////////////////////

		virtual bool            setThreshold(double threshold) { return true; }

	protected:

		///////////////////////////////////////////////////////////////
		// Attributes

		// Pattern detected ?
		bool                    m_isPattern;

		// Multi-class: what class some detected pattern belongs to?
                int                     m_patternClass;

                // Confidence (probability measure, score) on the result
                double                  m_confidence;
	};
}

#endif
