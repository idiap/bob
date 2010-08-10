#include "scanning/ContextDataSet.h"
#include "core/File.h"

namespace Torch
{

/////////////////////////////////////////////////////////////////////////
// Constructor

ContextDataSet::ContextDataSet(int pf_feature)
	: 	DataSet(Tensor::Double, true, Tensor::Double),
		m_feature(pf_feature),
		m_contexts(0),
		m_masks(0),
		m_capacity(0),
		m_target_neg(1),
		m_target_pos(1)
{
	m_target_neg.fill(0.1);
	m_target_pos.fill(0.9);

	reset(pf_feature);
}

/////////////////////////////////////////////////////////////////////////
// Destructor

ContextDataSet::~ContextDataSet()
{
	cleanup();
}

/////////////////////////////////////////////////////////////////////////
// Delete stored contexts

void ContextDataSet::cleanup()
{
	for (long i = 0; i < m_capacity; i ++)
	{
		delete m_contexts[i];
	}
	delete[] m_contexts;
	m_contexts = 0;

	delete[] m_masks;
	m_masks = 0;

	m_capacity = 0;
	m_n_examples = 0;
}

/////////////////////////////////////////////////////////////////////////
// Reset to new context feature

void ContextDataSet::reset(int pf_feature)
{
	m_feature = pf_feature;
}

/////////////////////////////////////////////////////////////////////////
// Access examples - overriden

Tensor* ContextDataSet::getExample(long index)
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ContextDataSet::getExample - invalid index!\n");
	}
	return &m_contexts[index]->m_features[m_feature];
}

Tensor& ContextDataSet::operator()(long index)
{
	return *getExample(index);
}

/////////////////////////////////////////////////////////////////////////
// Access targets - overriden

Tensor* ContextDataSet::getTarget(long index)
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ContextDataSet::getTarget - invalid index!\n");
	}

	const unsigned char mask = m_masks[index];
	return mask == Negative ? &m_target_neg : &m_target_pos;
}

void ContextDataSet::setTarget(long index, Tensor* target)
{
	// Nothing to do here!
}

/////////////////////////////////////////////////////////////////////////
// Context access in the distribution

const Context* ContextDataSet::getContext(long index) const
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ContextDataSet::getContext - invalid index!\n");
	}

	return m_contexts[index];
}

bool ContextDataSet::isPosContext(long index) const
{
	if (isIndex(index, m_n_examples) == false)
	{
		error("ContextDataSet::isPosContext - invalid index!\n");
	}

	return m_masks[index] != Negative;
}

/////////////////////////////////////////////////////////////////////////
// Clear cumulated contexts

void ContextDataSet::clear()
{
	m_n_examples = 0;
}

/////////////////////////////////////////////////////////////////////////
// Cumulate a new context (negative, positive or ground truth)

void ContextDataSet::cumulate(bool positive, const Context& context)
{
	static const int increment = 1024;

	if (m_n_examples >= m_capacity)
	{
		m_contexts = resize(m_contexts, m_capacity, increment);
		m_masks = resize(m_masks, m_capacity, increment);
		m_capacity += increment;
	}

	*m_contexts[m_n_examples] = context;
	m_masks[m_n_examples] = positive == true ? Positive : Negative;
	m_n_examples ++;
}

void ContextDataSet::cumulate(const Context& gt_context)
{
	static const int increment = 1024;

	// Add the detection
	if (m_n_examples >= m_capacity)
	{
		m_contexts = resize(m_contexts, m_capacity, increment);
		m_masks = resize(m_masks, m_capacity, increment);
		m_capacity += increment;
	}

	*m_contexts[m_n_examples] = gt_context;
	m_masks[m_n_examples] = GroundTruth;
	m_n_examples ++;
}

/////////////////////////////////////////////////////////////////////////
// Resize some distribution to fit new samples

Context** ContextDataSet::resize(Context** old_data, long capacity, long increment)
{
	Context** new_data = new Context*[capacity + increment];
	for (long i = 0; i < capacity; i ++)
	{
		new_data[i] = old_data[i];
	}
	for (long i = 0; i < increment; i ++)
	{
		new_data[capacity + i] = new Context;
	}
	delete[] old_data;

	return new_data;
}

unsigned char* ContextDataSet::resize(unsigned char* old_data, long capacity, long increment)
{
	unsigned char* new_data = new unsigned char[capacity + increment];
	for (long i = 0; i < capacity; i ++)
	{
		new_data[i] = old_data[i];
	}
	for (long i = 0; i < increment; i ++)
	{
		new_data[capacity + i] = 0x00;
	}
	delete[] old_data;

	return new_data;
}

/////////////////////////////////////////////////////////////////////////
// Save the distribution

bool ContextDataSet::save(const char* dir_data, const char* name) const
{
	char str[1024];

	// Save separate contexts for plotting
	{
		// Negative samples
		sprintf(str, "%s_%s_neg", dir_data, name);
		if (save(str, Negative) == false)
		{
			return false;
		}

		// Positive samples
		sprintf(str, "%s_%s_pos", dir_data, name);
		if (save(str, Positive) == false)
		{
			return false;
		}

		// Ground truth samples
		sprintf(str, "%s_%s_gt", dir_data, name);
		if (save(str, GroundTruth) == false)
		{
			return false;
		}
	}

	// Save altogether in a binary format
	{
		// Open the file
		File file;
		sprintf(str, "%s_%s.distribution", dir_data, name);
		if (file.open(str, "w") == false)
		{
			return false;
		}

		// Write the number of samples
		if (file.taggedWrite(&m_n_examples, 1, "NO") != 1)
		{
			print("ContextDataSet::save - failed to write <NO> tag!\n");
			return false;
		}

		// Write the masks (positive, negative, ground truth)
		if (file.taggedWrite(m_masks, m_n_examples, "MASKS") != m_n_examples)
		{
			print("ContextDataSet::save - failed to write <MASKS> tag!\n");
			return false;
		}

		// Write each context
		for (long s = 0; s < m_n_examples; s ++)
		{
			const Context* context = m_contexts[s];

			file.printf("%d %d %d %d %lf %d ",
				context->m_pattern.m_x, context->m_pattern.m_y,
				context->m_pattern.m_w, context->m_pattern.m_h,
				context->m_pattern.m_confidence, context->m_pattern.m_activation);

			for (int f = 0; f < NoFeatures; f ++)
			{
				const int size = FeatureSizes[f];
				const double* data = (const double*)context->m_features[f].dataR();
				for (int k = 0; k < size; k ++)
				{
					file.printf("%lf ", data[k]);
				}
			}
		}

		file.close();
	}

	// OK
	return true;
}

bool ContextDataSet::save(const char* basename, unsigned char mask) const
{
	// Open the files
	for (int f = 0; f < NoFeatures; f ++)
	{
		char str[1024];
		sprintf(str, "%s_%s.data", basename, FeatureNames[f]);

		File file;
		CHECK_FATAL(file.open(str, "w+") == true);

		const int fsize = FeatureSizes[f];
		for (int i = 0; i < m_n_examples; i ++)
			if (m_masks[i] == mask)
			{
				const double* values = (const double*)m_contexts[i]->m_features[f].dataR();
				for (int k = 0; k < fsize; k ++)
				{
					file.printf("%lf\t", values[k]);
				}
				file.printf("\n");
			}

		file.close();
	}

	return true;
}

/////////////////////////////////////////////////////////////////////////
// Load the distribution

bool ContextDataSet::load(const char* dir_data, const char* name)
{
	cleanup();

	// Open the file
	char str[1024];
	File file;
	sprintf(str, "%s_%s.distribution", dir_data, name);
	if (file.open(str, "r") == false)
	{
		return false;
	}

	// Read the number of samples
	if (file.taggedRead(&m_n_examples, 1, "NO") != 1)
	{
		print("ContextDataSet::load - failed to read <NO> tag!\n");
		return false;
	}

	// Allocate data
	m_masks = new unsigned char[m_n_examples];
	m_contexts = new Context*[m_n_examples];
	for (long s = 0; s < m_n_examples; s ++)
	{
		m_contexts[s] = new Context;
	}
	m_capacity = m_n_examples;

	// Read the masks (positive, negative, ground truth
	if (file.taggedRead(m_masks, m_n_examples, "MASKS") != m_n_examples)
	{
		print("ContextDataSet::load - failed to read <MASKS> tag!\n");
		return false;
	}

	// Read each context
	for (long s = 0; s < m_n_examples; s ++)
	{
		Context* context = m_contexts[s];

		if (	file.scanf("%d", &context->m_pattern.m_x) != 1 ||
			file.scanf("%d", &context->m_pattern.m_y) != 1 ||
			file.scanf("%d", &context->m_pattern.m_w) != 1 ||
			file.scanf("%d", &context->m_pattern.m_h) != 1 ||
			file.scanf("%lf", &context->m_pattern.m_confidence) != 1 ||
			file.scanf("%d", &context->m_pattern.m_activation) != 1)
		{
			print("ContextDataSet::load - failed to read context [%d/%d]!\n", s + 1, m_n_examples);
			return false;
		}

		for (int f = 0; f < NoFeatures; f ++)
		{
			const int size = FeatureSizes[f];
			double* data = (double*)context->m_features[f].dataW();
			for (int k = 0; k < size; k ++)
			{
				if (file.scanf("%lf", &data[k]) != 1)
				{
					print("ContextDataSet::load - failed to read context [%d/%d]!\n", s + 1, m_n_examples);
					return false;
				}
			}
		}
	}

	file.close();

	// OK
	return true;
}

/////////////////////////////////////////////////////////////////////////

}
