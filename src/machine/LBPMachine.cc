#include "LBPMachine.h"
#include "ipLBP8R.h"
#include "ipLBP4R.h"
#include "Tensor.h"
#include "File.h"

namespace Torch {

//////////////////////////////////////////////////////////////////////////
// Constructor

LBPMachine::LBPMachine(LBPType lbp_type)
	:	Machine(),
		m_lut(0), m_lut_size(0), m_x(0), m_y(0),
		m_lbp_type(lbp_type), m_ip_lbp(0)
{
	setLBPType(lbp_type);

	// Allocate the output
	m_output = new DoubleTensor(1);
}

//////////////////////////////////////////////////////////////////////////
// Destructor

LBPMachine::~LBPMachine()
{
        // Cleanup
	delete[] m_lut;
	delete m_ip_lbp;
	delete m_output;
}

//////////////////////////////////////////////////////////////////////////
// Change the LBP code type to work with

bool LBPMachine::setLBPType(LBPType lbp_type)
{
	// Delete the old data
	delete m_ip_lbp;
	delete[] m_lut;
	m_lut = 0;
	m_lut_size = 0;
	m_ip_lbp = 0;

	// Create the new ipLBP and allocate the LUT accordingly
	m_ip_lbp = makeIpLBP(lbp_type);
	m_lbp_type = lbp_type;
	m_ip_lbp->setInputSize(m_model_w, m_model_h);

	m_lut_size = m_ip_lbp->getMaxLabel();
	m_lut = new double[m_lut_size];
	for (int i = 0; i < m_lut_size; i ++)
	{
		m_lut[i] = 0.0;
	}

	// OK
	return true;
}

bool LBPMachine::setLBPRadius(int lbp_radius)
{
	if (m_ip_lbp == 0)
	{
		return false;
	}

	// OK
	return m_ip_lbp->setR(lbp_radius);
}

//////////////////////////////////////////////////////////////////////////
// Change the model size (need to set the model size to the <ipLBP>) - overriden

bool LBPMachine::setModelSize(int model_w, int model_h)
{
	if (Machine::setModelSize(model_w, model_h) == false)
	{
		return false;
	}

	// OK
	if (m_ip_lbp != 0)
	{
		m_ip_lbp->setInputSize(model_w, model_h);
	}
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Get the LBP radius

int LBPMachine::getLBPRadius() const
{
	if (m_ip_lbp == 0)
	{
		Torch::message("LBPMachine::getLBRadius - the ipLBP was not set!\n");
		return 0;
	}

	return m_ip_lbp->getR();
}

//////////////////////////////////////////////////////////////////////////
// Creates an <ipLBP> associated to the given LBP code type

ipLBP* LBPMachine::makeIpLBP(LBPType lbp_type)
{
	ipLBP* ip_lbp = 0;

	switch(lbp_type)
	{
	case LBP4RCenter:	// 4R LBP compared to the center value
		ip_lbp = new ipLBP4R;
		ip_lbp->setBOption("ToAverage", false);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP4RAverage:	// 4R LBP compared to the average value
		ip_lbp = new ipLBP4R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP4RAverageAddBit:// 4R LBP compared to the average value + extra bit
		ip_lbp = new ipLBP4R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", true);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP8RCenter:	// 8R LBP compared to the center value
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", false);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP8RCenter_RI:	// 8R LBP compared to the center value, rotation invariant
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", false);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", true);
		break;

	case LBP8RCenter_U2:	// 8R LBP compared to the center value, uniform
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", false);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", true);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP8RCenter_U2RI:	// 8R LBP compared to the center value, uniform, rotation invariant
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", false);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", true);
		ip_lbp->setBOption("RotInvariant", true);
		break;

	case LBP8RAverage:	// 8R LBP compared to the average value
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP8RAverage_RI:	// 8R LBP compared to the average value, rotation invariant
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", true);
		break;

	case LBP8RAverage_U2:	// 8R LBP compared to the average value, uniform
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", true);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	case LBP8RAverage_U2RI:	// 8R LBP compared to the average value, uniform, rotation invariant
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", false);
		ip_lbp->setBOption("Uniform", true);
		ip_lbp->setBOption("RotInvariant", true);
		break;

	case LBP8RAverageAddBit:// 8R LBP compared to the average value + extra bit
		ip_lbp = new ipLBP8R;
		ip_lbp->setBOption("ToAverage", true);
		ip_lbp->setBOption("AddAvgBit", true);
		ip_lbp->setBOption("Uniform", false);
		ip_lbp->setBOption("RotInvariant", false);
		break;

	default:
		Torch::error("LBPMachine::makeIpLBP - invalid LBP type!\n");
	}

	return ip_lbp;
}

//////////////////////////////////////////////////////////////////////////
// Set the machine's parameters

bool LBPMachine::setLUT(double* lut, int lut_size)
{
	if (lut_size != m_lut_size)
	{
		Torch::message("LBPMachine::setLUT - invalid lut size!. <setLBPType>  wasn't called?!\n");
		return false;
	}

	// OK
	for (int i = 0; i < m_lut_size; i ++)
	{
		m_lut[i] = lut[i];
	}
	return true;
}

bool LBPMachine::setXY(int x, int y)
{
	m_x = x;
	m_y = y;
	return true;
}

//////////////////////////////////////////////////////////////////////////
// Process the input tensor

bool LBPMachine::forward(const Tensor& input)
{
	if (m_ip_lbp == 0 || m_lut == 0)
	{
		Torch::message("LBPMachine::forward - invalid ipLBP or LUT!\n");
		return false;
	}

	// Initialize the ipLBP (the input size was already set in the <setModelSize> and <setLBPType> functions!)
	if (m_ip_lbp->setXY(m_x, m_y) == false)
	{
	        Torch::message("LBPMachine::forward - failed to initialize the ipLBP!\n");
		return false;
	}

	// Get the LBP code
	if (m_ip_lbp->process(input) == false)
	{
		Torch::message("LBPMachine::forward - failed to get the LBP code!\n");
		return false;
	}

	// OK
	const int lbp = m_ip_lbp->getLBP();
	m_output->set(0, m_lut[lbp]);
	return true;
}

///////////////////////////////////////////////////////////
// Loading/Saving the content from files (\emph{not the options}) - overriden

bool LBPMachine::loadFile(File& file)
{
	// Check the ID
	int id;
	if (file.taggedRead(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <ID> field!\n");
		return false;
	}
	if (id != getID())
	{
		Torch::message("LBPMachine::load - invalid <ID>, this is not a LBPMachine model!\n");
		return false;
	}

	// Read the LBP type - to know what kind of LBP features to use!
	int lbp_type;
	if (file.taggedRead(&lbp_type, sizeof(int), 1, "LBP_TYPE") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <LBP_TYPE> field!\n");
		return false;
	}
	if (setLBPType((LBPType)lbp_type) == false)
	{
		Torch::message("LBPMachine::load - invalid <LBP_TYPE field!\n");
		return false;
	}

	// Read the LBP radius
	int lbp_radius;
	if (file.taggedRead(&lbp_radius, sizeof(int), 1, "LBP_R") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <LBP_R> field!\n");
		return false;
	}
	if (m_ip_lbp->setR(lbp_radius) == false)
	{
		Torch::message("LBPMachine::load - invalid <LBP_R> field!\n");
		return false;
	}

	// Read the machine parameters
	if (file.taggedRead(&m_x, sizeof(int), 1, "LOCATION_X") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <LOCATION_X> field!\n");
		return false;
	}
	if (file.taggedRead(&m_y, sizeof(int), 1, "LOCATION_Y") != 1)
	{
		Torch::message("LBPMachine::load - failed to read <LOCATION_Y> field!\n");
		return false;
	}
	const int ret = file.taggedRead(m_lut, sizeof(double), m_lut_size, "LUT");
	if (ret != m_lut_size)
	{
	        Torch::message("LBPMachine::load - failed to read <LUT> field!\n");
		return false;
	}

	// OK
	return true;
}

bool LBPMachine::saveFile(File& file) const
{
	if (m_ip_lbp == 0 || m_lut == 0)
	{
		Torch::message("LBPMachine::save - nothing to save!\n");
		return false;
	}

	// Write the machine ID
	const int id = getID();
	if (file.taggedWrite(&id, sizeof(int), 1, "ID") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <ID> field!\n");
		return false;
	}

	// Write the LBP type & radius - to know what kind of LBP features to use!
	int lbp_type = (int)m_lbp_type;
	if (file.taggedWrite(&lbp_type, sizeof(int), 1, "LBP_TYPE") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <LBP_TYPE> field!\n");
		return false;
	}
	int lbp_radius = m_ip_lbp->getR();
	if (file.taggedWrite(&lbp_radius, sizeof(int), 1, "LBP_R") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <LBP_R> field!\n");
		return false;
	}

	// Write the machine parameters
	if (file.taggedWrite(&m_x, sizeof(int), 1, "LOCATION_X") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <LOCATION_X> field!\n");
		return false;
	}
	if (file.taggedWrite(&m_y, sizeof(int), 1, "LOCATION_Y") != 1)
	{
		Torch::message("LBPMachine::save - failed to write <LOCATION_Y> field!\n");
		return false;
	}
	if (file.taggedWrite(m_lut, sizeof(double), m_lut_size, "LUT") != m_lut_size)
	{
		Torch::message("LBPMachine::save - failed to write <LUT> field!\n");
		return false;
	}

	// OK
	return true;
}

//////////////////////////////////////////////////////////////////////////
}
