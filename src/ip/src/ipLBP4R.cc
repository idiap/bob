#include "ip/ipLBP4R.h"
#include "core/Tensor.h"
//#include "Scanner.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the 4R LBP code for a generic tensor

#define COMPUTE_LBP4R(tensorType, dataType)\
{\
	const int offset_center = 	(m_region.pos[0] + m_y) * m_input_stride_h +\
					(m_region.pos[1] + m_x) * m_input_stride_w;\
  const tensorType* t_input = static_cast<const tensorType*>(&input);\
\
	dataType tab[4];\
	tab[0] = (*t_input)(offset_center - m_R * m_input_stride_h);\
	tab[1] = (*t_input)(offset_center + m_R * m_input_stride_w);\
	tab[2] = (*t_input)(offset_center + m_R * m_input_stride_h);\
	tab[3] = (*t_input)(offset_center - m_R * m_input_stride_w);\
\
	const dataType center = (*t_input)(offset_center);\
	const dataType cmp_point = m_toAverage ?\
		(dataType)\
                        ( 0.2 * (tab[0] + tab[1] + tab[2] + tab[3] + center + 0.0))\
		:\
		center;\
\
	unsigned char lbp = 0;\
\
	lbp = lbp << 1;\
	if (tab[0] > cmp_point) lbp ++;\
	lbp = lbp << 1;\
	if (tab[1] > cmp_point) lbp ++;\
	lbp = lbp << 1;\
	if (tab[2] > cmp_point) lbp ++;\
	lbp = lbp << 1;\
	if (tab[3] > cmp_point) lbp ++;\
	if (m_addAvgBit == true && m_rot_invariant == false && m_uniform == false)\
	{\
		lbp = lbp << 1;\
		if (center > cmp_point) lbp ++;\
	}\
\
	*m_lbp = m_crt_lut[lbp];\
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Compute the 4R LBP code for a generic tensor (by interpolating using an integral image)

#define COMPUTE_LBP4R_INTEGRAL(tensorType, dataType)\
{\
  const tensorType* t_input = static_cast<const tensorType*>(&input);\
	const int& ii_dx = m_ii_factors.getDx();\
	const int& ii_dy = m_ii_factors.getDy();\
	const int& ii_w = m_ii_factors.getCellW();\
	const int& ii_w1 = m_ii_factors.getCellW1();\
	const int& ii_w12 = m_ii_factors.getCellW12();\
	const int& ii_h = m_ii_factors.getCellH();\
	const int& ii_h1 = m_ii_factors.getCellH1();\
	const int& ii_h12 = m_ii_factors.getCellH12();\
	const int offset_sw 	= 	m_region.pos[0] * m_input_stride_h +\
					m_region.pos[1] * m_input_stride_w;\
	const int offset1 = offset_sw + ii_dx + ii_dy;\
	const dataType P2 = (*t_input)(offset1 + ii_w1);\
	const dataType P3 = (*t_input)(offset1 + ii_w12);\
	const int offset2 = offset1 + ii_h1;\
	const dataType P5 = (*t_input)(offset2);\
	const dataType P6 = (*t_input)(offset2 + ii_w1);\
	const dataType P7 = (*t_input)(offset2 + ii_w12);\
	const dataType P8 = (*t_input)(offset2 + ii_w);\
	const int offset3 = offset1 + ii_h12;\
	const dataType P9 = (*t_input)(offset3);\
	const dataType P10 = (*t_input)(offset3 + ii_w1);\
	const dataType P11 = (*t_input)(offset3 + ii_w12);\
	const dataType P12 = (*t_input)(offset3 + ii_w);\
	const int offset4 = offset1 + ii_h;\
	const dataType P14 = (*t_input)(offset4 + ii_w1);\
	const dataType P15 = (*t_input)(offset4 + ii_w12);\
	dataType tab[4];\
	tab[0] = P7 + P2 - P3 - P6;\
	tab[1] = P12 + P7 - P8 - P11;\
	tab[2] = P15 + P10 - P11 - P14;\
	tab[3] = P10 + P5 - P6 - P9;\
	const dataType center = P11 + P6 - P7 - P10;\
	const dataType cmp_point = m_toAverage ?\
		(dataType)\
                        (0.2 * (tab[0] + tab[1] + tab[2] + tab[3] + center + 0.0))\
		:\
		center;\
\
	unsigned char lbp = 0;\
\
	lbp = lbp << 1;\
	if (tab[0] > cmp_point) lbp ++;\
	lbp = lbp << 1;\
	if (tab[1] > cmp_point) lbp ++;\
	lbp = lbp << 1;\
	if (tab[2] > cmp_point) lbp ++;\
	lbp = lbp << 1;\
	if (tab[3] > cmp_point) lbp ++;\
	if (m_addAvgBit == true && m_rot_invariant == false && m_uniform == false)\
	{\
		lbp = lbp << 1;\
		if (center > cmp_point) lbp ++;\
	}\
	*m_lbp = m_crt_lut[lbp];\
}

///////////////////////////////////////////////////////////////////////////////////////////////////

namespace Torch
{

///////////////////////////////////////////////////////////////////////////////////////////////////
// Constructor

ipLBP4R::ipLBP4R(int R)
	:	ipLBP(4, R)
{
	m_lut_RI = new unsigned short [16];
	m_lut_U2 = new unsigned short [16];
	m_lut_U2RI = new unsigned short [16];
	m_lut_addAvgBit = new unsigned short [32];
	m_lut_normal = new unsigned short [16];

	for (int i = 0; i < 16; i ++)
	{
		m_lut_RI[i] = 0;
		m_lut_U2[i] = 0;
		m_lut_U2RI[i] = 0;
	}
	for (int i = 0; i < 16; i ++)
	{
		m_lut_normal[i] = i;
	}
	for (int i = 0; i < 32; i ++)
	{
		m_lut_addAvgBit[i] = i;
	}

        init_lut_RI();
        init_lut_U2();
        init_lut_U2RI();

        optionChanged(0);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Destructor

ipLBP4R::~ipLBP4R()
{
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Get the maximum possible label

int ipLBP4R::getMaxLabel()
{
	return	m_rot_invariant ?
	       (m_uniform ? 	6 	// Rotation invariant + uniform
		:
		6)	// Rotation invariant
			       :
			       (m_uniform ?	15	// Uniform
				:
				(m_toAverage ?
				 (m_addAvgBit ? 32 : 16)	// i.e. 2^5=32 vs. 2^4=16
				 :
				 16)				// i.e. 2^4=16)
			       );
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Process some input tensor (the input is checked, the outputs are allocated) - overriden

bool ipLBP4R::processInput(const Tensor& input)
{
	// Force interpolation for the multiscale even at the model size
	if (CurrentScanType::getInstance().get() == ScanTypeMultiscale)
	{
			m_need_interp = true;
	}

	// No interpolation needed, the model size is the same as the region size to process!
	if (m_need_interp == false)
	{
		switch (input.getDatatype())
		{
		case Tensor::Char:
			COMPUTE_LBP4R(CharTensor, char);
			break;

		case Tensor::Short:
			COMPUTE_LBP4R(ShortTensor, short);
			break;

		case Tensor::Int:
			COMPUTE_LBP4R(IntTensor, int);
			break;

		case Tensor::Long:
			COMPUTE_LBP4R(LongTensor, long);
			break;

		case Tensor::Float:
			COMPUTE_LBP4R(FloatTensor, float);
			break;

		case Tensor::Double:
			COMPUTE_LBP4R(DoubleTensor, double);
			break;
		}
	}

	// Interpolation needed!
	else
	{
		switch (input.getDatatype())
		{
		case Tensor::Char:
			COMPUTE_LBP4R_INTEGRAL(CharTensor, char);
			break;

		case Tensor::Short:
			COMPUTE_LBP4R_INTEGRAL(ShortTensor, short);
			break;

		case Tensor::Int:
			COMPUTE_LBP4R_INTEGRAL(IntTensor, int);
			break;

		case Tensor::Long:
			COMPUTE_LBP4R_INTEGRAL(LongTensor, long);
			break;

		case Tensor::Float:
			COMPUTE_LBP4R_INTEGRAL(FloatTensor, float);
			break;

		case Tensor::Double:
			COMPUTE_LBP4R_INTEGRAL(DoubleTensor, double);
			break;
		}
	}

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ipLBP4R::init_lut_RI()
{
	// all 0's
	m_lut_RI[0] = 1;
	// 3 0's + 1 1's
	m_lut_RI[1] = 2;
	m_lut_RI[2] = 2;
	m_lut_RI[4] = 2;
	m_lut_RI[8] = 2;
	// 2 0's + 2 1's
	m_lut_RI[3] = 3;
	m_lut_RI[5] = 3;
	m_lut_RI[6] = 3;
	m_lut_RI[9] = 3;
	m_lut_RI[10] = 3;
	m_lut_RI[12] = 3;
	// 1 0's + 3 1's
	m_lut_RI[7] = 4;
	m_lut_RI[11] = 4;
	m_lut_RI[13] = 4;
	m_lut_RI[14] = 4;
	// all 1's
	m_lut_RI[15] = 5;

}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ipLBP4R::init_lut_U2()
{
	// A) all non uniform patterns have a label of 0.
	// already initialized to 0

	// B) LBP pattern with 0 bit to 1
	m_lut_U2[0] = 1;

	// C) LBP patterns with 1 bit to 1
	m_lut_U2[8] = 2;
	m_lut_U2[4] = 3;
	m_lut_U2[2] = 4;
	m_lut_U2[1] = 5;

	// D) LBP patterns with 2 bits to 1
	m_lut_U2[8+4] = 6;
	m_lut_U2[4+2] = 7;
	m_lut_U2[2+1] = 8;
	m_lut_U2[1+8] = 9;

	// E) LBP patterns with 3 bits to 1
	m_lut_U2[8+4+2] = 10;
	m_lut_U2[4+2+1] = 11;
	m_lut_U2[2+1+8] = 12;
	m_lut_U2[1+8+4] = 13;

	// F) LBP patterns with 4 bits to 1
	m_lut_U2[8+4+2+1] = 14;
}

///////////////////////////////////////////////////////////////////////////////////////////////////

void ipLBP4R::init_lut_U2RI()
{
	// A) all non uniform patterns have a label of 0.
	// already initialized to 0

	// All bits are 0
	m_lut_U2RI[0] = 1;

	// only one bit is 1 rest are 0's
	m_lut_U2RI[1] = 2;
	m_lut_U2RI[2] = 2;
	m_lut_U2RI[4] = 2;
	m_lut_U2RI[8] = 2;

	// only  two adjacent bits are 1 rest are 0's
	m_lut_U2RI[3] = 3;
	m_lut_U2RI[6] = 3;
	m_lut_U2RI[12] = 3;

	// only three adjacent bits are 1 rest are 0's
	m_lut_U2RI[7] = 4;
	m_lut_U2RI[14] = 4;

	// four adjacent bits are 1
	m_lut_U2RI[15] = 5;

}
///////////////////////////////////////////////////////////////////////////////////////
bool ipLBP4R::saveFile(File& file) const
{
	int idCore = getID();
	if (file.taggedWrite(&idCore, 1, "CoreID") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <CoreID> field!\n");
		return false;
	}


	//  m_modelSize[0]
	if (file.taggedWrite(&m_modelSize.size[1],1, "Width") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <Width> field!\n");
		return false;
	}

	if (file.taggedWrite(&m_modelSize.size[0], 1, "Height") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <height> field!\n");
		return false;
	}


	//m_P
	if (file.taggedWrite(&m_P, 1, "P") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <P> field!\n");
		return false;
	}

	//m_R
	if (file.taggedWrite(&m_R, 1, "Radius") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <Radius> field!\n");
		return false;
	}

	//m_x
	if (file.taggedWrite(&m_x,  1, "LocationX") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <LocationX> field!\n");
		return false;
	}

	//m_y
	if (file.taggedWrite(&m_y, 1, "LocationY") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <LocationY> field!\n");
		return false;
	}


	//m_toAverage
	if (file.taggedWrite(&m_toAverage, 1, "m_toAverage") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <m_toAverage> field!\n");
		return false;
	}


	//m_addAvgBit
	if (file.taggedWrite(&m_addAvgBit, 1, "m_addAvgBit") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <m_addAvgBit> field!\n");
		return false;
	}

	//m_toAverage
	if (file.taggedWrite(&m_uniform,1, "m_uniform") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <m_uniform> field!\n");
		return false;
	}
	//m_rot_invariant
	if (file.taggedWrite(&m_rot_invariant, 1, "m_rot_invariant") != 1)
	{
		Torch::message("ipLBP4R::save - failed to write <m_rot_invariant> field!\n");
		return false;
	}




	//print("ipHLBP4R()::saveFile()\n");
	//print("LBP P,R = %d %d, m_x, m_y = %d %d\n", m_P,m_R,m_x,m_y);
	// print("   X-Y = (%d-%d)\n", x, y);
	// print("   WxH = [%dx%d]\n", w, h);


	return true;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
bool ipLBP4R::loadFile(File& file)
{
	// int idCore =3; //should be written somewhere
//        if (file.taggedWrite(&idCore, sizeof(int), 1, "CoreID") != 1)
//        {
//            Torch::message("ipLBP4R::save - failed to write <CoreID> field!\n");
//            return false;
//        }

	int h,w;


	//  m_modelSize[0]
	if (file.taggedRead(&w, 1, "Width") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <Width> field!\n");
		return false;
	}

	if (file.taggedRead(&h, 1, "Height") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <height> field!\n");
		return false;
	}

	setModelSize(TensorSize(h, w));

	//m_P
	if (file.taggedRead(&m_P, 1, "P") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <P> field!\n");
		return false;
	}

	//m_R
	if (file.taggedRead(&m_R, 1, "Radius") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <Radius> field!\n");
		return false;
	}

	//m_x
	if (file.taggedRead(&m_x, 1, "LocationX") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <LocationX> field!\n");
		return false;
	}

	//m_y
	if (file.taggedRead(&m_y, 1, "LocationY") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <LocationY> field!\n");
		return false;
	}



	bool mavg, maddavg, muni, mroti;
	//m_toAverage
	if (file.taggedRead(&mavg, 1, "m_toAverage") != 1)
	{
		Torch::message("ipLBP4R::load - failed toRead <m_toAverage> field!\n");
		return false;
	}


	//m_addAvgBit
	if (file.taggedRead(&maddavg, 1, "m_addAvgBit") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <m_addAvgBit> field!\n");
		return false;
	}

	//m_toAverage
	if (file.taggedRead(&muni, 1, "m_uniform") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <m_uniform> field!\n");
		return false;
	}
	//m_rot_invariant
	if (file.taggedRead(&mroti, 1, "m_rot_invariant") != 1)
	{
		Torch::message("ipLBP4R::load - failed to Read <m_rot_invariant> field!\n");
		return false;
	}


	setBOption("ToAverage", mavg);
	setBOption("AddAvgBit", maddavg);
	setBOption("Uniform", muni);
	setBOption("RotInvariant", mroti);

	//print("ipHLBP4R()::loadFile()\n");
	// print("LBP P,R = %d %d, m_x, m_y = %d %d\n", m_P,m_R,m_x,m_y);
	// print("   X-Y = (%d-%d)\n", x, y);
	// print("   WxH = [%dx%d]\n", w, h);


	return true;
}

}
