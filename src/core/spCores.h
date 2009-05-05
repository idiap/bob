#ifndef _TORCH5SPRO_SP_CORES_H_
#define _TORCH5SPRO_SP_CORES_H_

//////////////////////////////////////////////////////////////////////////////////
// Include all the spCores, to make sure they are all registered to the spCoreManager
// NB: the test programs should include this header instead of each spCore's header

#define IP_HAAR_LIENHART_ID 2
#define IP_LBP_4R_ID 3
#define IP_LBP_8R_ID 4

#include "ipHaarLienhart.h"
#include "ipLBP4R.h"
#include "ipLBP8R.h"

//////////////////////////////////////////////////////////////////////////////////

#endif
