#include "spCoreManager.h"

namespace Torch
{

//////////////////////////////////////////////////////////////////////////
// Constructor

    spCoreManager::spCoreManager()

    {
    }


    spCore* spCoreManager::getCore(int id_)
    {
        switch (id_)
        {
        case 1 :
            return new ipHaarLienhart();
            break;
        case 2 :
            return new ipHaarLienhart();//19,19);
            break;
        case 3 :
            return new ipLBP4R();//19,19);
            break;

        case 4 :
            return new ipLBP8R();//19,19);
            break;
        default :
            {
                Torch::print("spCoreManager::getCore()\n");
                return NULL;
            }

        }
    }
//////////////////////////////////////////////////////////////////////////
// Destructor

    spCoreManager::~spCoreManager()
    {
    }

//////////////////////////////////////////////////////////////////////////

}
