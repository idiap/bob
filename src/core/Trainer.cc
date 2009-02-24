#include "Trainer.h"

namespace Torch {

////////////////////////////////////////////////////////////////////////
/// Constructor

Trainer::Trainer(): Object()
{
	m_dataset = NULL;
	m_machine = NULL;
}


bool Trainer::train()
{
 	return 0;
}
////////////////////////////////////////////////////////////////////////

bool Trainer::setData(DataSet* m_dataset_)
{
   	if(m_dataset_ == NULL) return false;
	m_dataset = m_dataset_;
	return true;
}
///////////////////////////////////////
bool Trainer::setMachine(Machine* m_machine_)
{
   	if(m_machine_ == NULL) return false;
	m_machine = m_machine_;
	return true;

}

/// Destructor
Trainer::~Trainer()
{
}

////////////////////////////////////////////////////////////////////////

}

