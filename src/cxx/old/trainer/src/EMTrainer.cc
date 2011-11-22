/**
 * @file cxx/old/trainer/src/EMTrainer.cc
 * @date Wed Apr 6 14:49:40 2011 +0200
 * @author Andre Anjos <andre.anjos@idiap.ch>
 *
 * Copyright (C) 2011 Idiap Reasearch Institute, Martigny, Switzerland
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, version 3 of the License.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
#include "trainer/EMTrainer.h"
#include "machine/ProbabilityDistribution.h"

namespace Torch
{
    EMTrainer::EMTrainer() : Trainer()
    {
	addIOption("max iter", 10, "maximum number of iterations");
	addFOption("end accuracy", 0.0001, "end accuracy");
    }

    bool EMTrainer::train()
    {
        print("EMTrainer::train() ...\n");

	//
        if (m_machine == NULL)
        {
            warning("EMTrainer::train() no machine.");

            return false;
        }

	/*
	int m_id_ = 1000 * (int) (m_machine->getID() / 1000); 
	if(m_id_ != PROBABILITY_DISTRIBUTION_MACHINE_ID)
        {
            warning("EMTrainer::train() the machine is not a ProbabilityDistribution.");

            return false;
        }
	*/

	//
        if (m_dataset == NULL)
        {
            warning("EMTrainer::train() no dataset.");

            return false;
        }

        //
        long m_n_examples = m_dataset->getNoExamples();
        if (m_n_examples < 2)
        {
            warning("EMTrainer::train() not enough examples in the  dataset.");

            return false;
        }

        // testing at least if the dataset has targets
        if (m_dataset->hasTargets() == true)
        {
            warning("EMTrainer::train() dataset has targets but will be ignore.");
        }

        //
	int max_iter = getIOption("max iter");
	float end_accuracy = getFOption("end accuracy");

	//
	ProbabilityDistribution *pd_machine = (ProbabilityDistribution *) m_machine;

	pd_machine->prepare();
	
	//
	double previous_nll;
	double nll = 0.0;

       	for (long i=0 ; i<m_n_examples ; i++)
       	{
  		DoubleTensor *example = (DoubleTensor *) m_dataset->getExample(i);
       	        
		// forward
		pd_machine->forward(example);

		nll += pd_machine->getOutput().get(0);
       	}
	nll /= (double) m_n_examples;

	print("nll @ 0 = %g\n", nll);
	
	int iter = 1;

	while(1)
	{
		pd_machine->EMinit();

		// E-step : computes and accumulates posterior probabilities
    for (long i=0 ; i<m_n_examples ; i++)
    {
      const DoubleTensor* example = static_cast<const DoubleTensor*>(m_dataset->getExample(i));
      pd_machine->EMaccPosteriors(*example, 0 /* log(1) */);
    }

		
		// M-step : Updates weights, means and variances 
		pd_machine->EMupdate();

		previous_nll = nll;
		nll = 0.0;

        	for (long i=0 ; i<m_n_examples ; i++)
        	{
   			Tensor *example = m_dataset->getExample(i);
        	        
			// forward
			pd_machine->forward(*example);

			nll += pd_machine->getOutput().get(0);
        	}
		nll /= (double) m_n_examples;

		print(".");

		print("nll @ %d = %g\n", iter, nll);

		if(fabs(previous_nll - nll) < end_accuracy)
 		{
			print("End of accuracy\n");
			break;
		}

		//
		iter++;
		if( (iter >= max_iter) && (max_iter > 0) )
		{
			print("\nMaximum number of iterations reached\n");
			break;
		}
	}

        return true;
    }

    EMTrainer::~EMTrainer()
    {
    }

}
