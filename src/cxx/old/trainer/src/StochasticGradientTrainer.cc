/**
 * @file cxx/old/trainer/src/StochasticGradientTrainer.cc
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
#include "trainer/StochasticGradientTrainer.h"
#include "machine/GradientMachine.h"
#include "core/Random.h"

namespace Torch
{
  StochasticGradientTrainer::StochasticGradientTrainer() : Trainer()
  {
  	addIOption("max iter", 10, "maximum number of iterations");
  	addFOption("end accuracy", 0.0001, "end accuracy");
	  addIOption("early stopping", 0, "maximum number of iterations for early stopping");
  	addFOption("learning rate", 0.01, "learning rate");
	  addFOption("learning rate decay", 0.0, "learning rate decay");
  	addFOption("weight decay", 0.0, "weight decay");
	  addFOption("momentum", 0.0, "Inertia momentum");

    m_shuffledindex = NULL;
  }


  bool StochasticGradientTrainer::train()
  {
    print("StochasticGradientTrainer::train() ...\n");

	  //
    if (m_machine == NULL)
    {
      warning("StochasticGradientTrainer::train() no machine.");

      return false;
    }

	/*
	int m_id_ = 1000 * (int) (m_machine->getID() / 1000); 
	if(m_id_ != GRADIENT_MACHINE_ID)
        {
            warning("GradientTrainer::train() the machine is not a GradientMachine.");

            return false;
        }
	*/

	//
    if (m_criterion == NULL)
    {
      warning("StochasticGradientTrainer::train() no criterion.");

      return false;
    }

	//
    if (m_dataset == NULL)
    {
      warning("StochasticGradientTrainer::train() no dataset.");

      return false;
    }

    //
    long m_n_examples = m_dataset->getNoExamples();
    if (m_n_examples < 2)
    {
      warning("StochasticGradientTrainer::train() not enough examples in the  dataset.");

      return false;
    }

    // testing at least if the dataset has targets
    if (m_dataset->hasTargets() != true)
    {
      warning("StochasticGradientTrainer::train() no targets in the dataset.");

      return false;
    }

    //
   	if (m_shuffledindex == NULL) delete []m_shuffledindex;
      m_shuffledindex = new long [m_n_examples];
    for (int i=0 ; i<m_n_examples ; i++) m_shuffledindex[i] = i;
    core::random::uniform_uint<unsigned int> rng;
    std::random_shuffle(m_shuffledindex, m_shuffledindex+m_n_examples, rng);
	
	  //
  	int max_iter = getIOption("max iter");
	  float end_accuracy = getFOption("end accuracy");
  	//int early_stopping = getIOption("early stopping");
	  float learning_rate = getFOption("learning rate");
  	float learning_rate_decay = getFOption("learning rate decay");
	  float weight_decay = getFOption("weight decay");
  	float momentum = getFOption("momentum");

	  //
  	GradientMachine *g_machine = (GradientMachine *) m_machine;

	  g_machine->setFOption("weight decay", weight_decay);
  	g_machine->setFOption("momentum", momentum);

	  g_machine->prepare();

  	g_machine->shuffle();

	  //int n_parameters_ = g_machine->m_parameters->getI("n_parameters");
  	//double *parameters_ = g_machine->m_parameters->getDarray("parameters");
	  //double *der_parameters_ = g_machine->m_parameters->getDarray("der_parameters");
																		                
  	//
	  double current_learning_rate = learning_rate;
  	double previous_error = 100000.0;
	  int iter = 0;

  	while(1)
	  {
		  double error = 0.0;

      for (long i=0 ; i<m_n_examples ; i++)
      {
		   	//
			  g_machine->Ginit();

   			Tensor *example = m_dataset->getExample(m_shuffledindex[i]);
  			Tensor *target = m_dataset->getTarget(m_shuffledindex[i]);
        	        
  			// forward
	  		g_machine->forward(*example);

		  	// criterion
			  m_criterion->forward(&g_machine->getOutput(), target);
			
  			// backward
	  		g_machine->backward(*example, m_criterion->m_beta);

		  	error += m_criterion->m_error->get(0);
		
			  //
  			g_machine->Gupdate(current_learning_rate);
      }

  		// !!!! consider saving the model at each iteration !!!!
	  	// ..

		  print(".");

   		//
  		error /= (double) m_n_examples;
	  	print("error @ %d = %g (%g)\n", iter, error, current_learning_rate);

		  //
  		if(fabs(previous_error - error) < end_accuracy)
 	  	{
		  	print("End of accuracy\n");
			  break;
  		}
	  	previous_error = error;

		  // !!!! implements early stopping (validation set required) !!!!
  		// ..

	  	//
		  iter++;
  		if( (iter >= max_iter) && (max_iter > 0) )
	  	{
		  	print("\nMaximum number of iterations reached\n");
			  break;
  		}

	  	//
		  current_learning_rate = learning_rate/(1.+((float)(iter))*learning_rate_decay);
  	}

    return true;
  }

  bool StochasticGradientTrainer::setCriterion(Criterion *m_criterion_)
  {
   	if(m_criterion_ == NULL) return false;
	  m_criterion = m_criterion_;
    return true;
  }

  StochasticGradientTrainer::~StochasticGradientTrainer()
  {
  	if (m_shuffledindex != NULL) delete []m_shuffledindex;
  }

}
