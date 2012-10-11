/**
 * @file bob/trainer/TwoDPCATrainer.h
 * @date Wed May 18 21:51:16 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2012 Idiap Research Institute, Martigny, Switzerland
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
#ifndef BOB_TRAINER_TWODPCA_TRAINER_H
#define BOB_TRAINER_TWODPCA_TRAINER_H

#include "Trainer.h"
#include "bob/core/logging.h"
#include "bob/machine/TwoDPCAMachine.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace bob 
{
  namespace trainer 
  {
  
    class TwoDPCATrainer: public Trainer<bob::machine::TwoDPCAMachine, blitz::Array<double,2> >
    {
      public:
        TwoDPCATrainer() {}
        virtual ~TwoDPCATrainer() {}
  
        virtual void train(bob::machine::TwoDPCAMachine& machine, const blitz::Array<double,3>& data); 

      protected:

    };

  }
}

#endif /* BOB_TRAINER_TWODPCA_TRAINER_H */
