/**
 * @file cxx/trainer/trainer/TwoDPCATrainer.h
 * @date Wed May 18 21:51:16 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
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
#ifndef BOB5SPRO_TRAINER_TWODPCA_TRAINER_H
#define BOB5SPRO_TRAINER_TWODPCA_TRAINER_H

#include "Trainer.h"
#include "core/logging.h"
#include "io/Arrayset.h"
#include "machine/TwoDPCAMachine.h"
#include <algorithm>
#include <utility>
#include <vector>

namespace bob 
{
  namespace trainer 
  {
  
    class TwoDPCATrainer : virtual public Trainer<bob::machine::TwoDPCAMachine, bob::io::Arrayset>
    {
      public:
        TwoDPCATrainer() {}
        virtual ~TwoDPCATrainer() {}
  
        void train(bob::machine::TwoDPCAMachine& machine, const bob::io::Arrayset& data); 

      protected:

    };

  }
}

#endif /* BOB5SPRO_TRAINER_TWODPCA_TRAINER_H */
