/**
 * @file src/cxx/machine/machine/ZTNorm.h
 * @date Tue Jul 19 15:33:20 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
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

#ifndef BOB_MACHINE_ZTNORM_H
#define BOB_MACHINE_ZTNORM_H

#include <blitz/array.h>

namespace bob { namespace machine {

  /**
   * Normalise raw scores with ZT-Norm
   *
   * @exception bob::core::UnexpectedShapeError matrix sizes are not consistent
   * 
   * @param rawscores_probes_vs_models
   * @param rawscores_zprobes_vs_models
   * @param rawscores_probes_vs_tmodels
   * @param rawscores_zprobes_vs_tmodels
   * @param mask_zprobes_tmodels_istruetrial
   * @param[out] scores normalized scores
   * @warning The destination score array should have the correct size
   *          (Same size as rawscores_probes_vs_models)
   */
  void ztNorm(const blitz::Array<double, 2>& rawscores_probes_vs_models,
              const blitz::Array<double, 2>& rawscores_zprobes_vs_models,
              const blitz::Array<double, 2>& rawscores_probes_vs_tmodels,
              const blitz::Array<double, 2>& rawscores_zprobes_vs_tmodels,
              const blitz::Array<bool,   2>& mask_zprobes_vs_tmodels_istruetrial,
              blitz::Array<double, 2>& normalizedscores);
  
  /**
   * Normalise raw scores with ZT-Norm.
   * Assume that znorm and tnorm have no common subject id.
   *
   * @exception bob::core::UnexpectedShapeError matrix sizes are not consistent
   *
   * @param rawscores_probes_vs_models
   * @param rawscores_zprobes_vs_models
   * @param rawscores_probes_vs_tmodels
   * @param rawscores_zprobes_vs_tmodels
   * @param[out] scores normalized scores
   * @warning The destination score array should have the correct size
   *          (Same size as rawscores_probes_vs_models)
   */
  void ztNorm(const blitz::Array<double, 2>& rawscores_probes_vs_models,
              const blitz::Array<double, 2>& rawscores_zprobes_vs_models,
              const blitz::Array<double, 2>& rawscores_probes_vs_tmodels,
              const blitz::Array<double, 2>& rawscores_zprobes_vs_tmodels,
              blitz::Array<double, 2>& normalizedscores);
}
}

#endif /* BOB_MACHINE_ZTNORM_H */
