/**
 * @date Tue Jul 19 15:33:20 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MACHINE_ZTNORM_H
#define BOB_MACHINE_ZTNORM_H

#include <blitz/array.h>

namespace bob { namespace machine {
/**
 * @ingroup MACHINE
 * @{
 */

/**
 * Normalise raw scores with ZT-Norm
 *
 * @exception std::runtime_error matrix sizes are not consistent
 * 
 * @param rawscores_probes_vs_models
 * @param rawscores_zprobes_vs_models
 * @param rawscores_probes_vs_tmodels
 * @param rawscores_zprobes_vs_tmodels
 * @param mask_zprobes_vs_tmodels_istruetrial
 * @param[out] normalizedscores normalized scores
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
 * @exception std::runtime_error matrix sizes are not consistent
 *
 * @param rawscores_probes_vs_models
 * @param rawscores_zprobes_vs_models
 * @param rawscores_probes_vs_tmodels
 * @param rawscores_zprobes_vs_tmodels
 * @param[out] normalizedscores normalized scores
 * @warning The destination score array should have the correct size
 *          (Same size as rawscores_probes_vs_models)
 */
void ztNorm(const blitz::Array<double,2>& rawscores_probes_vs_models,
            const blitz::Array<double,2>& rawscores_zprobes_vs_models,
            const blitz::Array<double,2>& rawscores_probes_vs_tmodels,
            const blitz::Array<double,2>& rawscores_zprobes_vs_tmodels,
            blitz::Array<double,2>& normalizedscores);

/**
 * Normalise raw scores with T-Norm.
 *
 * @exception std::runtime_error matrix sizes are not consistent
 *
 * @param rawscores_probes_vs_models
 * @param rawscores_probes_vs_tmodels
 * @param[out] normalizedscores normalized scores
 * @warning The destination score array should have the correct size
 *          (Same size as rawscores_probes_vs_models)
 */
void tNorm(const blitz::Array<double,2>& rawscores_probes_vs_models,
           const blitz::Array<double,2>& rawscores_probes_vs_tmodels,
           blitz::Array<double,2>& normalizedscores);

/**
 * Normalise raw scores with Z-Norm.
 *
 * @exception std::runtime_error matrix sizes are not consistent
 *
 * @param rawscores_probes_vs_models
 * @param rawscores_zprobes_vs_models
 * @param[out] normalizedscores normalized scores
 * @warning The destination score array should have the correct size
 *          (Same size as rawscores_probes_vs_models)
 */
void zNorm(const blitz::Array<double,2>& rawscores_probes_vs_models,
           const blitz::Array<double,2>& rawscores_zprobes_vs_models,
           blitz::Array<double,2>& normalizedscores);

/**
 * @}
 */
}}

#endif /* BOB_MACHINE_ZTNORM_H */
