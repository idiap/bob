/**
 * @date Tue Apr 12 21:33:32 2011 +0200
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * @brief This file implements the inverse normal cumulative distribution function,
 * as described by Peter Acklam:
 *   http://home.online.no/~pjacklam/notes/invnorm/
 *
 * Copyright (C) Idiap Research Institute, Martigny, Switzerland
 */

#ifndef BOB_MATH_NORMINV_H
#define BOB_MATH_NORMINV_H

namespace bob { namespace math {
/**
 * @ingroup MATH
 * @{
 *
 */

    /**
     * @brief Function which implements the inverse normal cumulative 
     *   distribution function, as described by Peter Acklam:
     *     http://home.online.no/~pjacklam/notes/invnorm/
     *   Please note that the boost library only provide such a feature
     *   since version 1.46.
     * @param p The argument (probability) of the inverse function. p should
     *   be in the range ]0.,1.[ strictly. The implementation even enforce a
     *   slightly narrower range.
     * @param mu The mean of the normal distribution
     * @param sigma The standard deviation of the normal distribution
     */
    double norminv(const double p, const double mu, const double sigma);

    /**
     * @brief Function which implements the inverse normal cumulative 
     *   distribution function (whose mean is 0 and standard deviation is 1), 
     *   as described by Peter Acklam:
     *     http://home.online.no/~pjacklam/notes/invnorm/
     *   Please note that the boost library only provide such a feature
     *   since version 1.46.
     * @param p The argument (probability) of the inverse function. p should
     *   be in the range ]0.,1.[ strictly. The implementation even enforce a
     *   slightly narrower range.
     * @warning The normal distribution is assumed to be zero mean and unit 
     *   variance.
     */
    double normsinv(const double p);

/**
 * @}
 */
}}

#endif /* BOB_MATH_NORMINV_H */
