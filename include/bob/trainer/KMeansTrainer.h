/**
 * @file bob/trainer/KMeansTrainer.h
 * @date Tue May 10 11:35:58 2011 +0200
 * @author Francois Moulin <Francois.Moulin@idiap.ch>
 * @author Laurent El Shafey <Laurent.El-Shafey@idiap.ch>
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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
#ifndef BOB_TRAINER_KMEANSTRAINER_H
#define BOB_TRAINER_KMEANSTRAINER_H

#include "bob/machine/KMeansMachine.h"
#include "bob/trainer/EMTrainer.h"

namespace bob {
namespace trainer {

/**
 * Trains a KMeans machine.
 * @brief This class implements the expectation-maximisation algorithm for a k-means machine.
 * @details See Section 9.1 of Bishop, "Pattern recognition and machine learning", 2006
 *          It uses a random initialisation of the means followed by the expectation-maximization algorithm
 */
class KMeansTrainer: public EMTrainer<bob::machine::KMeansMachine, blitz::Array<double,2> >
{
  public: 
    /**
     * Constructor
     */
    KMeansTrainer(double convergence_threshold=0.001, 
      size_t max_iterations=10, bool compute_likelihood=true, 
      bool check_no_duplicate=false);
    
    /**
     * virtualize destructor
     */
    virtual ~KMeansTrainer() {}

    /**
     * Copy constructor
     */
    KMeansTrainer(const KMeansTrainer& other);

    /**
     * Assigns from a different machine
     */
    KMeansTrainer& operator=(const KMeansTrainer& other);

    /**
     * Equal to
     */
    bool operator==(const KMeansTrainer& b) const;

    /**
     * Not equal to
     */
    bool operator!=(const KMeansTrainer& b) const;
 
    /**
     * Initialise the means randomly. 
     * Data is split into as many chunks as there are means, 
     * then each mean is set to a random example within each chunk.
     */
    virtual void initialization(bob::machine::KMeansMachine& kMeansMachine,
      const blitz::Array<double,2>& sampler);
    
    /**
     * Accumulate across the dataset:
     * - zeroeth and first order statistics
     * - average distance from the closest mean 
     * Implements EMTrainer::eStep(double &)
     */
    virtual void eStep(bob::machine::KMeansMachine& kmeans,
      const blitz::Array<double,2>& data);
    
    /**
     * Updates the mean based on the statistics from the E-step.
     */
    virtual void mStep(bob::machine::KMeansMachine& kmeans, 
      const blitz::Array<double,2>&);
    
    /**
     * This functions returns the average min distance (average distance to the
     * closest mean)
     */
    virtual double computeLikelihood(bob::machine::KMeansMachine& kmeans);

    /**
     * Function called at the end of the training 
     */
    virtual void finalization(bob::machine::KMeansMachine& kMeansMachine, const blitz::Array<double,2>& sampler);

    /**
     * Reset the statistics accumulators
     * to the correct size and a value of zero.
     */
    bool resetAccumulators(bob::machine::KMeansMachine& kMeansMachine);

    /**
     * Set the seed used to genrated pseudo-random numbers
     */
    void setSeed(int seed);

    /**
     * Get the seed
     */
    int getSeed() const { return m_seed; }

    /**
     * Tell whether duplicate means are checked or not during the initialization
     */
    void setCheckNoDuplicate(bool v) { m_check_no_duplicate = v; }

    /**
     * Tell whether duplicate means are checked or not during the initialization
     */
    bool getCheckNoDuplicate() const { return m_check_no_duplicate; }
  
    /**
     * Returns the internal statistics. Useful to parallelize the E-step
     */
    const blitz::Array<double,1>& getZeroethOrderStats() const { return m_zeroethOrderStats; }
    const blitz::Array<double,2>& getFirstOrderStats() const { return m_firstOrderStats; }
    double getAverageMinDistance() const { return m_average_min_distance; }
    /**
     * Sets the internal statistics. Useful to parallelize the E-step
     */
    void setZeroethOrderStats(const blitz::Array<double,1>& zeroethOrderStats); 
    void setFirstOrderStats(const blitz::Array<double,2>& firstOrderStats);
    void setAverageMinDistance(const double value) { m_average_min_distance = value; }

 
  protected:

    /**
     * Check that there is no duplicated means during the random initialization
     */
    bool m_check_no_duplicate;

    /**
     * Seed used to generate pseudo-random numbers
     */
    int m_seed;
   
    /**
     * Average min distance
     */
    double m_average_min_distance;

    /**
     * Zeroeth order statistics accumulator.
     * The k'th value in m_zeroethOrderStats is the denominator of 
     * equation 9.4, Bishop, "Pattern recognition and machine learning", 2006
     */
    blitz::Array<double,1> m_zeroethOrderStats;

    /** 
     * First order statistics accumulator.
     * The k'th row of m_firstOrderStats is the numerator of 
     * equation 9.4, Bishop, "Pattern recognition and machine learning", 2006
     */
    blitz::Array<double,2> m_firstOrderStats;
};

}
}

#endif // BOB_TRAINER_KMEANSTRAINER_H
