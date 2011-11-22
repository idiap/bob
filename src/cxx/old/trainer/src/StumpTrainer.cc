/**
 * @file cxx/old/trainer/src/StumpTrainer.cc
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
#include "trainer/StumpTrainer.h"

namespace Torch
{
    StumpTrainer::StumpTrainer(StumpMachine *stump_machine_, int n_features_, spCore **features_)
            : WeakLearner(stump_machine_, n_features_, features_)
    {
        m_stump_machine = stump_machine_;

        n_bins = 100;

        histogram = new float* [2];
        histogram[0] = new float [n_bins]; //for -ve patterns
        histogram[1] = new float [n_bins];

        cumulative_histogram = new float* [2];
        cumulative_histogram[0] = new float [n_bins];
        cumulative_histogram[1] = new float [n_bins];

        features_values = NULL;
    }
//////////////////////////////////////////////////////////////////////////////////////////
    bool StumpTrainer::train()
    {
        verbose  = getBOption("verbose");
        if (verbose)
            print(" \n StumpTrainer::train()\n");


        if (m_shuffledindex_dataset == NULL)
        {
            Torch::error("StumpTrainer::train() no shuffle index provided.");

            return false;
        }

        if (m_weights_dataset == NULL)
        {
            Torch::error("StumpTrainer::train() no weights provided.");

            return false;
        }

        if (m_stump_machine == NULL)
        {
            Torch::error("StumpTrainer::train() no stump machine provided.");

            return false;
        }

        // test the number of features
        if (verbose)
            print("    + Number of features = %d\n", m_n_features);
        if (m_n_features <= 0)
        {
            Torch::error("StumpTrainer::train() no features available.");

            return false;
        }

        // test the number of examples
        int n_examples = m_dataset->getNoExamples();
        if (verbose)
            print("    + Number of examples = %d\n", n_examples);
        if (n_examples <= 0)
        {
            Torch::error("StumpTrainer::train() no examples available.");

            return false;
        }

        // test if the dataset has targets
        if (m_dataset->hasTargets() != true)
        {
            Torch::error("StumpTrainer::train() no targets in the dataset.");

            return false;
        }

        // test target type, size and value
        for (int i = 0; i < n_examples ; i++)
        {

            Tensor *tensor = m_dataset->getTarget(i);

            // test the type
            if (tensor->getDatatype() != Tensor::Short)
            {
                Torch::error("StumpTrainer::train() targets should be ShortTensor.");

                return false;
            }

            // test the size
            ShortTensor *target = (ShortTensor *) tensor;

            if (target->nDimension() != 1)
            {
                Torch::error("StumpTrainer::train() target tensor should be 1 dimension.");

                return false;
            }

            if (target->size(0) != 1)
            {
                Torch::error("StumpTrainer::train() target tensor should be of size 1.");

                return false;
            }

            // test the value
            short target_value = (*target)(0);

            if (target_value != 0 && target_value != 1)
            {
                Torch::error("StumpTrainer::train() target values should be -1 or 1.");

                return false;
            }
        }

        //
        if (features_values != NULL) delete []features_values;
        features_values = new float [n_examples];

        float min_error = FLT_MAX;
        int bestFeature = -1;
        float bestThreshold = 0.0;
        int bestDirection = 0;
        float max__, min__;
        float *hp = new float [n_bins]; //for -ve patterns
        float *hn = new float [n_bins];
        int n_positive = 0;
        int n_negative = 0;
        max__ = 0;
        min__=0;


        for (int f = 0; f < m_n_features ; f++)
        {
            // compute the distribution of the current feature value across the dataset

            for (int i = 0; i < n_bins ; i++)
            {
                histogram[0][i] = 0.0;
                histogram[1][i] = 0.0;
                cumulative_histogram[0][i] = 0.0;
                cumulative_histogram[1][i] = 0.0;
            }

            // do a first pass to determine the min and max
            float min_ = FLT_MAX;
            float max_ = -FLT_MAX;


            n_positive = 0;
            n_negative = 0;

            //print("computing the feature %d for all examples ...\n", f);


            for (int i = 0; i < n_examples ; i++)
            {
                int index = m_shuffledindex_dataset[i];
                Tensor *example = m_dataset->getExample(index);



                // here we should test first the type and size of the returned tensor
                m_features[f]->process(*example);
                DoubleTensor *feature_value = (DoubleTensor *) &m_features[f]->getOutput(0);

                // store the features for more efficiency
                float z = (*feature_value)(0);
                features_values[i] = z;

                //
                if (z < min_) min_ = z;
                else if (z > max_) max_ = z;
            }

            double bin_size = (max_ - min_) / (double) n_bins;

            //print("building the histogram ...\n", f);

            //
            for (int i = 0; i < n_examples ; i++)
            {
                int index = m_shuffledindex_dataset[i];
                ShortTensor *target = (ShortTensor *) m_dataset->getTarget(index);
                short target_value = (*target)(0);

                float z = features_values[i] - min_;

                if (target_value == 1)
                {
                    // positive class

                    // binning
                    int bin = (int) floor(z / bin_size);
                    if (bin < 0) bin = 0;
                    if (bin >= n_bins) bin = n_bins-1;

                    // increment
                    histogram[1][bin] += 1;

                    n_positive++;
                }
                else if (target_value == 0)
                {
                    // negative class

                    // binning
                    int bin = (int) floor(z / bin_size);
                    if (bin < 0) bin = 0;
                    if (bin >= n_bins) bin = n_bins-1;

                    // increment
                    histogram[0][bin] += 1;

                    n_negative++;
                }
            }

            //
            float error;
            float error_r; // positive examples are above the threshold
            float error_l; // negative examples are below the threshold
            int direction;

            float norm_positive = 1.0 / (float) n_positive;
            float norm_negative = 1.0 / (float) n_negative;

            float cumul_positive = 0.0;
            float cumul_negative = 0.0;

            for (int i = 0; i < n_bins ; i++)
            {
                cumul_negative += histogram[0][i] * norm_negative;
                cumul_positive += histogram[1][i] * norm_positive;

                cumulative_histogram[0][i] = cumul_negative;
                cumulative_histogram[1][i] = cumul_positive;

                error_r = 0.0;
                error_l = 0.0;
                error = 0.0;
                direction = 0;

                error_r = ((1.0 - cumul_negative) * n_negative) + (cumul_positive * n_positive);
                error_l = (cumul_negative * n_negative) + ((1.0 - cumul_positive) * n_positive);

                if (error_r <= error_l)
                {
                    error = error_r;
                    direction = 1;
                }
                else
                {
                    error = error_l;
                    direction = -1;
                }

                if (error < min_error)
                {
                    min_error = error;
                    bestFeature = f;
                    bestThreshold = (i+1) * bin_size + min_;
                    bestDirection = direction;
                    max__ = max_;
                    min__ = min_;
                    for (int hm = 0;hm<n_bins;hm++)
                    {
                        hp[hm] = histogram[1][hm];
                        hn[hm] = histogram[0][hm];
                    }
                }
            }
        }

        if (verbose)
        {
            print("Max and Min of Stump Machine : %f, %f\n",max__,min__);

            print("   Selected feature (%d, %g, %d) E = %g.\n", bestFeature, bestThreshold, bestDirection, min_error);
            print(" N neg %d, N pos %d, N Examples %d\n",n_negative,n_positive, n_examples);
        }


        m_weak_classifier->setCore(m_features[bestFeature]);
        //m_stump_machine->setCore(m_features[bestFeature]);
        m_stump_machine->setParams(bestDirection, bestThreshold);

        if (verbose)
            print("\n");

        return true;
    }

////////////////////////////////////////////////////////////////////////////////////////////
    StumpTrainer::~StumpTrainer()
    {
        if (features_values != NULL) delete [] features_values;
        delete [] histogram[0];
        delete [] histogram[1];
        delete [] histogram;

        delete [] cumulative_histogram[0];
        delete [] cumulative_histogram[1];
        delete [] cumulative_histogram;
    }
//////////////////////////////////////////////////////////////////////////////////////////////
}
