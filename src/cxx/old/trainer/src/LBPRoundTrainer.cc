/**
 * @file cxx/old/trainer/src/LBPRoundTrainer.cc
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
#include "trainer/LBPRoundTrainer.h"
#include "ip/ipLBP.h"

namespace Torch
{

//////////////////////////////////////////////////////////////////////
    LBPRoundTrainer::LBPRoundTrainer(IntLutMachine *lbp_machine_, int n_features_, spCore **features_)
            : WeakLearner(lbp_machine_, n_features_, features_)
    {
        m_lbp_machine = lbp_machine_;

        //Assuming the maximum number of bins are 512
        m_n_bins = 512;

        m_histogram = new float* [2];
        m_histogram[0] = new float [m_n_bins];
        m_histogram[1] = new float [m_n_bins];

        m_features_values = NULL;

        m_b_lutsize = 0;
        m_bestlut_ = NULL;
    }
/////////////////////////////////////////////////////////////////////////////////////////////
    bool LBPRoundTrainer::train()
    {
        verbose = getBOption("verbose");
        if (verbose)
        {
            print("\n");
            print("   LBPRoundTrainer::train()\n");
        }


        if (m_shuffledindex_dataset == NULL)
        {
            Torch::error("LBPRoundTrainer::train() no shuffle index provided.");

            return false;
        }

        if (m_weights_dataset == NULL)
        {
            Torch::error("LBPRoundTrainer::train() no weights provided.");

            return false;
        }

        if (m_lbp_machine == NULL)
        {
            Torch::error("LBPRoundTrainer::train() no LBP machine provided.");

            return false;
        }

        // test the number of features
        if (verbose)
            print("    + Number of features = %d\n", m_n_features);

        if (m_n_features <= 0)
        {
            Torch::error("LBPRoundTrainer::train() no features available.");

            return false;
        }

        // test the number of examples
        int n_examples = m_dataset->getNoExamples();
        if (verbose)
        print("    + Number of examples = %d\n", n_examples);
        if (n_examples <= 0)
        {
            Torch::error("LBPRoundTrainer::train() no examples available.");

            return false;
        }

        // test if the dataset has targets
        if (m_dataset->hasTargets() != true)
        {
            Torch::error("LBPRoundTrainer::train() no targets in the dataset.");

            return false;
        }

        // test target type, size and value
        for (int i = 0; i < n_examples ; i++)
        {
            Tensor *tensor = m_dataset->getTarget(i);

            // test the type
            if (tensor->getDatatype() != Tensor::Short)
            {
                Torch::error("LBPRoundTrainer::train() targets should be ShortTensor.");

                return false;
            }

            // test the size
            ShortTensor *target = (ShortTensor *) tensor;

            if (target->nDimension() != 1)
            {
                Torch::error("LBPRoundTrainer::train() target tensor should be 1 dimension.");

                return false;
            }

            if (target->size(0) != 1)
            {
                Torch::error("LBPRoundTrainer::train() target tensor should be of size 1.");

                return false;
            }

            // test the value
            short target_value = (*target)(0);

            if (target_value != 0 && target_value != 1)
            {
                Torch::error("LBPRoundTrainer::train() target values should be 0 or 1.");

                return false;
            }
        }

        delete [] m_features_values;
        m_features_values = new int [n_examples];

        double *lut_ = new double [m_n_bins];

        m_bestlut_ = new double [m_n_bins];

        double min_error = 999999999.0;

        int lutsize; //should be able to compute lut for different types of LBP.
        m_b_lutsize=-1;

        int bestFeature = -1;

        bool passthrough;
        int lbp_code;
        Tensor *example;
        for (int f = 0; f < m_n_features ; f++)
        {
            //
            passthrough = true;
            if (m_mask)
            {
                if (m_featuremask[f] != 1)
                    passthrough = false;
            }
            if (passthrough)
            {

                //should get here the size of LUT.......for that particular LBP
                lutsize = ((ipLBP*)m_features[f])->getMaxLabel();

                int n_positive = 0;
                int n_negative = 0;
                // compute the distribution of the current feature value across the dataset

                for (int i = 0; i < lutsize ; i++)
                {
                    m_histogram[0][i] = 0.0;
                    m_histogram[1][i] = 0.0;
                    lut_[i]=0.0;

                }


                //   print("building the m_histogram ...\n", f);

                //
                for (int i = 0; i < n_examples ; i++)
                {
                    int index = m_shuffledindex_dataset[i];
                    ShortTensor *target = (ShortTensor *) m_dataset->getTarget(index);
                    short target_value = (*target)(0);


                    example = m_dataset->getExample(index);
                    // 	Tprint(example);
                    m_features[f]->process(*example);
                    IntTensor *lbp_value = (IntTensor *) &m_features[f]->getOutput(0);
                    lbp_code = (*lbp_value)(0);

                    m_features_values[i] = lbp_code;
                    if (target_value == 1)
                    {
                        // positive class

                        // increment
                        m_histogram[1][lbp_code] += 1;

                        n_positive++;
                    }
                    else if (target_value == 0)
                    {
                        // negative class



                        // increment
                        m_histogram[0][lbp_code] += 1;

                        n_negative++;
                    }
                }


                //So now the historam is ready

                // If the examples are selected randomly based on weights then each of them will have weights =1
                // Otherwise we can implement the LUT creation from MultiBlock LBP paper (implements GentleBoost)

                //
                //build the LUT

                for (int i=0;i<lutsize;i++)
                {

                    if (m_histogram[1][i]>m_histogram[0][i])
                        lut_[i] = 1;
                    else
                        lut_[i]=-1;
                }

                //now compute the error
                double error_lut = 0;
                for (int i=0;i<n_examples;i++)
                {


                    int index = m_shuffledindex_dataset[i];
                    ShortTensor *target = (ShortTensor *) m_dataset->getTarget(index);
                    short target_value = (*target)(0);
                    double tar_val=0;
                    if (target_value==1)
                        tar_val = 1;
                    else
                        tar_val=-1;
                    //print("target value %f\n",target_value);
                    //weights are assumed to be 1 due to random sampling based on weights
                    // otherwise the sample weights has to be considered while calculating error
                    //  error_lut = error_lut + (tar_val - lut_[bin]) * (tar_val - lut_[bin]);
                    if (lut_[m_features_values[i]] != tar_val)
                        error_lut++;// = error_lut + m_weights_dataset[index];



                }

                // print("Error_lut : %f\n",error_lut);
                if (error_lut<min_error)
                {
                    bestFeature = f;
                    for (int i=0;i<m_n_bins;i++)
                    {
                        m_bestlut_[i] = lut_[i];

                    }
                    min_error = error_lut;
                    m_b_lutsize = lutsize;

                }


            }
        }

        if (m_b_lutsize <0)
        {
            print("There is no proper lut size Aborting training\n");
            return false;
        }
        m_weak_classifier->setCore(m_features[bestFeature]);
       // delete m_lbp_machine->

        m_lbp_machine->setParams(m_b_lutsize, m_bestlut_);

        if (verbose)
            print( "Selected feature (%d) Error = %g.\n",bestFeature,min_error);
        m_featureID = bestFeature;


        delete [] lut_;

        return true;
    }
//////////////////////////////////////////////////////////////////////////////////////
    LBPRoundTrainer::~LBPRoundTrainer()
    {
        delete [] m_bestlut_;
        delete [] m_features_values;


        delete[] m_histogram[0];
        delete[] m_histogram[1];
        delete[] m_histogram;
    }

}
