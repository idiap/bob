#include "trainer/BoostingRoundTrainer.h"

namespace Torch
{
    BoostingRoundTrainer::BoostingRoundTrainer()
    {
        addBOption("boosting_by_sampling",	false,	"use sampling based on weights");
        addIOption("number_of_rounds",	10,	"number of rounds to select the already selected features");
        m_n_examples = 0;
        m_n_classifiers = 0;
        m_n_classifiers_trained = 0;

        m_weights = NULL;
        m_weights_samples = NULL;
        m_label_samples = NULL;
        m_shuffledindex_dataset = NULL;
        m_repartition = NULL;
        m_labelledmeasure = NULL;
        m_featuremask =NULL;
        m_mask =false;
        m_nrounds =0;
    }
    //............................................................................
    bool BoostingRoundTrainer::train()
    {
        print("BoostingTrainer::train() ...\n");

        //
        bool useSampling = getBOption("boosting_by_sampling");
        m_nrounds = getIOption("number_of_rounds");

        print("number of Classifiers %d, nRounds %d\n",m_n_classifiers,m_nrounds);

        //Check if the number of rounds are greater than number of weakClassifiers
        if(m_n_classifiers> m_nrounds)
        {
                Torch::error("BoostingRoundTrainer::train() Classfiers > nRounds.");

                return false;
        }
        //get the number of features to initialize m_featuremask

    if(m_n_classifiers>0)
        m_features = m_weak_learners[0]->getmnFeatures();

        m_featuremask =new int[m_features];

        for(int i=0;i<m_features;i++)
            m_featuremask[i] = 0;

        m_n_examples = m_dataset->getNoExamples();

        // testing at least if the dataset has targets
        // let the StumpTrainer test the number of classes (> 1) and eventually deal with multiple classes
        if (m_dataset->hasTargets() != true)
        {
            warning("BoostingRoundTrainer::train() no targets in the dataset.");

            return false;
        }

        //
        print(" + Number of weak classifiers: %d\n", m_n_classifiers);
        print(" + Number of examples: %d\n", m_n_examples);

        m_weights_samples = new double [m_n_examples];
        m_label_samples = new short [m_n_examples];
        m_shuffledindex_dataset = new long [m_n_examples];
        m_repartition = new double [m_n_examples+1];
        m_labelledmeasure = new LabelledMeasure [m_n_examples];

        //
        initWeights();

        //
        m_n_classifiers_trained = 0;
        //for (int classifierNo = 0; classifierNo < m_n_classifiers; classifierNo++)
        int  c_classifiers =0; //tracking of number of unique classifiers trained
        int featureID;
        for (int classifierNo = 0; classifierNo < m_nrounds; classifierNo++)
        {
            if(c_classifiers==m_n_classifiers)
                {
                    m_weak_learners[classifierNo]->setMask(true);
                    m_weak_learners[classifierNo]->setFeatureMask(m_featuremask);

                }
            if (useSampling == true)
            {
                randomSampling();
                m_weak_learners[classifierNo]->setDataShuffledIndex(m_shuffledindex_dataset);
                m_weak_learners[classifierNo]->setDataWeights(m_weights_samples);
            }
            else
            {
                noSampling();
                m_weak_learners[classifierNo]->setDataShuffledIndex(m_shuffledindex_dataset);
                m_weak_learners[classifierNo]->setDataWeights(m_weights_samples);
            }

            // Once the weights and the shuffle data are set we can start training a weak classifier
            m_weak_learners[classifierNo]->setData(m_dataset);

            if (m_weak_learners[classifierNo]->train() == false) return false;

            //
//            File filn;
//            char fil[200];
//            sprintf(fil,"weight%d.data",classifierNo);
//            filn.open(fil,"w");
//            for (int pk =0;pk<m_n_examples;pk++)
//                filn.printf("%g\n",m_weights_samples[pk]);
//            filn.close();



            //check for the unique features
            featureID = m_weak_learners[classifierNo]->getFeatureID();
            if(m_featuremask[featureID]==0)
            {
                m_featuremask[featureID] = 1;
                c_classifiers++;
            }







            updateWeights(); // update weights for all examples

           // print(".........in this\n");
            m_n_classifiers_trained++;
        }

        // Normalize the weights
        double z_ = 0.0;
        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            print("> %g\n", m_weights[j]);
            z_ += exp(m_weights[j]);
        }

        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            m_weights[j] = exp(m_weights[j]) / z_;
            m_weak_learners[j]->setWeight(m_weights[j]);
            print("< %g\n", m_weights[j]);
        }

        //
        double mean_positive = 0.0;
        int n_positive = 0;
        double mean_negative = 0.0;
        int n_negative = 0;

        for (int i=0 ; i<m_n_examples ; i++)
        {
            Tensor *example = m_dataset->getExample(i);

            ShortTensor *target = (ShortTensor *) m_dataset->getTarget(i);
            short target_value = (*target)(0);

            double s = 0.0;
            for (int j = 0 ; j < m_n_classifiers ; j++)
            {
                Machine *m_ = m_weak_learners[j]->getMachine();
                m_->forward(*example);
                DoubleTensor *t_output = (DoubleTensor *) &m_->getOutput();

                s += m_weights[j] * (*t_output)(0);
            }

            m_labelledmeasure[i].measure = s;
            m_labelledmeasure[i].label = target_value;

            //print(" %g %d\n", s, target_value);
            if (target_value > 0)
            {
                mean_positive += s;
                n_positive ++;
            }
            else
            {
                mean_negative += s;
                n_negative ++;
            }
        }

        double frr = 0.0;
        double far = 0.0;
        double threshold = computeEER(m_labelledmeasure, m_n_examples, &frr, &far);

        print("   EER Threshold = %g \t FRR = %g \t FAR = %g\n", threshold, frr*100.0, far*100.0);

        print("   Mean negative = %g \t Mean Positive = %g\n", mean_negative / (double) n_negative, mean_positive / (double) n_positive);

        return true;
    }

    BoostingRoundTrainer::~BoostingRoundTrainer()
    {
        delete m_featuremask;
    }

}
