#include "BoostingTrainer.h"

namespace Torch
{
/////////////////////////////////////////////////////////////////////////////////////////
    BoostingTrainer::BoostingTrainer()
    {
        addBOption("boosting_by_sampling",	false,	"use sampling based on weights");

        m_n_examples = 0;
        m_n_classifiers = 0;
        m_n_classifiers_trained = 0;

        m_weights = NULL;
        m_weights_samples = NULL;
        m_label_samples = NULL;
        m_shuffledindex_dataset = NULL;
        m_repartition = NULL;
        m_labelledmeasure = NULL;
        verbose = getBOption("verbose");
    }
/////////////////////////////////////////////////////////////////////////////////////////
    bool BoostingTrainer::noSampling()
    {
        verbose = getBOption("verbose");
        if (verbose)
            print("   BoostingTrainer::noSampling()\n");

        if (m_n_examples <= 0)
        {
            Torch::error("BoostingTrainer::noSampling() impossible to init shuffle index.");

            return false;
        }

        for (int i=0;i<m_n_examples;i++)
            m_shuffledindex_dataset[i] = i;

        return true;
    }
////////////////////////////////////////////////////////////////////////////////
    double BoostingTrainer::forward(Tensor *example_)
    {
        double s = 0.0;
        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            Machine *m_ = m_weak_learners[j]->getMachine();
            m_->forward(*example_);
            DoubleTensor *t_output = (DoubleTensor *) &m_->getOutput();

            s +=  m_weak_learners[j]->getWeight()*(*t_output)(0);

        }

        return s;
    }
///////////////////////////////////////////////////////////////////////////////
    bool BoostingTrainer::randomSampling()
    {
        verbose = getBOption("verbose");
        if (verbose)
            print("   BoostingTrainer::randomSampling()\n");


        if (m_n_examples <= 0)
        {
            Torch::error("BoostingTrainer::randomSampling() impossible to init random weights.");

            return false;
        }

        THRandom_manualSeed(THRandom_seed());

        m_repartition[0] = 0.0;
        for (int i=0;i<m_n_examples;i++)
            m_repartition[i+1] = m_repartition[i] + m_weights_samples[i];

        for (int i=0;i<m_n_examples;i++)
        {
            double z = THRandom_uniform(0, 1);
            int left = 0;
            int right = m_n_examples;
            while (left+1 != right)
            {
                int center = (left+right)/2;
                if (m_repartition[center] < z)
                    left = center;
                else
                    right = center;
            }
            m_shuffledindex_dataset[i] = left;
        }

        return true;
    }
/////////////////////////////////////////////////////////////////////////////////////
    void BoostingTrainer::initWeights()
    {
        verbose = getBOption("verbose");
        if (verbose)
            print("   BoostingTrainer::initWeights()\n");

        if (m_n_examples > 0)
        {
            for (int i=0;i<m_n_examples;i++)
            {
                m_weights_samples[i] = 1.0/(double)(m_n_examples);
                m_label_samples[i] = 0;
            }

            for (int i = 0 ; i < m_n_classifiers ; i++) m_weights[i] = 0.0;
        }
        else Torch::error("BoostingTrainer::initWeights() no examples to initialize the weights.");
    }
/////////////////////////////////////////////////////////////////////////////////////////////////
    void BoostingTrainer::updateWeights()
    {
        if (verbose)
            print("   BoostingTrainer::updateWeights()\n");

        int fa = 0;
        int fr = 0;
        int np = 0;
        int nn = 0;
        bool t;
        double error_ = 0.0;

        //
        Machine *m_ = m_weak_learners[m_n_classifiers_trained]->getMachine();

        if (verbose)
            print("Number of examples in updateweights %d\n",m_n_examples);


        for (int i=0 ; i<m_n_examples ; i++)
        {
            Tensor *example = m_dataset->getExample(i);


            t= m_->forward(*example);

            if (t==false)
                print("BoostingTrainer::updateWeights There is a bug in forward machine \n");

            ShortTensor *target = (ShortTensor *) m_dataset->getTarget(i);
            short target_value = (*target)(0);
            DoubleTensor *t_output = (DoubleTensor *) &m_->getOutput();

            m_labelledmeasure[i].measure = t_output->get(0);

            m_labelledmeasure[i].label = target_value;

            m_label_samples[i] = 0;

            if (target_value == 0)
            {
                if ((*t_output)(0) > 0)
                {
                    fa++;
                    error_ += m_weights_samples[i];
                    m_label_samples[i] = 1;
                }
                else m_label_samples[i] = -1;

                nn++;
            }
            else if (target_value == 1)
            {
                if ((*t_output)(0) < 0)
                {
                    fr++;
                    error_ += m_weights_samples[i];
                    m_label_samples[i] = 1;
                }
                else m_label_samples[i] = -1;
                np++;

            }
        }

        if (verbose)
            print("   error = %g \t FAR = %g \t FRR = %g\n", error_, ((float) fa * 100.0 / (float) (nn+1)), ((float) fr * 100.0 / (float) (np+1)));

        double frr = 0.0;
        double far = 0.0;
        double threshold = computeEER(m_labelledmeasure, m_n_examples, &frr, &far);
        if (verbose)
            print("   EER Threshold = %g \t FRR = %g \t FAR = %g\n", threshold, frr*100.0, far*100.0);

        double beta = error_ / (1.0 - error_);

        //
        m_weights[m_n_classifiers_trained] = -log(beta); // log(1 / beta)

        if (verbose)
            print("   Machine weights = %g\n", m_weights[m_n_classifiers_trained]);

        //
        double z_ = 0.0;
        for (int i=0 ; i<m_n_examples ; i++)
        {
            if (m_label_samples[i] < 0)
                m_weights_samples[i] *= beta; // in fact exp(log(beta) * I{classification error})
            z_ += m_weights_samples[i];
        }
        for (int i=0 ; i<m_n_examples ; i++) m_weights_samples[i] /= z_;

        if (verbose)
            print("\n\n");


    }


//////////////////////////////////////////////////////////////////////////////////////
    bool BoostingTrainer::setWeakLearners(int n_classifiers_, WeakLearner **weak_learners_)
    {
        m_n_classifiers = n_classifiers_;
        m_weak_learners = weak_learners_;

        cleanup();

        m_weights = new double [m_n_classifiers];

        return true;
    }
///////////////////////////////////////////////////////////////////////////////////////////
    bool BoostingTrainer::train()
    {
        verbose = getBOption("verbose");
        if (verbose)
            print("BoostingTrainer::train() ...\n");

        //
        bool useSampling = getBOption("boosting_by_sampling");

        //
        m_n_examples = m_dataset->getNoExamples();

        // testing at least if the dataset has targets
        // let the StumpTrainer test the number of classes (> 1) and eventually deal with multiple classes
        if (m_dataset->hasTargets() != true)
        {
            warning("BoostingTrainer::train() no targets in the dataset.");

            return false;
        }

        if (verbose)
        {
            print(" + Number of weak classifiers: %d\n", m_n_classifiers);
            print(" + Number of examples: %d\n", m_n_examples);
        }

        m_weights_samples = new double [m_n_examples];
        m_label_samples = new short [m_n_examples];
        m_shuffledindex_dataset = new long [m_n_examples];
        m_repartition = new double [m_n_examples+1];
        m_labelledmeasure = new LabelledMeasure [m_n_examples];

        //
        initWeights();

        //
        m_n_classifiers_trained = 0;
        for (int classifierNo = 0; classifierNo < m_n_classifiers; classifierNo++)
        {
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

            //........saving weights at each iteration.......
            //  File filn;
            // char fil[200];
            //  sprintf(fil,"weight%d.data",classifierNo);
            //  filn.open(fil,"w");
            //  for (int pk =0;pk<m_n_examples;pk++)
            //   filn.printf("%g\n",m_weights_samples[pk]);
            //  filn.close();


            updateWeights(); // update weights for all examples

            m_n_classifiers_trained++;
        }

        // Normalize the weights
        double z_ = 0.0;
        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            z_ += exp(m_weights[j]);
        }

        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            m_weights[j] = exp(m_weights[j]) / z_;
            m_weak_learners[j]->setWeight(m_weights[j]);

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

        if (verbose)
        {
            print("   EER Threshold = %g \t FRR = %g \t FAR = %g\n", threshold, frr*100.0, far*100.0);

            print("   Mean negative = %g \t Mean Positive = %g\n", mean_negative / (double) n_negative, mean_positive / (double) n_positive);
        }

        return true;
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void BoostingTrainer::cleanup()
    {
        delete []m_weights;
        delete []m_weights_samples;
        delete []m_label_samples;
        delete []m_shuffledindex_dataset;
        delete []m_repartition;
        delete []m_labelledmeasure;

        m_weights = NULL;
        m_weights_samples = NULL;
        m_label_samples = NULL;
        m_shuffledindex_dataset = NULL;
        m_repartition = NULL;
        m_labelledmeasure = NULL;
    }
////////////////////////////////////////////////////////////////////////////////////////////////////////
    BoostingTrainer::~BoostingTrainer()
    {
        cleanup();
    }

}
