#include "trainer/BoostingRoundLBPTrainer.h"
#include "ip/ipLBP.h"

namespace Torch
{
    BoostingRoundLBPTrainer::BoostingRoundLBPTrainer()
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
    ////////////////////////////////////////////////////////////////////////////////


    double BoostingRoundLBPTrainer::forward(const Tensor *example)
    {
        //   print("BoostingRoundLBPTrainer::forward()\n");
        double s = 0.0;
        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            Machine *m_ = m_weak_learners[j]->getMachine();
            m_->forward(*example);
            DoubleTensor *t_output = (DoubleTensor *) &m_->getOutput();

            s +=  1.0*(*t_output)(0);

        }
        // print("Score %f\n",s);
        return s;
    }

    float BoostingRoundLBPTrainer::forwardScan(const Tensor &example,TensorRegion &trregion)
    {


	//store the actual tensor region.
	//int tposy,tposx,tsizey,tsizex;
	spCore *tempcore;

	Machine *m_ = m_weak_learners[0]->getMachine();
	tempcore = m_->getCore();
	TensorRegion tr =       ((ipLBP*)tempcore)->getTensorRegion();
	trregion.size[0] = tr.size[0];
	trregion.size[1] = tr.size[1];
    	//print("BoostingRoundLBPTrainer::forwardscan.....\n");
        float s = 0.0;
        for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            Machine *m_ = m_weak_learners[j]->getMachine();
            m_->setRegion(trregion);
            m_->forward(example);
            DoubleTensor *t_output = (DoubleTensor *) &m_->getOutput();

            s +=  1.0*(*t_output)(0);

        }
         for (int j = 0 ; j < m_n_classifiers ; j++)
        {
            Machine *m_ = m_weak_learners[j]->getMachine();
            m_->setRegion(tr);
        }



        // print("Score %f\n",s);
        return s;
    }

    ////////////////////////////////////////////////////////////////////////////
    bool BoostingRoundLBPTrainer::train()
    {
        verbose = getBOption("verbose");
        if (verbose)
            print("BoostingRoundLBPTrainer::train() ...\n");

        //
        bool useSampling = getBOption("boosting_by_sampling");
        m_nrounds = getIOption("number_of_rounds");

        if (verbose)
            print("Number of Classifiers %d, nRounds %d\n",m_n_classifiers,m_nrounds);

        //Check if the number of rounds are greater than number of weakClassifiers
        if (m_n_classifiers> m_nrounds)
        {
            Torch::error("BoostingRoundLBPTrainer::train() Classfiers > nRounds.");

            return false;
        }

        //get the number of features to initialize m_featuremask

        if (m_n_classifiers>0)
            m_features = m_weak_learners[0]->getmnFeatures();

        m_featuremask =new int[m_features];
        m_trackfeatures = new int[m_nrounds];

        for (int i=0;i<m_features;i++)
            m_featuremask[i] = 0;

        m_n_examples = m_dataset->getNoExamples();

        // testing at least if the dataset has targets
        // let the StumpTrainer test the number of classes (> 1) and eventually deal with multiple classes
        if (m_dataset->hasTargets() != true)
        {
            warning("BoostingRoundLBPTrainer::train() no targets in the dataset.");

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

        int  c_classifiers =0;          //tracking of number of unique classifiers trained
        int featureID;
        for (int classifierNo = 0; classifierNo < m_nrounds; classifierNo++)
        {
            if (c_classifiers==m_n_classifiers)
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


            //check for the unique features
            featureID = m_weak_learners[classifierNo]->getFeatureID();
            if (m_featuremask[featureID]==0)
            {
                m_featuremask[featureID] = 1;
                c_classifiers++;
            }




            m_trackfeatures[m_n_classifiers_trained] = featureID;


            updateWeights(); // update weights for all examples


            m_n_classifiers_trained++;
        }

        // Normalize the weights
        double z_ = 0.0;
        for (int j = 0 ; j < m_nrounds ; j++)
        {
            // print("> %g\n", m_weights[j]);
            z_ += exp(m_weights[j]);
        }

        for (int j = 0 ; j < m_nrounds; j++)
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
            for (int j = 0 ; j < m_nrounds ; j++)
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

        compressmachines();

        delete [] m_trackfeatures;

        return true;
    }
//////////////////////////////////////////////////////////////////////////////////////////
    void BoostingRoundLBPTrainer::compressmachines()
    {
        // first map the featureid to first m_n_classifiers
        int c=0;
        int max_bins = 512;
        int bins;
        int *bintrack=new int[m_n_classifiers];
        m_lut_t1= new double*[m_n_classifiers];
        double *lut_t2;
        for (int i=0;i<m_n_classifiers;i++)
        {
            m_lut_t1[i] = new double[max_bins];
            for (int j=0;j<max_bins;j++)
                m_lut_t1[i][j] = 0;
        }


        for (int i=0;i<m_features;i++)
        {
            if (m_featuremask[i]==1)
            {
                m_featuremask[i] = c;
                c++;
            }

        }

        // now you have to get the lut and
        int track;
        int *trackf = new int[m_n_classifiers];
        for (int i=0;i<m_nrounds;i++)
        {
            bins =  ((LBPRoundTrainer*)m_weak_learners[i])->getLUTSize();
            lut_t2 = ((LBPRoundTrainer*)m_weak_learners[i])->getLUT();
            track  = m_featuremask[m_trackfeatures[i]];
            bintrack[track] = bins;
            trackf[track] = m_trackfeatures[i];
            for (int j=0;j<bins;j++)
            {

                m_lut_t1[track][j] += m_weights[i]*lut_t2[j];
            }


        }

        //so finally we get the lut for a single feature
        //now set the first m_n_classifiers machines to this new lut

        for (int i=0;i<m_n_classifiers;i++)
        {
            bins = bintrack[i];
            //   print("%d\n",trackf[i]);
            ((IntLutMachine*)(m_weak_learners[i]->m_weak_classifier))->setCore( m_weak_learners[i]->m_features[trackf[i]]);

            ((IntLutMachine*)(m_weak_learners[i]->m_weak_classifier))->setParams(bins, m_lut_t1[i]);

        }


        delete[] trackf;
        delete [] bintrack;


    }

//////////////////////////////////////////////////////////////////////////////////////////

    bool BoostingRoundLBPTrainer::setWeakLearners(int n_classifiers_, WeakLearner **weak_learners_)
    {
        m_n_classifiers = n_classifiers_;
        m_weak_learners = weak_learners_;

        m_nrounds = getIOption("number_of_rounds");
        delete []m_weights;
        m_weights = new double [m_nrounds];

        return true;
    }

////////////////////////////////////////////////////////////////////////////////////////////
    BoostingRoundLBPTrainer::~BoostingRoundLBPTrainer()
    {

        delete []m_featuremask;

        for ( int i = 0; i < m_n_classifiers; i++)
        {
            delete [] m_lut_t1[i];
        }
        delete [] m_lut_t1;

    }
//////////////////////////////////////////////////////////////////////////////////////////
}
