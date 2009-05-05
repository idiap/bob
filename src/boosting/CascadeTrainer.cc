#include "CascadeTrainer.h"

namespace Torch
{
    CascadeTrainer::CascadeTrainer()
    {
        //addBOption("boosting_by_sampling",	false,	"use sampling based on weights");

        p_examples = 0;
        n_examples =0;

        n_cascade = 0;
        m_pos_dataset = NULL;
        m_dataset=NULL;
        m_labelledmeasure = NULL;

    }

//////////////////////////////////////////////////////////////////////////////////////////////

    bool CascadeTrainer::setTrainers(FTrainer **m_ftrainer_,int n_cascade_, double *m_detection_rate_)
    {
        m_ftrainer = m_ftrainer_;
        n_cascade = n_cascade_;
        m_detection_rate = m_detection_rate_;

        return true;
    }
//////////////////////////////////////////////////////////////////////////////////////////////
    bool CascadeTrainer::setData(DataSet *m_pos_dataset_,DataSet *m_valid_dataset_,ImageScanDataSet *m_imagescandataset_)
    {

        m_pos_dataset  = m_pos_dataset_;
        m_valid_dataset = m_valid_dataset_;
        m_imagescandataset = m_imagescandataset_;
        return true;

    }
    //////////////////////////////////////////////////////////////////////////////////////////////
    bool CascadeTrainer::train()
    {
        print("CascadeTrainer::train() ...\n");

        //
//        //As of now you can either train with integral image or just
//        //otherwise if we pass both then it should know which plane to operate on
//
//
//        // first get the count of +ve examples which have target = 1
        ShortTensor *target0 = new ShortTensor(1);
        target0->fill(0);
        ShortTensor *target1 = new ShortTensor(1);
        target1->fill(1);
        // Test the targets (rejection of samples)
        DoubleTensor reject_target(1), accept_target(1);
        reject_target.fill(-1.0);
        accept_target.fill(1.0);


        Tensor *tensor;
        int tp_examples = m_pos_dataset->getNoExamples();
        //height and width can be obtained by looking at the size
        int height, width;
        int p_count;
        int n_examples;

        tensor = m_pos_dataset->getExample(0);
        print("height %d, width %d\n",tensor->size(0),tensor->size(1));
        height = tensor->size(0);
        width = tensor->size(1);
        Tensor *example;
        long n_scanexamples;
        int n_count;
        threshold = new double[n_cascade];
//
//
//
//
//





        for (int mt = 0;mt< n_cascade;mt++)
        {


            current_cascade = mt;
            p_count = 0;
            for (int i = 0;i<tp_examples;i++)
            {
                ShortTensor *target = (ShortTensor *) m_pos_dataset->getTarget(i);
                short target_value = (*target)(0);
                if (target_value == 1)
                    p_count++;
            }
            // print("Number of Positive patterns remaining: %d\n",p_count);
//
            //now fill the memory data set with equal number of +ve and -ve patterns

            if (m_dataset != NULL) delete m_dataset;
            //now create a new dataset
            n_examples = 2*p_count;
            m_dataset = 	new MemoryDataSet(n_examples, Tensor::Double, true, Tensor::Short);

            //first fill with positive patterns

            p_count=0;

            for (int i=0;i<tp_examples;i++)
            {
                ShortTensor *target = (ShortTensor *) m_pos_dataset->getTarget(i);
                short target_value = (*target)(0);
                if (target_value == 1)
                {
                    m_dataset->getExample(p_count)->resize(height, width);
                    example = m_pos_dataset->getExample(i);
                    m_dataset->getExample(p_count)->copy(example);
                    m_dataset->setTarget(p_count, target1);
                    p_count++;
                }
            }


            n_scanexamples = m_imagescandataset->getNoExamples();
            print("Number of Positive patterns remaining: %d\n",p_count);
            //next fill with negative patterns
            //double imgtarget;
            //is it possible to make it random here to get ramdom patterns
            n_count =0;
            for (long i=0;i<n_scanexamples;i++)
            {
                //have to check if the target is +1 and fill withit
                if (((DoubleTensor*)m_imagescandataset->getTarget(i))->get(0)>0.0)
                {

                    m_dataset->getExample(p_count+n_count)->resize(height, width);
                    example = m_imagescandataset->getExample(i);
                    m_dataset->getExample(p_count+n_count)->copy(example);
                    m_dataset->setTarget(p_count+n_count, target0);
                    n_count++;

                    if (n_count>=p_count)
                        break;


                }

            }
            if (n_count<p_count)
            {
                print("There are not enough -ve patterns for training Aborting\n");
                return false;
            }
            //---------------finished copying data to memory------------
//
            print("Cascade iteration number %d\n",mt);
            print("Number of examples in Memory is %d\n",m_dataset->getNoExamples());
            m_ftrainer[mt]->setData(m_dataset);
            //  print("Number of examples in Memory is %d\n",m_dataset->getNoExamples());
            m_ftrainer[mt]->train();


//obtain the threshold
            if (m_labelledmeasure !=NULL) delete m_labelledmeasure;
            m_labelledmeasure = new LabelledMeasure[p_count];
            print("Obtaining Threshold\n");
            getThreshold(m_valid_dataset);
            updateDataSet(mt,m_pos_dataset,"training");
            updateDataSet(mt,m_valid_dataset,"validation");
            updateImageScanDataSet(mt);
//
//
//
//
//
//
//        //---------------obtain the threshold--------------------
//        // as of now estimate on the +ve training dataset
//        // later can pass validation dataset for obtaining the threshold.
//
//
//        //update the +ve training patterns
//        //update the Imagescan data set with this threshold
//        ...
//
//
//
//
        }//end of for mt

        return true;
    }
    //////////////
    void CascadeTrainer::getThreshold(DataSet *m_data)
    {
       // Torch::print("CascadeTrainer::getThrehsold()\n");

        int tp_examples = m_data->getNoExamples();
        int count = 0;
        for (int i=0;i<tp_examples;i++)
        {
            ShortTensor *target = (ShortTensor *) m_data->getTarget(i);
            short target_value = (*target)(0);
            if (target_value == 1)
            {
                //m_dataset->getExample(p_count)->resize(height, width);
                // example = m_pos_dataset->getExample(i);
                //  m_dataset->getExample(p_count)->copy(example);
                //m_dataset->setTarget(p_count, target1);
                //  p_count++;
                Tensor *example = m_data->getExample(i);
                m_labelledmeasure[count].measure = m_ftrainer[current_cascade]->forward(example);

                // print("Score %f\n",m_labelledmeasure[count].measure);
                m_labelledmeasure[count].label = target_value;
                count++;

            }
        }
    // Torch::print("CascadeTrainer::getThrehsold 2()\n");


        threshold[current_cascade]= computeTH(m_labelledmeasure, count, m_detection_rate[current_cascade]);
        print("Threshold for stage %d is %f\n",current_cascade,threshold[current_cascade]);

    }
////////////////////////////////////////////////////////////////////////////////
    void CascadeTrainer::updateImageScanDataSet(int trainer_i)
    {
        long n_scanexamples = m_imagescandataset->getNoExamples();
        //next fill with negative patterns
        //double imgtarget;
        //is it possible to make it random here to get ramdom patterns
        int nc =0;
        Tensor *example;
        double score;
        DoubleTensor reject_target(1);

        reject_target.fill(-1.0);

        for (long i=0;i<n_scanexamples;i++)
        {
            //have to check if the target is +1 and fill withit
            if (((DoubleTensor*)m_imagescandataset->getTarget(i))->get(0)>0.0)
            {

                //   m_dataset->getExample(p_count+n_count)->resize(height, width);
                example = m_imagescandataset->getExample(i);
                // m_dataset->getExample(p_count+n_count)->copy(example);
                //  n_count++;
                //  if(n_count>=p_count)
                //      break;
                score=m_ftrainer[trainer_i]->forward(example);
                if (score>threshold[current_cascade])
                    // set the pattern to reject
                {
                    nc++;
                }
                else

                    m_imagescandataset->setTarget(i, &reject_target);

            }
            if(i%1000001 == 0)
                print("%d ..",i);

        }

        print("Number of -ve patterns left = %ld\n",nc);

    }
    double CascadeTrainer::forward(Tensor *example_)
    {
        return 1.0;
    }

    ////////////////////////////////////////////////////////////////////////////////
    void CascadeTrainer::updateDataSet(int trainer_i, DataSet *mdata_,char *str1)
    {

        //update the positive training and/or validation dataset

        //get the target value and only process it is = 1
        ShortTensor *target0 = new ShortTensor(1);
        target0->fill(0);

        int tp_examples = mdata_->getNoExamples();
        int count = 0;
        for (int i=0;i<tp_examples;i++)
        {
            ShortTensor *target = (ShortTensor *) mdata_->getTarget(i);
            short target_value = (*target)(0);
            if (target_value == 1)
            {
                //m_dataset->getExample(p_count)->resize(height, width);
                // example = m_pos_dataset->getExample(i);
                //  m_dataset->getExample(p_count)->copy(example);
                //m_dataset->setTarget(p_count, target1);
                //  p_count++;
                Tensor *example = mdata_->getExample(i);
                if (m_ftrainer[current_cascade]->forward(example)<threshold[current_cascade])
                {

                    mdata_->setTarget(i, target0);
                    // print("Score %f\n",m_labelledmeasure[count].measure);

                    count++;
                }

            }
        }
        print("Number of %s examples rejected  = %d\n",str1,count);


    }
    ////////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////////////////////
    CascadeTrainer::~CascadeTrainer()
    {
        //  cleanup();
    }

}
