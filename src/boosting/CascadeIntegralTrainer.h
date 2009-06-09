#ifndef _TORCH5SPRO_CASCADE_INTEGRAL_TRAINER_H_
#define _TORCH5SPRO_CASCADE_INTEGRAL_TRAINER_H_

#include "DataSet.h"
#include "ImageScanDataSet.h"
#include "measurer.h"
#include "MemoryDataSet.h"
#include "FTrainer.h"
#include "general.h"
#include "ipIntegral.h"
namespace Torch
{

    class CascadeIntegralTrainer : public FTrainer
    {
    public:
        ///
        CascadeIntegralTrainer();

        ///
        bool setTrainers(FTrainer **m_ftrainer_, int n_cascade,  double * m_detection_rate_);
        bool setData(DataSet *m_pos_dataset_,DataSet *m_valid_dataset_, ImageScanDataSet *m_imagescandataset_);
        ///
        virtual bool train();
        virtual double forward(Tensor *example_);

        ///
        virtual ~CascadeIntegralTrainer();
        double *getStageThreshold()
        {
            return m_threshold;
        }




    private:

        void updateImageScanDataSet(int trainer_i);
        void updateImageScanDataSet_check(int trainer_i);
        void updateDataSet(int trainer_i, DataSet *mdata_,const char *string_text);
        void getThreshold(DataSet *m_data);


        double *m_detection_rate;
        FTrainer ** m_ftrainer;
        ImageScanDataSet *m_imagescandataset;
        int m_n_cascade;
        int m_current_cascade;
        DataSet *m_pos_dataset; //storing all the positive data. the target will be changed to 0 if it is rejected.
        DataSet *m_valid_dataset;
        MemoryDataSet *m_dataset; //can we change the size of memory data set  by deleting and recreating it


        ShortTensor m_target0;
        ShortTensor m_target1;


        double *m_threshold;




        // Number of examples
        int m_p_examples;
        int m_n_examples;

        LabelledMeasure *m_labelledmeasure;
        bool verbose;

        int height, width;






    };

    extern "C" int cmp_value(const void *p1, const void *p2);

}

#endif
