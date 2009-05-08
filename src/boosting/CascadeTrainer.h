#ifndef _TORCH5SPRO_CASCADE_TRAINER_H_
#define _TORCH5SPRO_CASCADE_TRAINER_H_

#include "DataSet.h"
#include "ImageScanDataSet.h"
#include "measurer.h"
#include "MemoryDataSet.h"
#include "FTrainer.h"
#include "general.h"
namespace Torch
{

    class CascadeTrainer : public FTrainer
    {
    public:
        ///
    	CascadeTrainer();

	///
        bool setTrainers(FTrainer **m_ftrainer_, int n_cascade,  double * m_detection_rate_);
        bool setData(DataSet *m_pos_dataset_,DataSet *m_valid_dataset_, ImageScanDataSet *m_imagescandataset_);
	///
        virtual bool train();
       virtual double forward(Tensor *example_);

		///
        virtual ~CascadeTrainer();
        double *threshold;

    private:
	//
        //void cleanup();

        // Sample weights
        double *m_detection_rate;
        FTrainer ** m_ftrainer;
        ImageScanDataSet *m_imagescandataset;
        int n_cascade;
        int current_cascade;
        DataSet *m_pos_dataset; //storing all the positive data. the target will be changed to 0 if it is rejected.
        DataSet *m_valid_dataset;
        MemoryDataSet *m_dataset; //can we change the size of memory data set  by deleting and recreating it



	// Number of examples
	int p_examples;
	int n_examples;
//	double *threshold;
	 LabelledMeasure *m_labelledmeasure;



    void updateImageScanDataSet(int trainer_i);
    void updateDataSet(int trainer_i, DataSet *mdata_,char *str1);
    void getThreshold(DataSet *m_data);



    };

extern "C" int cmp_value(const void *p1, const void *p2);

}

#endif
