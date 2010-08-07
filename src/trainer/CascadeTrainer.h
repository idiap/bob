#ifndef _TORCH5SPRO_CASCADE_TRAINER_H_
#define _TORCH5SPRO_CASCADE_TRAINER_H_

#include "core/DataSet.h"
#include "ip/ImageScanDataSet.h"
#include "measurer/measurer.h"
#include "core/MemoryDataSet.h"
#include "FTrainer.h"
#include "core/general.h"
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
		virtual double forward(const Tensor *example_);

		///
		virtual ~CascadeTrainer();
		double *getStageThreshold()
		{
			return m_threshold;
		}


		// Set the preprocessing spCore
		void			setPreprocessor(spCore* core);

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

		spCore*		m_preprocessor;
};

extern "C" int cmp_value(const void *p1, const void *p2);

}

#endif
