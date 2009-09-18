#include "measurer.h"

namespace Torch {

extern "C" int cmp_labelledmeasure(const void *p1, const void *p2)
{
	const LabelledMeasure& v1 = *((const LabelledMeasure *)p1);
	const LabelledMeasure& v2 = *((const LabelledMeasure *)p2);

	if(v1.measure > v2.measure) return 1;
  	if(v1.measure < v2.measure) return -1;

	return 0;
}
double computeTH(LabelledMeasure* measures, int n_size, double dr)
{
   qsort(measures,n_size,sizeof(LabelledMeasure),cmp_labelledmeasure);
int n;
    n =(int)((1-dr)*n_size);
return measures[n].measure;
}



double computeEER(LabelledMeasure* measures, int n, double* frr, double* far, int number_of_positives_, bool sort)
{
	if(sort) qsort(measures, n, sizeof(LabelledMeasure), cmp_labelledmeasure);

	// find the number of positive and negative targets
	int number_of_positives = 0;
	if(number_of_positives_ < 0)
	{
		for(int i=0;i<n;i++)
			if(measures[i].label == 1) number_of_positives++;
	}
	else number_of_positives = number_of_positives_;

	int number_of_negatives = n - number_of_positives;

	int n_i = number_of_negatives;
	int n_c = 0;
	double cost_min = 100;
	double cost = cost_min - 1;
	double thrd_min = measures[0].measure;

	for(int i=0;i<n;i++)
	{
		if(measures[i].label) n_c++;
		else n_i--;

		double current_thrd = measures[i].measure;
		double current_far=n_i/(double)number_of_negatives;
		double current_frr=n_c/(double)number_of_positives;
		cost = fabs(current_far-current_frr);

		if(cost < cost_min)
		{
			cost_min = cost;
			*frr = current_frr;
			*far = current_far;
			if(i != n -1)
				thrd_min = (current_thrd + measures[i+1].measure)/2.0;
			else
				thrd_min = current_thrd;
		}
	}
	return (thrd_min);
}

double computeHTER(LabelledMeasure* measures, int n, double* frr, double* far, int number_of_positives_, bool sort, float ratio_far)
{
	if(sort) qsort(measures, n, sizeof(LabelledMeasure), cmp_labelledmeasure);

	//find the number of positive and negative targets
	int number_of_positives = 0;
	if(number_of_positives_ < 0)
	{
		for(int i=0;i<n;i++)
			if(measures[i].label == 1)
				number_of_positives++;
	}
	else number_of_positives = number_of_positives_;

	int number_of_negatives = n - number_of_positives;

	int n_i = number_of_negatives;
	int n_c = 0;
	double cost_min = 100;
	double cost = cost_min - 1;
	double thrd_min = measures[0].measure;

	for(int i=0;i<n;i++)
	{
		if (measures[i].label) n_c++;
		else n_i--;

		double current_thrd = measures[i].measure;
		double current_far=n_i/(double)number_of_negatives;
		double current_frr=n_c/(double)number_of_positives;
		cost = ratio_far*current_far+current_frr;
		if(cost < cost_min)
		{
			cost_min = cost;
			*frr = current_frr;
			*far = current_far;
			if(i != n -1)
				thrd_min = (current_thrd + measures[i+1].measure)/2.0;
			else
				thrd_min = current_thrd;
		}
	}
	return (thrd_min);
}

/*
	void computeFaFr(real thrd, Int_real* to_sort, int n, real* frr, real* far, int number_of_clients_,bool sort){

		//sort the scores
		if(sort)
			qsort(to_sort, n, sizeof(Int_real), compar_int_real);

		//find the number of positive and negative targets
		int number_of_clients = 0;
		if(number_of_clients_ < 0){
			for(int i=0;i<n;i++)
				if(to_sort[i].the_int == 1)
					number_of_clients++;
		}else
			number_of_clients = number_of_clients_;
		int number_of_impostors = n - number_of_clients;

		int n_i = number_of_impostors;
		int n_c = 0;
		real thrd_min = to_sort[0].the_real;

		int i = 0;
		while((i < n - 1) && (thrd>thrd_min)){
			if (to_sort[i].the_int){
				n_c++;
			}else{
				n_i--;
			}
			thrd_min = to_sort[i+1].the_real;
			i++;
		}
		*far=n_i/(real)number_of_impostors;
		*frr=n_c/(real)number_of_clients;
	}
*/

}
