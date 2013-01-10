/**
 * @file bob/sp/Ceps.h
 * @date Wed Jan 11:10:20 2013 +0200
 * @author Elie Khoury <Elie.Khoury@idiap.ch>
 *
 * @brief Implement Linear and Mel Frequency Cepstral Coefficients
 * functions (MFCC and LFCC)
 *
 * Copyright (C) 2011-2013 Idiap Research Institute, Martigny, Switzerland
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


#ifndef BOB_SP_CEPS_H
#define BOB_SP_CEPS_H

#include <blitz/array.h>


const double ENERGY_FLOOR = 1.0;
const double FBANK_OUT_FLOOR = 1.0;
namespace bob {
/**
 * \ingroup libsp_api
 * @{
 *
 */
namespace sp {

/**
 * @brief This class is used to test the Ceps class (private methods)
 */
class CepsTest;

/**
 * @brief This class allows the extraction of features from raw audio data.
 * References:
 *  1. SPro tools (http://www.irisa.fr/metiss/guig/spro/spro-4.0.1/spro.html)
 *  2. Wikipedia (http://en.wikipedia.org/wiki/Mel-frequency_cepstrum).
 */
class Ceps
{
public:
	/**
	 * @brief Constructor: Initialize working arrays
	 */
	Ceps( double sf, int win_length_ms, int win_shift_ms, int n_filters, int n_ceps,
			double f_min, double f_max, double delta_win, const blitz::Array<double,1>& data_array);

	/**
	 * @brief Get the Cepstral Shape
	 */
	blitz::TinyVector<int,2> getCepsShape(int n_size) const;

	/**
	 * @brief Compute Cepstral features
	 */
	void CepsAnalysis(int n_size, blitz::Array<double,2>& ceps_2D);

	/**
	 * @brief Reinitialize member variables
	 */
	void reinit(double dct_norm, bool fb_linear, bool withEnergy, bool withDelta,
			bool withDeltaDelta, bool withDeltaEnergy, bool withDeltaDeltaEnergy);

	/**
	 * @brief Destructor
	 */
	virtual ~Ceps();


private:
	/**
	 * @brief Compute the first derivatives
	 */
	void addDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size);
	/**
	 * @brief Compute the second derivatives
	 */
	void addDeltaDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size);
	/**
	 * @brief Mean Normalisation of the features
	 */
	blitz::Array<double,2> dataZeroMean(blitz::Array<double,2>& frames, bool norm_energy, int n_frames, int frame_size);

	static double mel(double f);
	static double MelInv(double f);
	void emphasis(blitz::Array<double,1> &data,int n,double a);
	void hammingWindow(blitz::Array<double,1> &data);
	void logFilterBank(blitz::Array<double,1>& x);
	void logTriangularFBank(blitz::Array<double,1>& data);
	double logEnergy(blitz::Array<double,1> &data);
	void transformDCT();
	void initWindowSize();
	void initCache();

	double m_sf;
	int m_win_length;
	int m_win_shift;
	int m_win_size;
	int m_nfilters;
	int m_nceps;
	double m_f_min;
	double m_f_max;
	int m_delta_win;
	blitz::Array<double,1> m_data_array;
	bool m_fb_linear;
	double m_dct_norm;
	bool m_withEnergy;
	bool m_withDelta;
	bool m_withDeltaDelta;
	bool m_withDeltaEnergy;
	bool m_withDeltaDeltaEnergy;
	blitz::Array<double,2> m_dct_kernel;
	blitz::Array<double,1> m_hamming_kernel;
	blitz::Array<double,1> m_frame;
	blitz::Array<double,1> m_filters;
	blitz::Array<double,1> m_ceps_coeff;
	blitz::Array<int,1>  m_p_index;

	friend class TestCeps;
};

class TestCeps
{
public:
	TestCeps(Ceps& ceps);
	Ceps& m_ceps;

	// Methods to test
	double mel(double f) { return m_ceps.mel(f); }
	double MelInv(double f) { return m_ceps.MelInv(f); }
	blitz::TinyVector<int,2> getCepsShape(int n_size) const
													{ return m_ceps.getCepsShape( n_size); }
	blitz::Array<double,1> getFilter(void) {return m_ceps.m_filters;}
	blitz::Array<double,1> getFeatures(void) {return m_ceps.m_ceps_coeff;}

	void CepsAnalysis(  int n_size, blitz::Array<double,2>& ceps_2D)
	{m_ceps.CepsAnalysis(n_size, ceps_2D);
	}
	void hammingWindow(blitz::Array<double,1>& data){ m_ceps.hammingWindow(data);}
	void emphasis(blitz::Array<double,1>& data,int n,double a){m_ceps.emphasis(data, n, a);}
	void logFilterBank(blitz::Array<double,1>& x){m_ceps.logFilterBank(x);}
	void logTriangularFBank(blitz::Array<double,1>& data){m_ceps.logTriangularFBank(data);}
	double logEnergy(blitz::Array<double,1> &data){return m_ceps.logEnergy(data);}
	void transformDCT(){m_ceps.transformDCT();}
	void addDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size){
		m_ceps.addDelta(frames, win_size, n_frames, frame_size);
	}
	void addDeltaDelta(blitz::Array<double,2>& frames, int win_size, int n_frames, int frame_size){
		m_ceps.addDeltaDelta(frames, win_size, n_frames, frame_size);
	}
	blitz::Array<double,2> dataZeroMean(blitz::Array<double,2>& frames, bool norm_energy, int n_frames, int frame_size){
		return m_ceps.dataZeroMean(frames,norm_energy, n_frames, frame_size);
	}

};

}
}

#endif /* BOB_SP_CEPS_H */
