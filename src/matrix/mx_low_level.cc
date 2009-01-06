#include "mx_low_level.h"

namespace Torch {

/* __ip__ -- inner product */
double mxIp__(double * dp1, double * dp2, int len)
{
#ifdef VUNROLL
	int len4;
	double sum1, sum2, sum3;
#endif
	double sum;

	sum = 0.0;
#ifdef VUNROLL
	sum1 = sum2 = sum3 = 0.0;

	len4 = (len >> 2);
	len = len % 4;

	for (int i = 0; i < len4; i++)
	{
		int z = i << 2;
		sum  += dp1[z] * dp2[z];
		sum1 += dp1[z + 1] * dp2[z + 1];
		sum2 += dp1[z + 2] * dp2[z + 2];
		sum3 += dp1[z + 3] * dp2[z + 3];
	}
	sum += sum1 + sum2 + sum3;
	dp1 += (len4 << 2);
	dp2 += (len4 << 2);
#endif

	for (int i = 0; i < len; i++)
		sum += dp1[i] * dp2[i];

	return sum;
}

/* __mltadd__ -- scalar multiply and add c.f. v_mltadd() */
void mxdoubleMulAdd__(double * dp1, double * dp2, double s, int len)
{
#ifdef VUNROLL
	int len4;

	len4 = len / 4;
	len = len % 4;
	for (int i = 0; i < len4; i++)
	{
		dp1[4 * i] += s * dp2[4 * i];
		dp1[4 * i + 1] += s * dp2[4 * i + 1];
		dp1[4 * i + 2] += s * dp2[4 * i + 2];
		dp1[4 * i + 3] += s * dp2[4 * i + 3];
	}
	dp1 += 4 * len4;
	dp2 += 4 * len4;
#endif

	for (int i = 0; i < len; i++)
		dp1[i] += s * dp2[i];
}

/* __smlt__ scalar multiply array c.f. sv_mlt() */
void mxdoubleMul__(double * dp, double s, double * out, int len)
{
	for (int i = 0; i < len; i++)
		out[i] = s * dp[i];
}

/* __add__ -- add arrays c.f. v_add() */
void mxAdd__(double * dp1, double * dp2, double * out, int len)
{
	for (int i = 0; i < len; i++)
		out[i] = dp1[i] + dp2[i];
}

/* __sub__ -- subtract arrays c.f. v_sub() */
void mxSub__(double * dp1, double * dp2, double * out, int len)
{
	for (int i = 0; i < len; i++)
		out[i] = dp1[i] - dp2[i];
}

/* __zero__ -- zeros an array of floating point numbers */
void mxZero__(double * dp, int len)
{
	/* else, need to zero the array entry by entry */
	for (int i = 0; i < len; i++)
		dp[i] = 0.0;
}

}

