#include "mat.h"
#include "general.h"

using namespace Torch;

namespace Torch
{
namespace Profile
{
	// Source: Numerical recipes!
	bool mat_ludcmp(double** a, int n, int* indx, double* d)
	{
		double* vv = new double[n];
		*d = 1.0;

		for (int i = 0; i < n; i ++)
		{
			double big = 0.0;
			for (int j = 0; j < n; j ++)
			{
				double temp;
				if ((temp = fabs(a[i][j])) > big)
					big = temp;
			}
			if (big == 0.0)
				return false;
			vv[i] = 1.0 / big;
		}

		for (int j = 0; j < n; j ++)
		{
			for (int i = 0; i < j; i ++)
			{
				double sum = a[i][j];
				for (int k = 0; k < i; k ++)
					sum -= a[i][k] * a[k][j];
				a[i][j] = sum;
			}

			double big = 0.0;
			int imax;
			for (int i = j; i < n; i ++)
			{
				double sum = a[i][j];
				for (int k = 0; k < j; k ++)
					sum -= a[i][k] * a[k][j];
				a[i][j] = sum;

				double dum;
				if ((dum = vv[i] * fabs(sum)) >= big)
				{
					big = dum;
					imax = i;
				}
			}

			if (j != imax)
			{
				for (int k = 0; k < n; k ++)
				{
					double dum = a[imax][k];
					a[imax][k] = a[j][k];
					a[j][k] = dum;
				}
				*d = -(*d);
				vv[imax] = vv[j];
			}

			indx[j] = imax;
			if (a[j][j] == 0.0)
				a[j][j] = 1.0e-20;

			if (j != n - 1)
			{
				double dum = 1.0 / (a[j][j]);
				for (int i = j + 1; i < n; i ++)
					a[i][j] *= dum;
			}

		}

		delete[] vv;
		return true;
	}
	void mat_lubksb(double** a, int n, int* indx, double b[])
	{
		int ii = -1;
		for (int i = 0; i < n; i ++)
		{
			int ip = indx[i];
			double sum = b[ip];
			b[ip] = b[i];
			if (ii >= 0)
			{
				for (int j = ii; j <= i - 1; j ++)
					sum -= a[i][j] * b[j];
			}
			else if (sum)
			{
				ii = i;
			}
			b[i] = sum;
		}
		for (int i = n - 1; i >= 0; i --)
		{
			double sum = b[i];
			for (int j = i + 1; j < n; j ++)
				sum -= a[i][j] * b[j];
			b[i] = sum / a[i][i];
		}
	}

	// Invert MxM matrixes and returns its determinant
	double mat_invert(double** mat, double** inv_mat, int size)
	{
		// Copy the initial matrix
		double** temp = new double*[size];
		for (int i = 0; i < size; i ++)
		{
			temp[i] = new double[size];
			for (int j = 0; j < size; j ++)
			{
				temp[i][j] = mat[i][j];
			}
		}

		// LU decomposition
		int* indx = new int[size];
		double d;
		if (mat_ludcmp(mat, size, indx, &d) == true)
		{
			// Compute the determinant
			for (int i = 0; i < size; i ++)
			{
				d *= mat[i][i];
			}

			// Invert the matrix
			double* col = new double[size];
			for (int j = 0; j < size; j ++)
			{
				for (int i = 0; i < size; i ++)
				{
					col[i] = 0.0;
				}
				col[j] = 1.0f;
				mat_lubksb(mat, size, indx, col);
				for (int i = 0; i < size; i ++)
				{
					inv_mat[i][j] = col[i];
				}
			}
			delete[] col;
		}
		else
		{
			d = 0.0;
		}

		// Check the inversion
		for (int i = 0; i < size; i ++)
		{
			for (int j = 0; j < size; j ++)
			{
				double sum = 0.0;
				for (int k = 0; k < size; k ++)
				{
					sum += temp[i][k] * inv_mat[k][j];
				}
				CHECK_FATAL(fabs(sum - (i == j ? 1.0 : 0.0)) < 0.000005);
			}
		}

		// OK
		for (int i = 0; i < size; i ++)
		{
			delete[] temp[i];
		}
		delete[] temp;
		delete[] indx;
		return d;
	}

	double mat_hypot(double a, double b)
	{
		double r;

		if (fabs(a) > fabs(b))
		{
			r = b/a;
			r = fabs(a) * sqrt(1.0 + r * r);

		}
		else if (b != 0.0)
		{
			r = a/b;
			r = fabs(b) * sqrt(1.0 + r * r);

		}
		else
		{
			r = 0.0;
		}

		return r;

	}

	// Symmetric Householder reduction to tridiagonal form.
	void mat_tred2(double** mat, double** evectors, double* evalues, double* buf, int size)
	{
		//  This is derived from the Algol procedures tred2 by
		//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
		//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
		//  Fortran subroutine in EISPACK.
		for (int j = 0; j < size; j ++)
		{
			evalues[j] = evectors[size - 1][j];
		}

		// Householder reduction to tridiagonal form.
		for (int i = size - 1; i > 0; i --)
		{
			// Scale to avoid under/overflow.
			double scale = 0.0f;
			double h = 0.0;
			for (int k = 0; k < i; k ++)
			{
				scale = scale + fabs(evalues[k]);
			}
			if (scale == 0.0)
			{
				buf[i] = evalues[i - 1];
				for (int j = 0; j < i; j ++)
				{
					evalues[j] = evectors[i - 1][j];
					evectors[i][j] = 0.0;
					evectors[j][i] = 0.0;
				}
			}
			else
			{
				// Generate Householder vector.
				for (int k = 0; k < i; k ++)
				{
					evalues[k] /= scale;
					h += evalues[k] * evalues[k];
				}
				double f = evalues[i - 1];
				double g = sqrt(h);
				if (f > 0.0)
				{
					g = -g;
				}
				buf[i] = scale * g;
				h = h - f * g;
				evalues[i - 1] = f - g;
				for (int j = 0; j < i; j ++)
				{
					buf[j] = 0.0;
				}

				// Apply similarity transformation to remaining columns.
				for (int j = 0; j < i; j ++)
				{
					f = evalues[j];
					evectors[j][i] = f;
					g = buf[j] + evectors[j][j] * f;
					for (int k = j + 1; k <= i - 1; k ++)
					{
						g += evectors[k][j] * evalues[k];
						buf[k] += evectors[k][j] * f;
					}
					buf[j] = g;
				}
				f = 0.0;
				for (int j = 0; j < i; j ++)
				{
					buf[j] /= h;
					f += buf[j] * evalues[j];
				}
				double hh = f / (h + h);
				for (int j = 0; j < i; j ++)
				{
					buf[j] -= hh * evalues[j];
				}
				for (int j = 0; j < i; j ++)
				{
					f = evalues[j];
					g = buf[j];
					for (int k = j; k <= i - 1; k ++)
					{
						evectors[k][j] -= (f * buf[k] + g * evalues[k]);
					}
					evalues[j] = evectors[i - 1][j];
					evectors[i][j] = 0.0;
				}
			}
			evalues[i] = h;
		}

		// Accumulate transformations.
		for (int i = 0; i < size - 1; i ++)
		{
			evectors[size - 1][i] = evectors[i][i];
			evectors[i][i] = 1.0;
			double h = evalues[i + 1];
			if (h != 0.0)
			{
				for (int k = 0; k <= i; k ++)
				{
					evalues[k] = evectors[k][i+1] / h;
				}
				for (int j = 0; j <= i; j ++)
				{
					double g = 0.0;
					for (int k = 0; k <= i; k ++)
					{
						g += evectors[k][i + 1] * evectors[k][j];
					}
					for (int k = 0; k <= i; k ++)
					{
						evectors[k][j] -= g * evalues[k];
					}
				}
			}
			for (int k = 0; k <= i; k ++)
			{
				evectors[k][i + 1] = 0.0;
			}
		}
		for (int j = 0; j < size; j ++)
		{
			evalues[j] = evectors[size - 1][j];
			evectors[size - 1][j] = 0.0;
		}
		evectors[size - 1][size - 1] = 1.0;
		buf[0] = 0.0;
	}

	// Symmetric tridiagonal QL algorithm.
	void mat_tql2(double** mat, double** evectors, double* evalues, double* buf, int size)
	{
		//  This is derived from the Algol procedures tql2, by
		//  Bowdler, Martin, Reinsch, and Wilkinson, Handbook for
		//  Auto. Comp., Vol.ii-Linear Algebra, and the corresponding
		//  Fortran subroutine in EISPACK.

		for (int i = 1; i < size; i ++)
		{
			buf[i - 1] = buf[i];
		}
		buf[size - 1] = 0.0;

		double f = 0.0;
		double tst1 = 0.0;
		static const double eps = pow(2.0, -52.0);
		for (int l = 0; l < size; l ++)
		{
			// Find small subdiagonal element
			tst1 = max(tst1, fabs(evalues[l]) + fabs(buf[l]));
			int m = l;
			while (m < size && fabs(buf[m]) > eps * tst1)
			{
				m ++;
			}

			// If m == l, evalues[l] is an eigenvalue,
			// otherwise, iterate.
			if (m > l)
			{
				int iter = 0;
				do
				{
					iter = iter + 1;  // (Could check iteration count here.)

					// Compute implicit shift
					double g = evalues[l];
					double p = (evalues[l + 1] - g) / (2.0 * buf[l]);
					double r = hypot(p, 1.0);
					if (p < 0.0)
					{
						r = -r;
					}
					evalues[l] = buf[l] / (p + r);
					evalues[l + 1] = buf[l] * (p + r);
					double dl1 = evalues[l + 1];
					double h = g - evalues[l];
					for (int i = l + 2; i < size; i ++)
					{
						evalues[i] -= h;
					}
					f = f + h;

					// Implicit QL transformation.
					p = evalues[m];
					double c = 1.0;
					double c2 = c;
					double c3 = c;
					double el1 = buf[l + 1];
					double s = 0.0;
					double s2 = 0.0;
					for (int i = m - 1; i >= l; i --)
					{
						c3 = c2;
						c2 = c;
						s2 = s;
						g = c * buf[i];
						h = c * p;
						r = hypot(p, buf[i]);
						buf[i + 1] = s * r;
						s = buf[i] / r;
						c = p / r;
						p = c * evalues[i] - s * g;
						evalues[i + 1] = h + s * (c * g + s * evalues[i]);

						// Accumulate transformation.
						for (int k = 0; k < size; k ++)
						{
							h = evectors[k][i + 1];
							evectors[k][i + 1] = s * evectors[k][i] + c * h;
							evectors[k][i] = c * evectors[k][i] - s * h;
						}
					}
					p = -s * s2 * c3 * el1 * buf[l] / dl1;
					buf[l] = s * p;
					evalues[l] = c * p;

					// Check for convergence.
				} while (fabs(buf[l]) > eps * tst1);
			}
			evalues[l] = evalues[l] + f;
			buf[l] = 0.0;
		}

		// Sort eigenvalues and corresponding vectors.
		for (int i = 0; i < size - 1; i ++)
		{
			int k = i;
			double p = evalues[i];
			for (int j = i + 1; j < size; j ++)
			{
				if (evalues[j] < p)
				{
					k = j;
					p = evalues[j];
				}
			}
			if (k != i)
			{
				evalues[k] = evalues[i];
				evalues[i] = p;
				for (int j = 0; j < size; j ++)
				{
					p = evectors[j][i];
					evectors[j][i] = evectors[j][k];
					evectors[j][k] = p;
				}
			}
		}
	}

	bool mat_eigen_sym(double** mat, double** evectors, double* evalues, int size)
	{
		// Check if the matrix is symetric
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
			{
				CHECK_FATAL(fabs(mat[i][j] - mat[j][i]) < 0.000005);
			}

		// Copy the initial matrix
		double** temp = new double*[size];
		for (int i = 0; i < size; i ++)
		{
			temp[i] = new double[size];
			for (int j = 0; j < size; j ++)
			{
				temp[i][j] = mat[i][j];
			}
		}

		// Compute the eigen vectors
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
			{
				evectors[i][j] = mat[i][j];
			}

		double* buf = new double[size];
		mat_tred2(mat, evectors, evalues, buf, size);	// Tridiagonalize
		mat_tql2(mat, evectors, evalues, buf, size);	// Diagonalize
		delete[] buf;

		// Check the eigen vectors
		for (int e = 0; e < size - 1; e ++)
		{
			CHECK_FATAL(evalues[e] < evalues[e + 1]);
		}
		for (int e = 0; e < size; e ++)
		{
			for (int i = 0; i < size; i ++)
			{
				double sum = 0.0;
				for (int j = 0; j < size; j ++)
				{
					sum += temp[i][j] * evectors[j][e];
				}
				CHECK_FATAL(fabs(sum - evalues[e] * evectors[i][e]) < 0.000005);
			}
		}

		// OK
		for (int i = 0; i < size; i ++)
		{
			delete[] temp[i];
		}
		delete[] temp;
		return true;
	}

	// Computes the square root of some symetrical matrix
	bool mat_sqrt_sym(double** mat, double** sqrt_mat, int size)
	{
		// Check if the matrix is symetric
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
			{
				CHECK_FATAL(mat[i][j] == mat[j][i]);
			}

		// Copy the initial matrix
		double** temp = new double*[size];
		for (int i = 0; i < size; i ++)
		{
			temp[i] = new double[size];
			for (int j = 0; j < size; j ++)
			{
				temp[i][j] = mat[i][j];
			}
		}

		// Compute the eigen vectors
		double** evectors = new double*[size];
		double* evalues = new double[size];
		for (int i = 0; i < size; i ++)
		{
			evectors[i] = new double[size];

		}
		mat_eigen_sym(mat, evectors, evalues, size);

		// Check the eigenvectors are orthogonal
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
			{
				double sum = 0.0;
				for (int k = 0; k < size; k ++)
				{
					sum += evalues[k] * evectors[i][k] * evectors[j][k];
				}
				CHECK_FATAL(fabs(sum - temp[i][j]) < 0.000005);

				sum = 0.0;
				for (int k = 0; k < size; k ++)
				{
					sum += evectors[i][k] * evectors[j][k];
				}
				CHECK_FATAL(fabs(sum - (i == j ? 1.0 : 0.0)) < 0.000005);
			}

		// Compute the square root
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
			{
				double sum = 0.0;
				for (int k = 0; k < size; k ++)
				{
					sum += sqrt(fabs(evalues[k])) * evectors[i][k] * evectors[j][k];
				}
				sqrt_mat[i][j] = sum;
			}

		// Check the square root
		for (int i = 0; i < size; i ++)
			for (int j = 0; j < size; j ++)
			{
				double sum = 0.0;
				for (int k = 0; k < size; k ++)
				{
					sum += sqrt_mat[i][k] * sqrt_mat[k][j];
				}
				CHECK_FATAL(fabs(sum - temp[i][j]) < 0.000005);
			}

		// OK
		for (int i = 0; i < size; i ++)
		{
			delete[] temp[i];
			delete[] evectors[i];
		}
		delete[] temp;
		delete[] evectors;
		delete[] evalues;
		return true;
	}
}
}
