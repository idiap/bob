#ifndef _TORCHSPRO_OOURA_H_
#define _TORCHSPRO_OOURA_H_

namespace Torch
{

unsigned int nexthigher(unsigned int k);

}

#ifdef HAVE_OOURAFFT
extern "C"
{
double **alloc_2d_double(int, int);
void free_2d_double(double **);

double *alloc_1d_double(int);
void free_1d_double(double *);

int *alloc_1d_int(int);
void free_1d_int(int *);

void cdft(int, int, double *, int *, double *);
void cdft2d(int, int, int, double **, double *, int *, double *);
void ddct(int, int, double *, int *, double *);
void ddct2d(int, int, int, double **, double *, int *, double *);
void ddct8x8s(int, double **);
void ddct16x16s(int, double **);
}
#endif

#endif

