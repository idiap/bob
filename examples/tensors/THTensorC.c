#include "torch5spro.h"

void print(const char *fmt, ...)
{
	static char msg[1024];
	va_list args;

         /* vasprintf not standard */
         /* vsnprintf: how to handle if does not exists? */
	va_start(args, fmt);
	vsnprintf(msg, 1024, fmt, args);
	va_end(args);

	printf(msg);
}

int main()
{
	print("THTensors\n");

	int w = 10;
	int h = 10;
	int x, y;

	double dv;
	char cv;
	short sv;
	int iv;
	long lv;
	float fv;

	//
	// default Tensor (in double)
	print("\n\n");
	print("Testing default Tensor (double)\n\n");

	print("Allocating a Tensor of double of size %dx%d ...\n", w, h);
	THTensor *dt = THTensor_newWithSize2d(w, h);
	print("Tensor allocated (n_dim=%d size=[%dx%d])\n", dt->nDimension, dt->size[0], dt->size[1]);

	for(x = 0 ; x < w ; x++)
		for(y = 0 ; y < h ; y++)
			THTensor_set2d(dt, x, y, 0.0);
	for(x = 0 ; x < w ; x++)
	{
	   	for(y = 0 ; y < h ; y++)
		{
			dv = THTensor_get2d(dt, x, y);
			print(" %g", dv);
		}
		print("\n");
	}

	THTensor *t = THTensor_newWithSize2d(h, w);
	TH_TENSOR_APPLY2(double, t, double, dt, *t_p = *dt_p + 1;);
	print("Tensor apply (n_dim=%d size=[%dx%d]):\n", t->nDimension, t->size[0], t->size[1]);

	for(x = 0 ; x < w ; x++)
	{
	   	for(y = 0 ; y < h ; y++)
		{
			dv = THTensor_get2d(t, x, y);
			print(" %g", dv);
		}
		print("\n");
	}
	THTensor_free(t);
	THTensor_free(dt);

	//
	// Tensor of char
	print("\n\n");
	print("Testing Tensor of char\n\n");

	print("Allocating a Tensor of char of size %d ...\n", w);
	THCharTensor *ct = THCharTensor_newWithSize1d(w);
	print("Tensor allocated (n_dim=%d size=[%d])\n", ct->nDimension, ct->size[0]);

	for(x = 0 ; x < w ; x++) THCharTensor_set1d(ct, x, 'a');

	for(x = 0 ; x < w ; x++)
	{
		cv = THCharTensor_get1d(ct, x);
		print(" %c", cv);
	}
	print("\n");

	THCharTensor_free(ct);

	//
	// Tensor of short
	print("\n\n");
	print("Testing Tensor of short\n\n");

	print("Allocating a Tensor of short of size %d ...\n", w);
	THShortTensor *st = THShortTensor_newWithSize1d(w);
	print("Tensor allocated (n_dim=%d size=[%d])\n", st->nDimension, st->size[0]);

	for(x = 0 ; x < w ; x++) THShortTensor_set1d(st, x, 1);

	for(x = 0 ; x < w ; x++)
	{
		sv = THShortTensor_get1d(st, x);
		print(" %d", sv);
	}
	print("\n");

	THShortTensor_free(st);

	//
	// Tensor of int
	print("\n\n");
	print("Testing Tensor of int\n\n");

	print("Allocating a Tensor of int of size %d ...\n", w);
	THIntTensor *it = THIntTensor_newWithSize1d(w);
	print("Tensor allocated (n_dim=%d size=[%d])\n", it->nDimension, it->size[0]);

	for(x = 0 ; x < w ; x++) THIntTensor_set1d(it, x, 1);

	for(x = 0 ; x < w ; x++)
	{
		iv = THIntTensor_get1d(it, x);
		print(" %d", iv);
	}
	print("\n");

	THIntTensor_free(it);

	//
	// Tensor of long
	print("\n\n");
	print("Testing Tensor of long\n\n");

	print("Allocating a Tensor of long of size %d ...\n", w);
	THLongTensor *lt = THLongTensor_newWithSize1d(w);
	print("Tensor allocated (n_dim=%d size=[%d])\n", lt->nDimension, lt->size[0]);

	for(x = 0 ; x < w ; x++) THLongTensor_set1d(lt, x, 1);

	for(x = 0 ; x < w ; x++)
	{
		lv = THLongTensor_get1d(lt, x);
		print(" %ld", lv);
	}
	print("\n");

	THLongTensor_free(lt);

	//
	// Tensor of float
	print("\n\n");
	print("Testing Tensor of float\n\n");

	print("Allocating a Tensor of float of size %d ...\n", w);
	THFloatTensor *ft = THFloatTensor_newWithSize1d(w);
	print("Tensor allocated (n_dim=%d size=[%d])\n", ft->nDimension, ft->size[0]);

	for(x = 0 ; x < w ; x++) THFloatTensor_set1d(ft, x, 1);

	for(x = 0 ; x < w ; x++)
	{
		fv = THFloatTensor_get1d(ft, x);
		print(" %f", fv);
	}
	print("\n");

	THFloatTensor_free(ft);

	return 0;
}

