#include "torch5spro.h"

using namespace Torch;

int main()
{
  //---

  print("Building an array of Tensors of different types and dimensions ...\n");

  Tensor **tarray = (Tensor **) THAlloc(6*sizeof(Tensor));
  for(int i = 0 ; i < 6 ; i++) tarray[i] = NULL;

  int n = 0;
  print("new DoubleTensor ...\n");
  DoubleTensor *dt = new DoubleTensor(3, 5);
  dt->fill(1);
  tarray[n++] = dt;

  print("new FloatTensor ...\n");
  FloatTensor *ft = new FloatTensor(5);
  ft->fill(0);
  tarray[n++] = ft;

  print("new LongTensor ...\n");
  LongTensor *lt = new LongTensor(3, 5);
  lt->fill(0);
  tarray[n++] = lt;

  print("new IntTensor ...\n");
  IntTensor *it = new IntTensor(5);
  it->fill(0);
  tarray[n++] = it;

  print("new ShortTensor ...\n");
  ShortTensor *st = new ShortTensor(10, 10);
  st->fill(0);
  tarray[n++] = st;

  print("new CharTensor ...\n");
  CharTensor *ct = new CharTensor(5);
  ct->fill('.');
  Tprint(ct);
  ct->set(0, 'T');
  ct->set(1, 'O');
  ct->set(2, 'R');
  ct->set(3, 'C');
  ct->set(4, 'H');
  tarray[n++] = ct;

  for(int i = 0 ; i < 6 ; i++)
    {
      print("Tensor of %s:\n", str_TensorTypeName[tarray[i]->getDatatype()]);
      print("   dimension = %d\n", tarray[i]->nDimension());
      for(int j = 0 ; j < tarray[i]->nDimension() ; j++)
	print("   size of dimension %d = %d\n", j, tarray[i]->size(j));
      tarray[i]->sprint("T%d", i);

      print("\n");
    }

  print("Copying a CharTensor to a FloatTensor ...\n");
  ft->copy(ct);
  Tprint(ft);

  print("Copying a DoubleTensor to a LongTensor ...\n");
  lt->copy(dt);
  Tprint(lt);

  it->print("it before");
  (*it)(0) = 1;
  (*it)(4) = 2;
  it->print("it after");

  st->print("st before");
  for(long i = 0 ; i < 10 ; i++)
    (*st)(i,i) = i;
  st->print("st after");

  delete dt;
  delete ft;
  delete lt;
  delete it;
  delete st;
  delete ct;
  THFree(tarray);



  //---



  tarray = (Tensor **) THAlloc(5*sizeof(Tensor));
  for(int i = 0 ; i < 5 ; i++) tarray[i] = NULL;
  ct = new CharTensor(5);
  ct->set(0, 'T');
  ct->set(1, 'O');
  ct->set(2, 'R');
  ct->set(3, 'C');
  ct->set(4, 'H');
  Tprint(ct);

  n = 0;
  tarray[n++] = new DoubleTensor(5);
  tarray[n++] = new FloatTensor(5);
  tarray[n++] = new LongTensor(5);
  tarray[n++] = new IntTensor(5);
  tarray[n++] = new ShortTensor(5);
  for(int i = 0 ; i < n ; i++)
    {
      tarray[i]->copy(ct);
      tarray[i]->print();
    }

  for(int i = 0 ; i < n ; i++) delete tarray[i];
  delete ct;
  THFree(tarray);



  //---

  print("new IntTensor ...\n");
  IntTensor *it1 = new IntTensor(5, 10);
  IntTensor *it2 = new IntTensor();

  n = 0;
  for(long y = 0 ; y < 5 ; y++)
    for(long x = 0 ; x < 10 ; x++)
      it1->set(y, x, n++);

  it1->print("it1");
  it2->print("it2");
  print("Transpose tensor it1 to it2...\n");
  it2->transpose(it1, 1, 0);
  it2->print("it2");

  delete it2;
  delete it1;

  print("new IntTensor ...\n");
  it1 = new IntTensor(5, 10, 2);
  it2 = new IntTensor();

  n = 0;
  for(long y = 0 ; y < 5 ; y++)
    for(long x = 0 ; x < 10 ; x++)
      {
	for(long z = 0 ; z < 2 ; z++)
	  it1->set(y, x, z, n);
	n++;
      }

  it1->print("it1");
  it2->print("it2");
  print("Transpose tensor it1 to it2...\n");
  it2->transpose(it1, 1, 0);
  it2->print("it2");

  IntTensor *it3 = new IntTensor();
  it3->narrow(it1, 0, 2, 1);
  it3->print("it3");

  IntTensor *it4 = new IntTensor();
  it4->narrow(it1, 1, 8, 1);
  it4->print("it4");

  delete it4;
  delete it3;

  IntTensor *it2copy = new IntTensor(10, 5, 2);
  it2copy->copy(it2);
  it2copy->print("it2copy");

  IntTensor *it5 = new IntTensor();
  it5->select(it1, 0, 1);
  it5->print("it5");
  it5->fill(0);
  it2->print("it2 modified");
  delete it5;

  it2->copy(it2copy);
  it2->print("it2 recopied");

  it5 = new IntTensor();
  it5->select(it1, 0, 1);
  IntTensor *it6 = new IntTensor();
  it6->select(it5, 0, 1);
  it6->fill(0);
  it2->print("it2 modified");
  delete it5;
  delete it6;

  delete it2copy;
  delete it2;
  delete it1;


  //---

  long h = 10;
  long w = 20;
  long x, y;

  print("Allocating a grayscale image as 2D Tensor of size %dx%d ...\n", h, w);
  ShortTensor *grayimage = new ShortTensor(h, w);
  print("Tensor allocated (n_dim=%d size=[%dx%d])\n", grayimage->nDimension(), grayimage->size(0), grayimage->size(1));

  print("Setting to 1.0 ...\n");
  for(y = 0 ; y < h ; y++)
    for(x = 0 ; x < w ; x++)
      grayimage->set(y, x, 1);
  print("done.\n");

  print("Filling to 0.0 ...\n");
  grayimage->fill(0);
  print("done.\n");

  print("Tensor (n_dim=%d size=[%dx%d]):\n", grayimage->nDimension(), grayimage->size(0), grayimage->size(1));
  grayimage->print("grayimage");

  long sub_x, sub_y, sub_w, sub_h;

  sub_y = 4;
  sub_h = 2;
  sub_x = 2;
  sub_w = 6;

  ShortTensor *t_ = new ShortTensor();
  t_->narrow(grayimage, 0, sub_y, sub_h);

  print("subTensor (n_dim=%d size=[%dx%d]):\n", t_->nDimension(), t_->size(0), t_->size(1));
  t_->print("subtensor");

  ShortTensor *t__ = new ShortTensor();
  t__->narrow(t_, 1, sub_x, sub_w);

  print("subTensor (n_dim=%d size=[%dx%d]):\n", t__->nDimension(), t__->size(0), t__->size(1));
  t__->print("subtensor");

  t__->fill(1);

  print("Tensor (n_dim=%d size=[%dx%d]):\n", grayimage->nDimension(), grayimage->size(0), grayimage->size(1));
  grayimage->print("grayimage");

  delete t_;
  delete t__;
  delete grayimage;

  h /= 2;
  w /= 2;

  print("Allocating a color image as a 3D Tensor of size %dx%dx3 ...\n", h, w);
  ShortTensor *colorimage = new ShortTensor(h, w, 3);
  print("Tensor allocated (n_dim=%d size=[%dx%dx3])\n", colorimage->nDimension(), colorimage->size(0), colorimage->size(1), colorimage->size(2));

  print("Filling to 0.0 ...\n");
  colorimage->fill(0);
  print("done.\n");

  print("Tensor (n_dim=%d size=[%dx%dx3]):\n", colorimage->nDimension(), colorimage->size(0), colorimage->size(1));
  colorimage->print("colorimage");

  sub_y = 1;
  sub_h = 3;
  sub_x = 3;
  sub_w = 5;

  t_ = new ShortTensor();
  t_->narrow(colorimage, 0, sub_y, sub_h);
  t__ = new ShortTensor();
  t__->narrow(t_, 1, sub_x, sub_w);
  t__->fill(1);
  print("Setting to 2.0 ...\n");
  for(y = 0 ; y < sub_h ; y++)
    for(x = 0 ; x < sub_w ; x++)
      {
	t__->set(y, x, 1, 2);
	t__->set(y, x, 2, 3);
      }
  print("done.\n");

  print("Tensor (n_dim=%d size=[%dx%dx3]):\n", colorimage->nDimension(), colorimage->size(0), colorimage->size(1));
  colorimage->print("colorimage");

  ShortTensor *r_ = new ShortTensor();
  r_->select(t__, 2, 0);
  print("R color slice Tensor (n_dim=%d size=[%dx%d]):\n", r_->nDimension(), r_->size(0), r_->size(1));
  r_->print("r_");

  ShortTensor *g_ = new ShortTensor();
  g_->select(t__, 2, 1);
  print("G color slice Tensor (n_dim=%d size=[%dx%d]):\n", g_->nDimension(), g_->size(0), g_->size(1));
  g_->print("g_");

  ShortTensor *b_ = new ShortTensor();
  b_->select(t__, 2, 2);
  print("B color slice Tensor (n_dim=%d size=[%dx%d]):\n", b_->nDimension(), b_->size(0), b_->size(1));
  b_->print("b_");

  r_->fill(4);
  g_->fill(5);
  b_->fill(6);

  print("Tensor (n_dim=%d size=[%dx%dx3]):\n", colorimage->nDimension(), colorimage->size(0), colorimage->size(1));
  colorimage->print("colorimage");

  delete r_;
  delete g_;
  delete b_;
  delete t_;
  delete t__;
  delete colorimage;



  //---

  print("Building a video, i.e. a ShortTensor of 4 dimensions (x, y, color, time) ...\n");
  st = new ShortTensor(640, 480, 3, 50); // a 2 seconds color (RGB) video of VGA (640x480) resolution

  print("Filling it black ...\n");
  st->fill(0);

  print("Deleting the video ...\n");
  delete st;


  //---

  print("Building a simple sequence of numbers ...\n");
  FloatTensor *sequence = new FloatTensor(7);

  print("Init sequence ...\n");
  for(long i = 0 ; i < 7 ; i++)
  	sequence->set(i, i+1);

  sequence->print("sequence of numbers");

  FloatTensor *unfold_sequence1 = new FloatTensor();

  unfold_sequence1->unfold(sequence, 0, 2, 1);
  unfold_sequence1->print("unfolded sequence of numbers along dim 0 from 1D to 2D with a step of 1");

  FloatTensor *unfold_sequence2 = new FloatTensor();

  unfold_sequence2->unfold(sequence, 0, 2, 2);
  unfold_sequence2->print("unfolded sequence of numbers along dim 0 from 1D (size 7) to 2D (size 2) with a step of 2");

  print("Free sequences ...\n");
  delete unfold_sequence1;
  delete unfold_sequence2;
  delete sequence;

  print("Building a image sequence...\n");
  int seqi_h = 4;
  int seqi_w = 3;
  FloatTensor *seqimage = new FloatTensor(seqi_h * seqi_w);

  print("Init sequence ...\n");
  for(long i = 0 ; i < seqi_h * seqi_w ; i++)
  	seqimage->set(i, i+1);

  seqimage->sprint("image sequence (%d x %d)", seqi_h, seqi_w);

  FloatTensor *unfold_seqimage = new FloatTensor();

  unfold_seqimage->unfold(sequence, 0, seqi_w, seqi_w);
  unfold_seqimage->sprint("unfolded image sequence along dim 0 from 1D (size %d) to 2D (size %d) with a step of %d", seqi_h * seqi_w, seqi_w, seqi_w);

  print("Free images sequences ...\n");
  delete unfold_seqimage;
  delete seqimage;




  print("Building a sequence of 7 concatenated 3D frames ...\n");
  sequence = new FloatTensor(21);

  print("Init sequence ...\n");
  for(long i = 0 ; i < 21 ; i++)
  	sequence->set(i, i+1);

  sequence->print("sequence of frames");

  unfold_sequence2 = new FloatTensor();
  unfold_sequence2->unfold(sequence, 0, 3, 3);
  unfold_sequence2->print("unfolded concatenated sequence along dim 0 from 1D (size 21) to 2D (size 3) with a step of 3");

  print("Free sequences ...\n");
  delete unfold_sequence2;
  delete sequence;


  print("Building a sequence of 100 concatenated 20D frames ...\n");
  sequence = new FloatTensor(200);

  print("Init sequence ...\n");
  for(long i = 0 ; i < 200 ; i++)
  	sequence->set(i, i+1);

  sequence->print("sequence of frames");

  unfold_sequence2 = new FloatTensor();
  unfold_sequence2->unfold(sequence, 0, 20, 20);
  unfold_sequence2->print("unfolded concatenated sequence along dim 0 from 1D (size 200) to 2D (size 20) with a step of 20");

  print("Free sequences ...\n");
  delete unfold_sequence2;
  delete sequence;

  //---

  print("Building a new sequence, i.e. a FloatTensor of 2 dimensions (t, coefficients) ...\n");
  sequence = new FloatTensor(2000000,20);

  print("Tensor of %s:\n", str_TensorTypeName[sequence->getDatatype()]);
  print("   dimension = %d\n", sequence->nDimension());
  for(int j = 0 ; j < sequence->nDimension() ; j++)
	print("   size of dimension %d = %d\n", j, sequence->size(j));

  print("Init sequence ...\n");
  sequence->fill(0);

  print("Free sequence ...\n");
  delete sequence;

  print("Building an array of sequences with different length ...\n");

  Tensor **sarray = (Tensor **) THAlloc(500*sizeof(Tensor));
  for(int i = 0 ; i < 500 ; i++)
  	sarray[i] = new FloatTensor(4000,20);

  print("Init sequences ...\n");
  for(int i = 0 ; i < 500 ; i++)
  {
  	FloatTensor * t_ = (FloatTensor *) sarray[i];
  	t_->fill(0);
  }

  print("Free sequences ...\n");
  for(int i = 0 ; i < 500 ; i++) delete sarray[i];
  THFree(sarray);



  //---

  print("Building a dataset of multiple classes ...\n");

  // classes
  int n_classes = 2;
  IntTensor **target_one_label = (IntTensor **) THAlloc(n_classes*sizeof(IntTensor));
  IntTensor **target_one_hot_encoding = (IntTensor **) THAlloc(n_classes*sizeof(IntTensor));
  for(int i = 0 ; i < n_classes ; i++)
  {
  	target_one_label[i] = new IntTensor(1);
	target_one_label[i]->fill(i);

  	target_one_hot_encoding[i] = new IntTensor(n_classes);
	target_one_hot_encoding[i]->fill(0);
	target_one_hot_encoding[i]->set(i, 1);
  }
  print("Classes:\n");
  for(int i = 0 ; i < n_classes ; i++)
  {
	target_one_label[i]->print("");
	target_one_hot_encoding[i]->print("");
  }

  THRandom_manualSeed(950305);

  //
  long n_examples_per_class = 5000;
  long n_examples = n_examples_per_class * n_classes;
  long inputsize = 10;

  DoubleTensor **examples = (DoubleTensor **) THAlloc(n_examples*sizeof(DoubleTensor));
  IntTensor **targets = (IntTensor **) THAlloc(n_examples*sizeof(IntTensor));
  DoubleTensor **means = (DoubleTensor **) THAlloc(n_classes*sizeof(DoubleTensor));
  DoubleTensor **variances = (DoubleTensor **) THAlloc(n_classes*sizeof(DoubleTensor));
  long n_examples_ = 0;
  for(int c = 0 ; c < n_classes ; c++)
  {
	means[c] = new DoubleTensor(inputsize);
	means[c]->fill(0);
	variances[c] = new DoubleTensor(inputsize);
	variances[c]->fill(0);

	DoubleTensor m(inputsize);
	m.fill(0);
	DoubleTensor v(inputsize);
	v.fill(0);

  	for(int i = 0 ; i < n_examples_per_class ; i++)
  	{
		examples[n_examples_] = new DoubleTensor(inputsize);
  		for(long j = 0 ; j < inputsize ; j++)
		{
			double random_ = THRandom_uniform(0, 255);
			examples[n_examples_]->set(j,random_);

			m(j) += random_;
			v(j) += random_ * random_;
		}

		double sum_ = THTensor_sum(examples[n_examples_]->t);
		double mean_ = THTensor_mean(examples[n_examples_]->t);
		double min_ = THTensor_min(examples[n_examples_]->t);
		double max_ = THTensor_max(examples[n_examples_]->t);
		double norm_ = THTensor_norm(examples[n_examples_]->t, 2);

		//print("> %g %g %g %g %g\n", sum_, mean_, min_, max_, norm_);

		THTensor_addTensor(means[c]->t, 1, examples[n_examples_]->t);
		THTensor_addcmul(variances[c]->t, 1, examples[n_examples_]->t, examples[n_examples_]->t);

		targets[n_examples_] = target_one_hot_encoding[c];
		n_examples_++;
  	}

	THTensor_div(means[c]->t, (double) n_examples_per_class);
	THTensor_div(variances[c]->t, (double) n_examples_per_class);
	THTensor_addcmul(variances[c]->t, -1, means[c]->t, means[c]->t);

	means[c]->print("mean");
	variances[c]->print("variance");

  	for(long j = 0 ; j < inputsize ; j++)
	{
		m(j) /= (double) n_examples_per_class;
		v(j) /= (double) n_examples_per_class;
		v(j) -= m(j) * m(j);
	}

	m.print("m");
	v.print("v");
  }

  /*
  for(int i = 0 ; i < n_examples ; i++)
  {
  	examples[i]->print("example");
  	targets[i]->print("target");
  }
  */

  for(int i = 0 ; i < n_examples ; i++) delete examples[i];
  for(int i = 0 ; i < n_classes ; i++)
  {
  	delete means[i];
  	delete variances[i];
  	delete target_one_label[i];
  	delete target_one_hot_encoding[i];
  }
  THFree(means);
  THFree(variances);
  THFree(examples);
  THFree(targets);
  THFree(target_one_label);
  THFree(target_one_hot_encoding);

  print("Testing Tensor resizing ...\n");

  print("new empty DoubleTensor ...\n");
  DoubleTensor *resizable_dt = new DoubleTensor();
  resizable_dt->fill(0);
  resizable_dt->print("resizable Tensor");
  print("Sum of elements = %g\n", resizable_dt->sum());

  print("resizing to 3D ...\n");
  resizable_dt->resize(5, 4, 3);
  resizable_dt->fill(3);
  resizable_dt->print("resized Tensor");
  print("Sum of elements = %g\n", resizable_dt->sum());

  print("resizing to 2D ...\n");
  resizable_dt->resize(5, 4);
  resizable_dt->fill(2);
  resizable_dt->print("resized Tensor");
  print("Sum of elements = %g\n", resizable_dt->sum());

  print("resizing to 1D ...\n");
  resizable_dt->resize(5);
  resizable_dt->fill(1);
  resizable_dt->print("resized Tensor");
  print("Sum of elements = %g\n", resizable_dt->sum());

  delete resizable_dt;

  return 0;
}

