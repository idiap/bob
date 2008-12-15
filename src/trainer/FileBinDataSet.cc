#include "FileBinDataSet.h"

namespace Torch
{
   
FileBinDataSet::FileBinDataSet()
    : DataSet(Tensor::Short, Tensor::Short)
{
}

void FileBinDataSet::setData(char *filename, Tensor::Type type_inp, Tensor::Type type_tar , int width, int height)
{
        int n_files;
        char binfiles[1000][1000];
        print("type of input %d, %d\n",type_inp,type_tar);
        File file;
        File binfile;

       // tensor_return = new Tensor();
        print("NOTE ......................... .\n...................\n...............This is now only for ShortTensors....\n");
        if (file.open(filename, "r") == false)
        {
            print("ERROR: loading bindata list [%s]!\n", filename);
            //return 1;
        }
        file.scanf("%d",&n_files);
        print("Number of files in %s : %d\n",filename,n_files);

        IntTensor *t=new IntTensor(n_files);
        // read the target and file names
        int ta;

        for (int i=0;i<n_files;i++)
        {
            file.scanf("%d",&ta);
            file.scanf("%s",binfiles[i]);
            (*t)(i)=ta;
            print(" %d , %s\n",(*t)(i),binfiles[i]);
        }

        file.close();
        // Now you have the target and the filenames
        // Will have to read the number of examples
        n_examples = 0;
        int n_size;
        int n_samples;
        for (int i=0;i<n_files;i++)
        {

            if (binfile.open(binfiles[i], "r") == false)
            {
                print("ERROR: loading bindata [%s]!\n", binfiles[i]);
                //return 1;
            }
            binfile.read(&n_samples, sizeof(int), 1);
            binfile.read(&n_size, sizeof(int), 1);

            n_examples += n_samples;

            binfile.close();

            print("N_samples = %d, N_size = %d\n",n_samples,n_size);
        }

        if (width*height != n_size)
            print("The input size does not match with width*height.......\n");
        else
        {
            // allocate memory for examples and targets
            /// Can be made Adaptive to users request !........
            short_example = new ShortTensor(height,width,n_examples);
            examples = short_example;
            short_target = new ShortTensor(n_examples);
            target = short_target;
            current_example = new ShortTensor();
            short_currentT = new ShortTensor(1);
            current_target = short_currentT;
            //current_target = new ShortTensor(1);

            int n_count=0;
            // now load the files
            float *bdata = new float [n_size];
            for (int i=0;i<n_files;i++)
            {
                if (binfile.open(binfiles[i], "r") == false)
                {
                    print("ERROR: loading bindata [%s]!\n", binfiles[i]);
                    //return 1;
                }
                binfile.read(&n_samples, sizeof(int), 1);
                binfile.read(&n_size, sizeof(int), 1);


                for (int j=0;j<n_samples;j++)
                {
                    binfile.read(bdata, sizeof(float), n_size);
                    for (int m=0;m<width;m++)
                        for (int n=0;n<height;n++)
                        {
                            short_example->set(n, m, n_count, (short)(bdata[n*width+m] * 255.0f + 0.5f));

                        }
                    short_target->set(n_count,(*t)(i));
                    n_count++;
                }

                binfile.close();

            }

        }
        //

        delete t;
}

Tensor* FileBinDataSet::getExample(long k)
{
	current_example = examples->select(2,k);

	return current_example;
}

Tensor& FileBinDataSet::operator()(long k)
{
	current_example = examples->select(2,k);

	return *current_example;
}

Tensor* FileBinDataSet::getTarget(long k)
{
	short m = short_target->get(k);

	short_currentT->set(0,m);

	return current_target;
}

FileBinDataSet::~FileBinDataSet()
{
}

}

