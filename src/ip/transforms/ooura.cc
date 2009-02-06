
namespace Torch
{

unsigned int nexthigher(unsigned int k) 
{
	if (k == 0) return 1;
	k--;	           
	for (int i=1; i<sizeof(unsigned int)*8; i<<=1)
		k = k | k >> i;
	return k+1;
}

}

