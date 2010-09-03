
/* Addons */
int THIntTensor_sum(THIntTensor *tensor)
{
  int sum = 0;
  TH_TENSOR_APPLY(int, tensor, sum += *tensor_p;);
  return sum;
}

char THCharTensor_sum(THCharTensor *tensor)
{
  char sum = 0;
  TH_TENSOR_APPLY(char, tensor, sum += *tensor_p;);
  return sum;
}

short int THShortTensor_sum(THShortTensor *tensor)
{
  short int sum = 0;
  TH_TENSOR_APPLY(short int, tensor, sum += *tensor_p;);
  return sum;
}

long int THLongTensor_sum(THLongTensor *tensor)
{
  long int sum = 0;
  TH_TENSOR_APPLY(long int, tensor, sum += *tensor_p;);
  return sum;
}

float THFloatTensor_sum(THFloatTensor *tensor)
{
  float sum = 0;
  TH_TENSOR_APPLY(float, tensor, sum += *tensor_p;);
  return sum;
}

void THFloatTensor_mul(THFloatTensor *tensor, float value)
{
  TH_TENSOR_APPLY(float, tensor, *tensor_p *= value;);
}
/* End of addons */

