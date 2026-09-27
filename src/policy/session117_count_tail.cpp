// Exact Poisson-binomial tail and analytic logit gradient, cap five.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>
extern "C" int count_tail(const double*prob,const uint8_t*mask,const int*minimum,int batch,int length,double*out,double*gradient){
 for(int b=0;b<batch;++b){
  int k=minimum[b];if(k<0||k>5)return 1;
  std::vector<double> pre((length+1)*6,0.),post((length+1)*6,0.);pre[0]=1;post[length*6]=1;
  auto step=[](const double*a,double*z,double p){z[0]=a[0]*(1-p);for(int j=1;j<5;++j)z[j]=a[j]*(1-p)+a[j-1]*p;z[5]=a[5]+a[4]*p;};
  for(int i=0;i<length;++i){int ix=b*length+i;double p=mask[ix]?1-prob[3*ix]:0;step(&pre[i*6],&pre[(i+1)*6],p);}
  for(int i=length-1;i>=0;--i){int ix=b*length+i;double p=mask[ix]?1-prob[3*ix]:0;step(&post[(i+1)*6],&post[i*6],p);}
  double tail=0;for(int j=k;j<6;++j)tail+=pre[length*6+j];out[b]=k?std::log(std::max(tail,1e-15)):0;
  for(int i=0;i<length;++i){int ix=b*length+i;gradient[2*ix]=gradient[2*ix+1]=0;
   if(!mask[ix]||k==0||tail<=1e-15)continue;
   double boundary=0;for(int j=0;j<k;++j)boundary+=pre[i*6+j]*post[(i+1)*6+k-1-j];
   double scale=boundary/tail*prob[3*ix];gradient[2*ix]=scale*prob[3*ix+1];gradient[2*ix+1]=scale*prob[3*ix+2];
  }
 }
 return 0;
}
