// Batch implementation of the verified101 betting state; no model or labels.
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace {
constexpr int W=45;
double remaining(const double*s,int j){return s[j]-s[6+j];}
double bet(const double*s){return *std::max_element(s+12,s+18);}
double call(const double*s,int j){return std::min(remaining(s,j),std::max(0.,bet(s)-s[12+j]));}
void clean(double*s){
 int alive=0,eligible=0,last=-1;
 for(int j=0;j<6;++j){alive+=int(s[18+j]);bool ok=s[18+j]&&remaining(s,j)>0;s[24+j]=s[24+j]&&ok;if(ok){++eligible;last=j;}}
 if(alive<=1||(eligible==1&&s[12+last]>=bet(s)))std::fill(s+24,s+30,0.);
}
int next_actor(double*s){clean(s);for(int k=1;k<=6;++k){int j=(int(s[38])+k)%6;if(s[24+j])return j;}return -1;}
int next_decision(double*s){
 int j=next_actor(s);
 while(j<0){
  int alive=0,eligible=0;for(int k=0;k<6;++k){alive+=int(s[18+k]);eligible+=s[18+k]&&remaining(s,k)>0;}
  if(s[43]==3||alive<=1||eligible<=1)return -1;
  s[43]+=1;std::fill(s+12,s+18,0.);s[36]=s[44];
  for(int k=0;k<6;++k)s[24+k]=s[30+k]=s[18+k]&&remaining(s,k)>0;
  s[38]=s[37];s[41]=-1;s[42]=0;j=next_actor(s);
 }
 return j;
}
int apply(double*s,int j,int k,double amount){
 double c=call(s,j),rem=remaining(s,j),oldbet=bet(s);int error=0;
 if(!s[24+j])error|=1;
 if(amount<0||amount>rem)error|=2;
 if(k==0&&(amount!=0||c<=0))error|=4;
 if(k==1&&(amount!=0||c>0))error|=8;
 if(k==2&&amount!=c)error|=16;
 if(k==3){if(amount<=c)error|=32;if(!s[30+j])error|=64;if(s[12+j]+amount-oldbet<s[36]&&amount!=rem)error|=128;}
 s[24+j]=s[30+j]=0;if(k==0)s[18+j]=0;s[6+j]+=amount;s[12+j]+=amount;
 if(k==3&&s[12+j]>oldbet){
  double increment=s[12+j]-oldbet;
  if(increment>=s[36]){
   s[36]=increment;for(int h=0;h<6;++h)s[24+h]=s[30+h]=s[18+h]&&remaining(s,h)>0;
   s[24+j]=s[30+j]=0;
  }else{double b=bet(s);for(int h=0;h<6;++h)s[24+h]=s[24+h]||(s[18+h]&&remaining(s,h)>0&&s[12+h]<b);}
  s[41]=j;s[42]+=1;
 }
 s[38]=j;s[40]=k;s[39]+=1;clean(s);
 if(k==3){double b=bet(s);for(int h=0;h<6;++h)s[30+h]=s[30+h]||(s[18+h]&&remaining(s,h)>0&&b-s[12+h]>=s[36]);}
 return error;
}
}
extern "C" {
int state_width(){return W;}
void decisions(double*states,int n,int*actors){for(int i=0;i<n;++i)actors[i]=next_decision(states+W*i);}
void dynamic_features(const double*states,int n,const int*actors,float*x,uint8_t*legal,double*params){
 for(int i=0;i<n;++i){int j=actors[i];if(j<0)continue;const double*s=states+W*i;double c=call(s,j),rem=remaining(s,j),pot=0;int active=0;bool competitor=false;
  for(int h=0;h<6;++h){pot+=s[6+h];active+=int(s[18+h]);competitor|=h!=j&&s[18+h]&&remaining(s,h)>0;}
  double fields[13]={s[43],s[39],double(active),s[40],s[42],s[44],pot/s[44],c/s[44],rem/s[44],c/std::max(pot+c,1.),c/std::max(rem,1.),double((j-int(s[37])+6)%6),double(s[41]==j)};
  for(int h=0;h<13;++h)x[13*i+h]=float(fields[h]);
  legal[4*i]=c>0;legal[4*i+1]=c==0;legal[4*i+2]=c>0;legal[4*i+3]=rem>c&&s[30+j]&&competitor;
  params[4*i]=c;params[4*i+1]=rem;params[4*i+2]=pot;params[4*i+3]=s[36];
 }
}
int apply_batch(double*states,int n,const int*actors,const int*kinds,const double*amounts){
 int error=0;for(int i=0;i<n;++i)if(actors[i]>=0)error|=apply(states+W*i,actors[i],kinds[i],amounts[i]);return error;
}
int payouts(const double*states,int n,const uint32_t*ranks,double*out){
 for(int i=0;i<n;++i){const double*s=states+W*i;const uint32_t*r=ranks+6*i;double*net=out+6*i;double levels[6];std::copy(s+6,s+12,levels);std::sort(levels,levels+6);double previous=0;
  for(int k=0;k<6;++k)net[k]=0;
  for(double level:levels){if(level<=previous)continue;int present=0,winners=0;uint32_t high=0;bool eligible=false;
   for(int j=0;j<6;++j)if(s[6+j]>=level){++present;if(s[18+j]){eligible=true;high=std::max(high,r[j]);}}
   if(!eligible)return 1;
   for(int j=0;j<6;++j)if(s[6+j]>=level&&s[18+j]&&r[j]==high)++winners;
   double share=(level-previous)*present/winners;
   for(int j=0;j<6;++j)if(s[6+j]>=level&&s[18+j]&&r[j]==high)net[j]+=share;
   previous=level;
  }
  for(int j=0;j<6;++j)net[j]-=s[6+j];
 }
 return 0;
}
}
