#include <cstdint>
#include <algorithm>
#include <cmath>
static int straight(uint32_t mask){
 if(mask&(1u<<14))mask|=1u<<1;
 for(int h=14;h>=5;--h)if((mask&(31u<<(h-4)))==(31u<<(h-4)))return h;
 return 0;
}
static uint32_t pack(int cat,const int *r,int n){uint32_t v=uint32_t(cat)<<24;for(int i=0;i<n;++i)v|=uint32_t(r[i])<<(20-4*i);return v;}
extern "C" uint32_t poker_rank(const int8_t *cards,int n){
 int count[15]={},suits[4]={};uint32_t masks[4]={},mask=0;
 for(int i=0;i<n;i++){int r=cards[i]/4+2,s=cards[i]%4;++count[r];++suits[s];mask|=1u<<r;masks[s]|=1u<<r;}
 int fs=-1;for(int s=0;s<4;s++)if(suits[s]>=5){fs=s;int h=straight(masks[s]);if(h){int rr[]={h};return pack(8,rr,1);}}
 int qu=0,tr=0,pa=0;for(int r=14;r>=2;r--){if(count[r]==4&&!qu)qu=r;if(count[r]>=3&&!tr)tr=r;}
 int rr[5]={};if(qu){rr[0]=qu;for(int r=14;r>=2;r--)if(r!=qu&&count[r]){rr[1]=r;break;}return pack(7,rr,2);}
 if(tr){for(int r=14;r>=2;r--)if(r!=tr&&count[r]>=2){pa=r;break;}if(pa){rr[0]=tr;rr[1]=pa;return pack(6,rr,2);}}
 if(fs>=0){int k=0;for(int r=14;r>=2&&k<5;r--)if(masks[fs]&(1u<<r))rr[k++]=r;return pack(5,rr,5);}
 int h=straight(mask);if(h){rr[0]=h;return pack(4,rr,1);}
 if(tr){rr[0]=tr;int k=1;for(int r=14;r>=2&&k<3;r--)if(r!=tr&&count[r])rr[k++]=r;return pack(3,rr,3);}
 int pairs[3]={},np=0;for(int r=14;r>=2;r--)if(count[r]>=2&&np<3)pairs[np++]=r;
 if(np>=2){rr[0]=pairs[0];rr[1]=pairs[1];for(int r=14;r>=2;r--)if(r!=rr[0]&&r!=rr[1]&&count[r]){rr[2]=r;break;}return pack(2,rr,3);}
 if(np){rr[0]=pairs[0];int k=1;for(int r=14;r>=2&&k<4;r--)if(r!=rr[0]&&count[r])rr[k++]=r;return pack(1,rr,4);}
 int k=0;for(int r=14;r>=2&&k<5;r--)if(count[r])rr[k++]=r;return pack(0,rr,5);
}
static uint64_t rngnext(uint64_t &x){x^=x>>12;x^=x<<25;x^=x>>27;return x*2685821657736338717ULL;}
static int draw(uint64_t &seed,uint64_t &used){int c;do{c=rngnext(seed)%52;}while(used&(1ULL<<c));used|=1ULL<<c;return c;}
extern "C" void poker_features(const int8_t *states,int n,int simulations,float *out){
 // Each row has 2 hole cards and up to 5 visible board cards, padded by -1.
 for(int i=0;i<n;i++){
  const int8_t *c=states+7*i;int nb=0;for(int j=2;j<7;j++)if(c[j]>=0)nb++;
  uint64_t used=0,seed=0x9e3779b97f4a7c15ULL;for(int j=0;j<2+nb;j++){used|=1ULL<<c[j];seed=seed*1099511628211ULL+uint64_t(c[j]+1);}if(!seed)seed=1;
  uint32_t rank=nb>=3?poker_rank(c,nb+2):0;float wins=0;
  for(int t=0;t<simulations;t++){
   uint64_t u=used;int8_t own[7],opp[7];own[0]=c[0];own[1]=c[1];opp[0]=draw(seed,u);opp[1]=draw(seed,u);
   for(int j=0;j<5;j++){int v=j<nb?c[j+2]:draw(seed,u);own[j+2]=opp[j+2]=v;}
   uint32_t a=poker_rank(own,7),b=poker_rank(opp,7);wins+=a>b?1.f:(a==b?.5f:0.f);
  }
  out[3*i]=float(rank>>24);out[3*i+1]=float(rank&0xFFFFFF)/float(1<<24);out[3*i+2]=wins/simulations;
 }
}
extern "C" void poker_pair_equity(const int8_t *states,int n,int simulations,float *out){
 // Rows: two own cards, two partner cards, then visible board padded with -1.
 // Used-card seed makes swapping the two players exactly complementary.
 for(int i=0;i<n;i++){
  const int8_t *c=states+9*i;int nb=0;for(int j=4;j<9;j++)if(c[j]>=0)nb++;
  uint64_t used=0;for(int j=0;j<4+nb;j++)used|=1ULL<<c[j];uint64_t seed=used^0x9e3779b97f4a7c15ULL;if(!seed)seed=1;
  int8_t a[7],b[7];a[0]=c[0];a[1]=c[1];b[0]=c[2];b[1]=c[3];for(int j=0;j<nb;j++)a[2+j]=b[2+j]=c[4+j];double wins=0;int trials=0;
  if(nb==5){uint32_t ar=poker_rank(a,7),br=poker_rank(b,7);out[i]=ar>br?1.f:(ar==br?.5f:0.f);continue;}
  if(nb==4){for(int v=0;v<52;v++)if(!(used&(1ULL<<v))){a[6]=b[6]=v;uint32_t ar=poker_rank(a,7),br=poker_rank(b,7);wins+=ar>br?1.:(ar==br?.5:0.);trials++;}}
  else{for(int k=0;k<simulations;k++){uint64_t u=used;for(int j=nb;j<5;j++)a[2+j]=b[2+j]=draw(seed,u);uint32_t ar=poker_rank(a,7),br=poker_rank(b,7);wins+=ar>br?1.:(ar==br?.5:0.);trials++;}}
  out[i]=float(wins/trials);
 }
}
