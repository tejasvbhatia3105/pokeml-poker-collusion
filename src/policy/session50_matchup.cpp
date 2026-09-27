#include "cards.cpp"
// Two actual holdings and only the visible board. Other players' holdings are
// marginalized rather than conditioned on; this is not a strategy/EV solver.
extern "C" void poker_pair_matchup(const int8_t* states,int n,int sims,float* out){
 for(int i=0;i<n;i++){
  const int8_t*c=states+9*i;uint64_t used=0;int nb=0;
  for(int k=0;k<9;k++)if(c[k]>=0){used|=1ULL<<c[k];if(k>=4)nb++;}
  uint64_t seed=used^0x9e3779b97f4a7c15ULL;int8_t a[7],b[7];
  a[0]=c[0];a[1]=c[1];b[0]=c[2];b[1]=c[3];
  for(int j=0;j<nb;j++)a[j+2]=b[j+2]=c[j+4];
  uint32_t ra=nb>=3?poker_rank(a,nb+2):0,rb=nb>=3?poker_rank(b,nb+2):0;
  double w=0,tie=0,ai=0,bi=0;int trials=0;
  auto score=[&](){auto x=poker_rank(a,7),y=poker_rank(b,7);w+=(x>y);tie+=(x==y);ai+=(x>ra);bi+=(y>rb);trials++;};
  if(nb==5)score();
  else if(nb==4){for(int v=0;v<52;v++)if(!(used&(1ULL<<v))){a[6]=b[6]=v;score();}}
  else if(nb==3){for(int v=0;v<52;v++)if(!(used&(1ULL<<v)))for(int u=v+1;u<52;u++)if(!(used&(1ULL<<u))){a[5]=b[5]=v;a[6]=b[6]=u;score();}}
  else for(int s=0;s<sims;s++){uint64_t u=used;for(int j=nb;j<5;j++)a[j+2]=b[j+2]=draw(seed,u);score();}
  float *p=out+7*i;p[0]=(w+.5*tie)/trials;p[1]=w/trials;p[2]=tie/trials;p[3]=nb>=3?int(ra>rb)-int(ra<rb):-2;p[4]=nb>=3?ai/trials:-2;p[5]=nb>=3?bi/trials:-2;p[6]=nb>=3?(double(ra)-double(rb))/double(1ULL<<24):-2;
 }
}
