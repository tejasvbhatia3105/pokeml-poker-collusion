#include "cards.cpp"
// Six known two-card hands, five visible-board slots (-1 padding), active mask.
// Folded players' cards remain known dead cards. This is retrospective equity,
// not an ordinary player's information set or a side-pot strategy solver.
extern "C" void poker_multiway(const int8_t* states,const uint8_t* masks,int n,int sims,float* out){
 for(int i=0;i<n;i++){
  const int8_t*c=states+17*i;uint64_t used=0;int nb=0;
  for(int k=0;k<17;k++)if(c[k]>=0){used|=1ULL<<c[k];if(k>=12)nb++;}
  uint64_t seed=used^0x9e3779b97f4a7c15ULL;if(!seed)seed=1;
  double win[6]={};int trials=0;int8_t hands[6][7];
  for(int p=0;p<6;p++){hands[p][0]=c[2*p];hands[p][1]=c[2*p+1];for(int j=0;j<nb;j++)hands[p][2+j]=c[12+j];}
  auto score=[&](){uint32_t r[6]={},best=0;int ties=0;
   for(int p=0;p<6;p++)if(masks[i]&(1<<p)){r[p]=poker_rank(hands[p],7);best=std::max(best,r[p]);}
   for(int p=0;p<6;p++)if((masks[i]&(1<<p))&&r[p]==best)ties++;
   for(int p=0;p<6;p++)if((masks[i]&(1<<p))&&r[p]==best)win[p]+=1.0/ties;trials++;
  };
  if(nb==5)score();
  else if(nb==4){for(int v=0;v<52;v++)if(!(used&(1ULL<<v))){for(int p=0;p<6;p++)hands[p][6]=v;score();}}
  else for(int t=0;t<sims;t++){uint64_t u=used;for(int j=nb;j<5;j++){int v=draw(seed,u);for(int p=0;p<6;p++)hands[p][2+j]=v;}score();}
  for(int p=0;p<6;p++)out[6*i+p]=float(win[p]/trials);
 }
}
