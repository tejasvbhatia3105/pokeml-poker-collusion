#include "cards.cpp"

// Ordinary heads-up equity against unknown opposing cards. No opponent's actual
// cards or unrevealed board enter this estimate. River enumerates every legal
// opposing holding; earlier streets use the existing sampling implementation.
extern "C" void poker_precise_equity(const int8_t *states,int n,int simulations,float *out){
 for(int i=0;i<n;i++){
  const int8_t *c=states+7*i;int nb=0;
  for(int j=2;j<7;j++)if(c[j]>=0)nb++;
  if(nb<5){float buffer[3];poker_features(c,1,simulations,buffer);out[i]=buffer[2];continue;}
  uint64_t used=0;for(int j=0;j<7;j++)used|=1ULL<<c[j];
  uint32_t own=poker_rank(c,7);int8_t opp[7];for(int j=2;j<7;j++)opp[j]=c[j];
  double wins=0;int trials=0;
  for(int a=0;a<52;a++)if(!(used&(1ULL<<a)))
   for(int b=a+1;b<52;b++)if(!(used&(1ULL<<b))){
    opp[0]=a;opp[1]=b;uint32_t other=poker_rank(opp,7);
    wins+=own>other?1.:(own==other?.5:0.);trials++;
   }
  out[i]=float(wins/trials);
 }
}
