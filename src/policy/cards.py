import ctypes,itertools
from pathlib import Path
import numpy as np
LIB=ctypes.CDLL(str(Path('artifacts/policy/cards.dylib').resolve()))
LIB.poker_rank.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int];LIB.poker_rank.restype=ctypes.c_uint32
LIB.poker_features.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
RANK='23456789TJQKA';SUIT='cdhs';CARD={r+s:4*i+j for i,r in enumerate(RANK) for j,s in enumerate(SUIT)}
def rank(cards):
 a=np.asarray(cards,dtype=np.int8);return LIB.poker_rank(a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(a))
def features(states,simulations=96):
 a=np.ascontiguousarray(states,dtype=np.int8);o=np.empty((len(a),3),dtype=np.float32)
 LIB.poker_features(a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(a),simulations,o.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
 return o

def preflop_table():
 path=Path('artifacts/policy/preflop_equity.npy')
 if path.exists():return np.load(path)
 states=[];keys=[]
 for hi in range(13):
  for lo in range(hi+1):
   for suited in ([False] if hi==lo else [False,True]):
    states.append([4*hi,4*lo+(0 if suited else 1),-1,-1,-1,-1,-1]);keys.append((hi,lo,int(suited)))
 eq=features(states,12000)[:,2];table=np.zeros((13,13,2),np.float32)
 for (hi,lo,suited),v in zip(keys,eq):table[hi,lo,suited]=table[lo,hi,suited]=v
 np.save(path,table);return table
