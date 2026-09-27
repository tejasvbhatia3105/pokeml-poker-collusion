import ctypes,numpy as np
from cards import LIB
LIB.poker_pair_equity.argtypes=[ctypes.POINTER(ctypes.c_int8),ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_float)]
def pair_equity(states,simulations=128):
 a=np.ascontiguousarray(states,dtype=np.int8);out=np.empty(len(a),np.float32)
 assert a.ndim==2 and a.shape[1]==9
 LIB.poker_pair_equity(a.ctypes.data_as(ctypes.POINTER(ctypes.c_int8)),len(a),simulations,out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
 return out
