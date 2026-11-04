"""Read numeric-only NumPy caches; reject arbitrary pickle globals."""
import collections
import pickle
import pickletools
import numpy as np

def dtype_check(dtype):
    dtype=np.dtype(dtype)
    if dtype.hasobject or dtype.kind not in 'biufc':raise ValueError('nonnumeric dtype')
    return dtype

def numeric_array(buffer,dtype,shape,order):
    dtype=dtype_check(dtype)
    if order not in ('C','F') or any(type(n) is not int or n<0 for n in shape):raise ValueError('array layout')
    return np.frombuffer(buffer,dtype=dtype).reshape(shape,order=order)

def numeric_scalar(dtype,raw):
    dtype=dtype_check(dtype)
    if len(raw)!=dtype.itemsize:raise ValueError('scalar length')
    return np.frombuffer(raw,dtype=dtype,count=1)[0]

class NumericReader(pickle.Unpickler):
    def find_class(self,module,name):
        permitted={('numpy','dtype'):np.dtype,('collections','Counter'):collections.Counter,
            ('numpy._core.numeric','_frombuffer'):numeric_array,('numpy.core.numeric','_frombuffer'):numeric_array,
            ('numpy._core.multiarray','scalar'):numeric_scalar,('numpy.core.multiarray','scalar'):numeric_scalar}
        if (module,name) not in permitted:raise ValueError(f'forbidden pickle global {module}.{name}')
        return permitted[module,name]

def load(path):
    opcodes=collections.Counter()
    with path.open('rb') as f:
        for op,arg,pos in pickletools.genops(f):opcodes[op.name]+=1
    with path.open('rb') as f:
        value=NumericReader(f).load()
        if f.read(1):raise ValueError('trailing bytes')
    return value,dict(opcodes)
