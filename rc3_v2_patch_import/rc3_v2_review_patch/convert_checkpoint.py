"""Lossless tensor export for the v2 checkpoint schema; no training or general pickle execution."""
import argparse, collections, hashlib, io, json, pickle, zipfile
from pathlib import Path
import numpy as np

def load_checkpoint(path, expected_phase=None):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        prefix = name[:-len('data.pkl')]
        arrays = {}

        def rebuild(storage, offset, size, stride, requires_grad, hooks, *args):
            offset, size, stride = int(offset), tuple(map(int, size)), tuple(map(int, stride))
            if offset < 0 or any(x < 0 for x in (*size, *stride)):
                raise ValueError('unsupported negative tensor layout')
            end = offset + sum((a-1)*b for a, b in zip(size, stride)) if all(size) else offset
            if size and end >= storage.size:
                raise ValueError('tensor layout outside storage')
            return np.ndarray(size, dtype=storage.dtype, buffer=storage,
                              offset=offset*storage.itemsize,
                              strides=tuple(x*storage.itemsize for x in stride)).copy()

        class SafeTensorReader(pickle.Unpickler):
            def find_class(self, module, name):
                allowed = {('collections', 'OrderedDict'): collections.OrderedDict,
                           ('torch._utils', '_rebuild_tensor_v2'): rebuild,
                           ('torch', 'FloatStorage'): np.dtype('<f4'),
                           ('torch', 'LongStorage'): np.dtype('<i8')}
                if (module, name) not in allowed:
                    raise ValueError(f'not allowlisted: {module}.{name}')
                return allowed[module, name]

            def persistent_load(self, pid):
                kind, dtype, key, device, count = pid
                if kind != 'storage' or '/' in key or '..' in key:
                    raise ValueError('unsupported storage reference')
                if key not in arrays:
                    arrays[key] = np.frombuffer(archive.read(prefix+'data/'+key), dtype=dtype)
                    if arrays[key].size != count:
                        raise ValueError('storage count mismatch')
                return arrays[key]

        checkpoint = SafeTensorReader(io.BytesIO(archive.read(name))).load()
        if expected_phase is not None:
            if expected_phase not in ('bc','ppo'):
                raise ValueError('expected_phase must be bc or ppo')
            if checkpoint.get('phase') != expected_phase:
                raise ValueError(f"checkpoint phase {checkpoint.get('phase')!r} does not match {expected_phase!r}")
        return checkpoint

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--stage',choices=['bc','ppo'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    checkpoint=load_checkpoint(args.input,expected_phase=args.stage)
    metadata=dict(stage=args.stage,dim=checkpoint['dim'],feature_names=checkpoint['feature_names'],
        source_checkpoint=args.input.name,source_sha256=hashlib.sha256(args.input.read_bytes()).hexdigest(),
        conversion='lossless arrays; no training',original_metadata={k:v for k,v in checkpoint.items()
        if k not in ('state_dict','mean','std','feature_names')})
    args.output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,metadata=np.asarray(json.dumps(metadata,ensure_ascii=False)),**checkpoint['state_dict'])
    print(json.dumps(dict(output=str(args.output),stage=args.stage,source_sha256=metadata['source_sha256'])))

if __name__=='__main__': main()
