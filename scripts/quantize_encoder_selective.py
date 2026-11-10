"""Build selective-dynamic INT8 encoder excluding nodes by name/block patterns (mixed precision)."""
import argparse, os
from pathlib import Path
import onnx
from onnxruntime.quantization.quant_utils import QuantType
from onnxruntime.quantization.quantize import quantize_dynamic

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src', required=True)
    ap.add_argument('--dst', required=True)
    ap.add_argument('--exclude', default='', help='comma-separated substrings; nodes whose name contains any are kept FP32')
    ap.add_argument('--op-types', default='Conv,MatMul,Gemm')
    ap.add_argument('--reduce-range', action='store_true')
    ap.add_argument('--per-channel', action='store_true', default=True)
    ap.add_argument('--no-per-channel', dest='per_channel', action='store_false')
    a=ap.parse_args()
    m=onnx.load(a.src)
    pats=[p.strip() for p in a.exclude.split(',') if p.strip()]
    excl=[]
    for n in m.graph.node:
        nm=n.name or ''
        if any(p in nm for p in pats):
            excl.append(n.name)
    excl=sorted({x for x in excl if x})
    os.makedirs(os.path.dirname(a.dst) or '.', exist_ok=True)
    quantize_dynamic(a.src, a.dst, op_types_to_quantize=a.op_types.split(','), nodes_to_exclude=excl or None,
                     per_channel=a.per_channel, reduce_range=a.reduce_range, weight_type=QuantType.QInt8)
    print('wrote', a.dst, os.path.getsize(a.dst), 'excluded_nodes', len(excl), 'patterns', pats)

if __name__=='__main__': main()
