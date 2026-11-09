"""Resilient Imago benchmark: subprocess-per-image to survive native crashes (free(): invalid pointer).
Produces same JSON shape as benchmark_ocsr + caches predictions."""
from __future__ import annotations
import json, subprocess, sys, time, os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from toolchain.frozen import load_frozen, resolve_frozen_rows, PredictionCache
from toolchain.benchmark_run import _canonical_smiles, _resolve_image, _image_id

VENV_PY = str(ROOT / ".venv/bin/python")
ONE = r'''
import json,sys
from imago import Imago
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
img=sys.argv[1]
try:
    im=Imago(); im.load_image_from_file(img); im.recognize(); mol=im.molecule
    if not mol:
        print(json.dumps({"smiles":"","incomplete":True}))
    else:
        m=Chem.MolFromMolBlock(str(mol))
        if m is None: print(json.dumps({"smiles":"","incomplete":True}))
        else:
            s=Chem.MolToSmiles(m,canonical=True)
            print(json.dumps({"smiles":s or "","incomplete":not bool(s)}))
except Exception as e:
    print(json.dumps({"smiles":"","incomplete":True,"error":str(e)[:200]}))
'''

def run_one(img: Path, timeout: float = 60.0):
    t0=time.perf_counter()
    try:
        r=subprocess.run([VENV_PY,'-c',ONE,str(img)], capture_output=True, text=True, timeout=timeout)
        dt=(time.perf_counter()-t0)*1000
        if r.returncode != 0:
            return {"smiles":"","incomplete":True,"error":(r.stderr or r.stdout)[-200:] or f"rc={r.returncode}"}, dt
        line=(r.stdout.strip().splitlines() or ['{}'])[-1]
        return json.loads(line), dt
    except subprocess.TimeoutExpired:
        return {"smiles":"","incomplete":True,"error":"timeout"}, (time.perf_counter()-t0)*1000
    except Exception as e:
        return {"smiles":"","incomplete":True,"error":str(e)[:200]}, (time.perf_counter()-t0)*1000

def main():
    frozen_path = Path(sys.argv[1]) if len(sys.argv)>1 else ROOT/"data/manifests/frozen/printed.frozen.json"
    split = sys.argv[2] if len(sys.argv)>2 else "eval"
    out = Path(sys.argv[3]) if len(sys.argv)>3 else ROOT/"data/benchmark/eval-1000/printed.imago.eval1000.json"
    frozen=load_frozen(frozen_path)
    from toolchain.benchmark_data import ensure_benchmark_data
    ensure_benchmark_data(frozen.get('model') or frozen.get('dossier') or '', registry=None)
    rows, labels_path = resolve_frozen_rows(frozen, split=split)
    model_key='ocsr-printed'; variant_key='imago-2.0.0rc1'
    cache=PredictionCache()
    cached=cache.load(model_key, variant_key)
    out_rows=[]; exact=tfm=valid=incomplete_n=0
    canonical_match_n=canonical_comparable_n=0
    latencies=[]; cache_hits=0; errors=0
    for i,row in enumerate(rows):
        img=_resolve_image(row, labels_path); iid=_image_id(row)
        expected=(row.get('smiles') or row.get('SMILES') or '').strip()
        if iid in cached and cached[iid] is not None:
            pred=cached[iid]; dt=0.0; hit=True
        else:
            pred,dt=run_one(img); hit=False
            cache.append(model_key,variant_key,iid,pred)
            cached[iid]=pred
        cache_hits+=int(hit)
        if not hit: latencies.append(dt)
        if pred.get('error'): errors+=1
        got=(pred.get('smiles') or '').strip()
        is_exact = got==expected
        is_incomplete=bool(pred.get('incomplete')); is_valid=bool(got) and not is_incomplete
        exact+=int(is_exact); incomplete_n+=int(is_incomplete); valid+=int(is_valid)
        ce,cg=_canonical_smiles(expected), _canonical_smiles(got)
        can_match=None
        if ce is not None and cg is not None:
            canonical_comparable_n+=1; can_match=(ce==cg); canonical_match_n+=int(can_match)
        out_rows.append({"id":iid,"expected":expected,"got":got,"exact_match":is_exact,
                         "canonical_expected":ce,"canonical_got":cg,"canonical_match":can_match,
                         "valid":is_valid,"incomplete":is_incomplete,"latency_ms":round(dt,2),
                         "cache_hit":hit,"error":pred.get("error")})
        if (i+1)%100==0: print(f"progress {i+1}/{len(rows)} can={canonical_match_n}/{canonical_comparable_n} err={errors}", flush=True)
    n=len(rows) or 1
    res={"task":"ocsr","count":len(rows),"model_key":model_key,"variant_key":variant_key,"split":split,
         "frozen_manifest":str(frozen_path),"predictor":"imago","predictor_version":"2.0.0rc1",
         "exact_match":exact,"exact_match_rate":exact/n,
         "canonical_match":canonical_match_n,"canonical_comparable":canonical_comparable_n,
         "canonical_match_rate":(canonical_match_n/canonical_comparable_n if canonical_comparable_n else None),
         "canonical_note":"RDKit canonical SMILES equality; primary metric. exact_match is raw string equality.",
         "valid":valid,"valid_rate":valid/n,"incomplete":incomplete_n,"incomplete_rate":incomplete_n/n,
         "latency_ms":{"p50":float(np.percentile(latencies,50)) if latencies else None,
                       "p95":float(np.percentile(latencies,95)) if latencies else None,
                       "mean":float(np.mean(latencies)) if latencies else None,"n_measured":len(latencies)},
         "cache_hits":cache_hits,"errors":errors,"details":out_rows}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res,indent=2,ensure_ascii=False)+"\n",encoding='utf-8')
    cr=res['canonical_match_rate']
    print(f"wrote {out}")
    print(f"count={len(rows)} canonical_rdkit(primary)={canonical_match_n}/{canonical_comparable_n} ({cr:.3f}) exact_string={exact} ({exact/n:.3f}) valid={valid} incomplete={incomplete_n} errors={errors} cache_hits={cache_hits}")
    if latencies: print(f"latency_ms p50={res['latency_ms']['p50']:.1f} p95={res['latency_ms']['p95']:.1f} n_measured={len(latencies)}")

if __name__=='__main__': main()
