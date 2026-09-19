"""top_k ablation with the ridge penalty tuned per k.

Alpha has to be tuned per k or the comparison is confounded: in_dim goes
512 -> 2560 across the sweep, so a fixed penalty is a different amount of
regularisation at each k, and an apparent "more source layers hurt" result is then
indistinguishable from "the penalty was too small for a wider design matrix".

Three-way split BY PROMPT: 30 fit / 10 validation / 10 test. alpha is chosen on
validation, never on test. The reported number is test.
"""
from __future__ import annotations
import json, statistics, torch
from train_mlp_mapper import get_topk_source_layers, compute_ridge_weights

KS=[1,2,3,4,5]; ALPHAS=[1.0,10.0,100.0,1000.0,10000.0]; SEEDS=[0,1,2]

def ridge(X,Y,a):
    # The project's centered estimator, so only alpha varies between configs.
    return compute_ridge_weights(X,Y,alpha=a)

def r2(Y,P):
    ss=torch.sum((Y-P)**2).item(); st=torch.sum((Y-Y.mean(0,keepdim=True))**2).item()
    return 1.0-ss/(st+1e-10)

def flat(data,n,key,idxs):
    out={}
    for l in range(n):
        rows=[]
        for p in idxs:
            t=data[p]["layers"][l][key].squeeze(0); s=data[p]["seq_len"]
            rows.append(t.permute(1,0,2).contiguous().view(s,-1).float())
        out[l]=torch.cat(rows,0)
    return out

ds=torch.load("data/calibration_kv_pairs.pt",map_location="cpu",weights_only=False)
n1,n3=ds["num_layers_1b"],ds["num_layers_3b"]; P=len(ds["data_1b"])
res={k:{"k":[],"v":[],"alpha":[]} for k in KS}
for seed in SEEDS:
    g=torch.Generator().manual_seed(seed)
    pm=torch.randperm(P,generator=g).tolist()
    te,va,tr=pm[:10],pm[10:20],pm[20:]
    S={}
    for nm,data,n in [("1b",ds["data_1b"],n1),("3b",ds["data_3b"],n3)]:
        for key,tag in [("k_clean","k"),("v","v")]:
            for sp,idx in [("tr",tr),("va",va),("te",te)]:
                S[f"{nm}_{tag}_{sp}"]=flat(data,n,key,idx)
    for k in KS:
        acc={"k":[],"v":[]}; chosen=[]
        for j in range(n3):
            src=get_topk_source_layers(j,n3,n1,k=k)
            for tag in ("k","v"):
                Xtr=torch.cat([S[f"1b_{tag}_tr"][l] for l in src],-1)
                Xva=torch.cat([S[f"1b_{tag}_va"][l] for l in src],-1)
                Xte=torch.cat([S[f"1b_{tag}_te"][l] for l in src],-1)
                Ytr,Yva,Yte=S[f"3b_{tag}_tr"][j],S[f"3b_{tag}_va"][j],S[f"3b_{tag}_te"][j]
                best=(-9e9,None,None,None)
                for a in ALPHAS:
                    W,b=ridge(Xtr,Ytr,a)
                    sc=r2(Yva,Xva@W+b)
                    if sc>best[0]: best=(sc,W,b,a)
                _,W,b,a_best=best
                acc[tag].append(r2(Yte,Xte@W+b))
                chosen.append(a_best)
        for tag in ("k","v"): res[k][tag].append(sum(acc[tag])/len(acc[tag]))
        res[k]["alpha"].append(statistics.median(chosen))
    print(f"  seed {seed} done",flush=True)

print("\n"+"="*70)
print("TEST R^2, ridge penalty tuned on a separate validation split")
print("30 fit / 10 validation / 10 test, all splits by prompt, 3 seeds")
print("="*70)
print(f"{'top_k':>6}{'in_dim':>8}{'KEYS test':>22}{'VALUES test':>22}{'alpha':>9}")
print("-"*70)
for k in KS:
    kk,vv=res[k]["k"],res[k]["v"]
    print(f"{k:>6}{k*512:>8}{statistics.mean(kk)*100:>14.1f}%  [{min(kk)*100:5.1f},{max(kk)*100:5.1f}]"
          f"{statistics.mean(vv)*100:>12.1f}%  [{min(vv)*100:5.1f},{max(vv)*100:5.1f}]"
          +f"{statistics.median(res[k]['alpha']):>9.0f}"
          +("  <- shipped" if k==3 else ""))
print("="*70)
m=lambda k:statistics.mean(res[k]["k"])+statistics.mean(res[k]["v"])
print(f"\nbest top_k on test = {max(KS,key=m)}")
print(f"shipped k=3 vs k=1: keys {(statistics.mean(res[3]['k'])-statistics.mean(res[1]['k']))*100:+.1f} pts, "
      f"values {(statistics.mean(res[3]['v'])-statistics.mean(res[1]['v']))*100:+.1f} pts")
json.dump({str(k):{a:res[k][a] for a in ('k','v')} for k in KS},
          open("data/topk_ablation_tuned.json","w"),indent=2)
print("saved -> data/topk_ablation_tuned.json")
