"""top_k ablation across several prompt-level splits, at fixed alpha.

One split of 10 held-out prompts is ~258 tokens, not enough to trust an ordering.
This repeats the ablation over 5 random splits and reports mean and spread, so any
claim about which top_k is best has to survive resampling. Alpha is held fixed
here; see ablate_topk_tuned.py for the version that tunes it per k, which is the
one to read.
"""
from __future__ import annotations
import json, statistics, torch
from train_mlp_mapper import compute_ridge_weights, get_topk_source_layers

KS=[1,2,3,4,5]; SEEDS=[0,1,2,3,4]; N_TEST=10

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
res={k:{"k":[],"v":[]} for k in KS}
for seed in SEEDS:
    g=torch.Generator().manual_seed(seed)
    perm=torch.randperm(P,generator=g).tolist()
    te,tr=perm[:N_TEST],perm[N_TEST:]
    S={}
    for nm,data,n in [("1b",ds["data_1b"],n1),("3b",ds["data_3b"],n3)]:
        for key,tag in [("k_clean","k"),("v","v")]:
            S[f"{nm}_{tag}_tr"]=flat(data,n,key,tr); S[f"{nm}_{tag}_te"]=flat(data,n,key,te)
    for k in KS:
        acc={"k":[],"v":[]}
        for j in range(n3):
            src=get_topk_source_layers(j,n3,n1,k=k)
            for tag in ("k","v"):
                Xtr=torch.cat([S[f"1b_{tag}_tr"][l] for l in src],-1)
                Xte=torch.cat([S[f"1b_{tag}_te"][l] for l in src],-1)
                W,b=compute_ridge_weights(Xtr,S[f"3b_{tag}_tr"][j],alpha=1.0)
                acc[tag].append(r2(S[f"3b_{tag}_te"][j],Xte@W+b))
        for tag in ("k","v"): res[k][tag].append(sum(acc[tag])/len(acc[tag]))
    print(f"  seed {seed} done",flush=True)

print("\n"+"="*72)
print("HELD-OUT R^2 ACROSS 5 PROMPT-LEVEL SPLITS (mean, min..max over seeds)")
print("="*72)
print(f"{'top_k':>6}{'KEYS held-out':>28}{'VALUES held-out':>28}")
print("-"*72)
for k in KS:
    kk,vv=res[k]["k"],res[k]["v"]
    mark=" <- shipped" if k==3 else ""
    print(f"{k:>6}{statistics.mean(kk)*100:>18.1f}%  [{min(kk)*100:5.1f},{max(kk)*100:5.1f}]"
          f"{statistics.mean(vv)*100:>16.1f}%  [{min(vv)*100:5.1f},{max(vv)*100:5.1f}]{mark}")
print("="*72)
bk=max(KS,key=lambda k:statistics.mean(res[k]["k"])+statistics.mean(res[k]["v"]))
print(f"\nbest mean held-out top_k = {bk}")
o1=statistics.mean(res[1]["k"]); o3=statistics.mean(res[3]["k"])
print(f"shipped k=3 vs k=1 on keys: {(o3-o1)*100:+.1f} pts (mean over 5 splits)")
json.dump({str(k):res[k] for k in KS},open("data/topk_ablation_seeds.json","w"),indent=2)
print("saved -> data/topk_ablation_seeds.json")
