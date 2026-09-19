"""Perplexity evaluation with enough data to support a claim.

Note on the parity metric used elsewhere in this repo:

    parity_pct = ppl_native / ppl_trans * 100

A "70.5% parity" under that definition means the transplant's perplexity is
1/0.705 = 1.42x native -- a 42% degradation. PPL ratios are reported directly here
to avoid the inversion.

This evaluation:
  * WikiText-103 test, concatenated and chunked into fixed windows, the standard
    way LM perplexity is measured. Hundreds of windows rather than 4 passages.
  * Every window is scored under the SAME prefix/target construction as the
    original, so the numbers stay comparable.
  * Per-window losses are kept, so the transplant-vs-native gap gets a paired
    bootstrap CI instead of a single ratio.
  * A small technical-prose set is scored alongside it, so the effect of
    evaluation-set choice is visible rather than assumed.

Note on domain: the calibration prompts are technical prose, so WikiText is
out-of-domain for the projector. That is the honest generalisation test, and it is
harsher than the original's in-domain passages. Both are reported.
"""
from __future__ import annotations
import json, math, random, sys
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from fused_projector import FusedKVProjector
from rope_utils import inverse_rope
from evaluate_perplexity import EVALUATION_DATASET

M1, M3 = "unsloth/Llama-3.2-1B", "unsloth/Llama-3.2-3B"
PREFIX_LEN, TARGET_LEN, N_WINDOWS = 128, 128, 200
MAPPERS = [("shipped  (1.4k tok, a=1)", "weights/mapper_shipped.pt"),
           ("tuned    (1.4k tok, a=100)", "weights/mapper_tuned.pt"),
           ("LARGE    (200k tok, a tuned)", "weights/mapper_large.pt")]


def get_kv(past):
    if hasattr(past, "layers"):
        return ([getattr(l, "keys", getattr(l, "key_states", None)) for l in past.layers],
                [getattr(l, "values", getattr(l, "value_states", None)) for l in past.layers])
    if hasattr(past, "key_cache"):
        return past.key_cache, past.value_cache
    return [l[0] for l in past], [l[1] for l in past]


def boot(d, n=10000, seed=0):
    rng = random.Random(seed); N = len(d)
    ms = sorted(sum(d[rng.randrange(N)] for _ in range(N)) / N for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def windows(tok, device):
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="test")
    text = "\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt")["input_ids"][0]
    W = PREFIX_LEN + TARGET_LEN
    n = min(N_WINDOWS, ids.shape[0] // W)
    print(f"wikitext-103 test: {ids.shape[0]:,} tokens -> {n} windows of {W}")
    return [(ids[i * W:i * W + PREFIX_LEN].unsqueeze(0).to(device),
             ids[i * W + PREFIX_LEN:(i + 1) * W].unsqueeze(0).to(device)) for i in range(n)]


def score(m3, m1, projs, pairs, label):
    rec = {k: [] for k in ["native3b", "native1b"] + [n for n, _ in MAPPERS]}
    for pi, (pre, tgt) in enumerate(pairs):
        full = torch.cat([pre, tgt], -1); pl = pre.shape[1]
        with torch.no_grad():
            lg = m3(full).logits[:, pl - 1:-1, :]
            rec["native3b"].append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item())
            lg = m1(full).logits[:, pl - 1:-1, :]
            rec["native1b"].append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item())
            out1 = m1(pre, use_cache=True)
            ks, vs = get_kv(out1.past_key_values)
            ei = torch.cat([pre[:, -1:], tgt[:, :-1]], -1)
            for name, proj in projs.items():
                uk = [inverse_rope(k, head_dim=proj.head_dim_1b, base=500000.0) for k in ks]
                cache = proj.project_and_build_cache(uk, vs, pl)
                lg = m3(ei, past_key_values=cache, use_cache=True).logits
                rec[name].append(F.cross_entropy(lg.reshape(-1, lg.shape[-1]), tgt.reshape(-1)).item())
        if (pi + 1) % 25 == 0: print(f"  {label}: {pi+1}/{len(pairs)}", flush=True)
    return rec


def report(rec, title, n):
    print("\n" + "=" * 82); print(title + f"   (n = {n})"); print("=" * 82)
    nat = rec["native3b"]; mn = sum(nat) / len(nat)
    print(f"{'condition':<28}{'loss':>9}{'PPL':>10}{'vs 3B (nats)':>16}{'95% CI':>19}")
    print("-" * 82)
    for k in ["native3b", "native1b"] + [n2 for n2, _ in MAPPERS]:
        v = rec[k]; m = sum(v) / len(v)
        if k == "native3b":
            print(f"{'native 3B (ceiling)':<28}{m:>9.4f}{math.exp(m):>10.2f}{'—':>16}{'—':>19}")
        else:
            d = [a - b for a, b in zip(v, nat)]
            lo, hi = boot(d); dm = sum(d) / len(d)
            lbl = "native 1B (floor)" if k == "native1b" else k
            print(f"{lbl:<28}{m:>9.4f}{math.exp(m):>10.2f}{dm:>+16.4f}   [{lo:+.4f}, {hi:+.4f}]")
    print("-" * 82)
    for k in [n2 for n2, _ in MAPPERS]:
        m = sum(rec[k]) / len(rec[k])
        print(f"  {k}: PPL ratio to native 3B = {math.exp(m)/math.exp(mn):.2f}x"
              f"   (project's 'parity %' = {math.exp(mn)/math.exp(m)*100:.1f}%)")


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(M1)
    projs = {n: FusedKVProjector(weights_path=p, device=device) for n, p in MAPPERS}
    m1 = AutoModelForCausalLM.from_pretrained(M1, torch_dtype=torch.float16).to(device).eval()
    m3 = AutoModelForCausalLM.from_pretrained(M3, torch_dtype=torch.float16).to(device).eval()

    pairs = windows(tok, device)
    r_wiki = score(m3, m1, projs, pairs, "wikitext")
    report(r_wiki, "WIKITEXT-103 TEST  (held out, standard corpus)", len(pairs))

    orig = [(tok(i["prefix"], return_tensors="pt")["input_ids"].to(device),
             tok(i["target"], return_tensors="pt", add_special_tokens=False)["input_ids"].to(device))
            for i in EVALUATION_DATASET]
    r_orig = score(m3, m1, projs, orig, "technical")
    report(r_orig, "TECHNICAL PROSE SET  (in-domain comparison)", len(orig))

    json.dump({"wikitext": r_wiki, "technical": r_orig},
              open("data/ppl_proper_results.json", "w"), indent=2)
    print("\nsaved -> data/ppl_proper_results.json")


if __name__ == "__main__":
    main()
