import glob
import json
import sys

for f in sorted(glob.glob(f"{sys.argv[1]}/*/metrics.jsonl")):
    rows = [json.loads(line) for line in open(f)]
    r = rows[-1]
    b = json.load(open(f.replace("metrics.jsonl", "baselines.json")))
    print(f.split("/")[-2][:24], f"evals={len(rows)}",
          f"id={r['mse_id_last']:.3f} ood={r['mse_ood_last']:.3f} alpha_id={r['alpha_id']:.2f}",
          f"| ridge_ood={b['ridge_ood_mse_pos'][-1]:.3f} dmmse_id={b['dmmse_id_mse_pos'][-1]:.3f}",
          "ymass", [round(v, 2) for v in r["attn_ymass"]], next(iter(r["spec"].items())))
