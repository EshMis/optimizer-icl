"""Print ms per training step for the full-size model: decides CPU vs GPU and run lengths."""
import sys
import time

import torch

sys.path.insert(0, __import__("os").path.dirname(__file__) + "/..")
from icl.data import sample  # noqa: E402
from icl.model import ICLTransformer  # noqa: E402
from icl.optim import Routed  # noqa: E402

dev = sys.argv[1] if len(sys.argv) > 1 else "cpu"
for th in ([int(t) for t in sys.argv[2].split("+")] if len(sys.argv) > 2 else [0]):
    if th:
        torch.set_num_threads(th)
    for opt, lr in [("adamw", 1e-3), ("muon", 0.02)]:
        m = ICLTransformer(8, 16).to(dev)
        o = Routed(m, opt, lr, 1e-3)
        g = torch.Generator(device=dev).manual_seed(0)
        for B in (64, 256):
            for i in range(25):
                if i == 5:
                    if dev == "cuda":
                        torch.cuda.synchronize()
                    t = time.time()
                x, y, _ = sample(B, 16, 8, 0.25, None, g, dev)
                loss = ((m(x, y) - y) ** 2).mean()
                o.zero_grad()
                loss.backward()
                o.step()
            if dev == "cuda":
                torch.cuda.synchronize()
            ms = (time.time() - t) / 20 * 1000
            print(f"threads={th} {opt} B={B}: {ms:.0f} ms/step -> 30k steps = {ms * 30:.0f} s", flush=True)
