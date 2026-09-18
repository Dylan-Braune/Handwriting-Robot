"""Diagnostic, not part of the pipeline: can StyleTransferRNN's architecture
learn ANYTHING useful for just ONE author (91 lines, ~82 train/9 val), with
more capacity than the shared 10-author run? If this ALSO collapses to a
near-flat correction, the problem is the point-correspondence/formulation,
not just "not enough capacity/data shared across 10 authors". If this DOES
produce legible, author-shaped output, the shared run just needs more
capacity or per-author fine-tuning, not a redesign.
"""
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from StyleTransferRNN import PairDataset, collate, StyleTransferRNN, render_xy

device = torch.device("cpu")
writers = ["150"]
train_ds = PairDataset(writers, is_train=True)
val_ds = PairDataset(writers, is_train=False)
print(f"train {len(train_ds)}  val {len(val_ds)}")
train_loader = torch.utils.data.DataLoader(train_ds, batch_size=8, shuffle=True, collate_fn=collate)
val_loader = torch.utils.data.DataLoader(val_ds, batch_size=8, shuffle=False, collate_fn=collate)

model = StyleTransferRNN(len(writers), hidden=256, layers=2).to(device)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(1, 151):
    model.train()
    tot, n = 0.0, 0
    for wb, ab, tb in train_loader:
        pred = model(ab, wb)
        loss = F.smooth_l1_loss(pred, tb) + F.smooth_l1_loss(torch.cumsum(pred, dim=1), torch.cumsum(tb, dim=1))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); n += 1
    if epoch % 10 == 0 or epoch == 1:
        model.eval()
        vtot, vn = 0.0, 0
        with torch.no_grad():
            for wb, ab, tb in val_loader:
                pred = model(ab, wb)
                vl = F.smooth_l1_loss(pred, tb) + F.smooth_l1_loss(torch.cumsum(pred, dim=1), torch.cumsum(tb, dim=1))
                vtot += vl.item(); vn += 1
        print(f"epoch {epoch:03d} train {tot/n:.5f} val {vtot/max(1,vn):.5f}")

# render one TRAIN example (upper bound sanity check) and one VAL example
model.eval()
with torch.no_grad():
    for tag, ds, idx in [("TRAIN", train_ds, 0), ("VAL", val_ds, 0)]:
        w, anchor_d, target_d = ds[idx]
        pred_d = model(torch.from_numpy(anchor_d).unsqueeze(0), torch.tensor([w]))
        pred_xy = np.cumsum(pred_d[0].numpy(), axis=0)
        anchor_xy = np.cumsum(anchor_d, axis=0)
        target_xy = np.cumsum(target_d, axis=0)
        render_xy(pred_xy).save(rf"C:\Users\braun\AppData\Local\Temp\claude\diag_{tag}_pred.png")
        render_xy(anchor_xy).save(rf"C:\Users\braun\AppData\Local\Temp\claude\diag_{tag}_anchor.png")
        render_xy(target_xy).save(rf"C:\Users\braun\AppData\Local\Temp\claude\diag_{tag}_target.png")
        print(tag, "writer", w)
print("done, images in C:\\Users\\braun\\AppData\\Local\\Temp\\claude\\diag_*.png")
