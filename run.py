import os
import numpy as np
import torch
from timeit import default_timer as timer

from utils.common_utils import setup_seed
from utils.demo_config import get_demo_config
from utils.select_data_demo import select_data_demo
from utils.metrics_demo import metric
from model.model_network import STDiff


def main():
    setup_seed(2025)
    config = get_demo_config()

    config, n_utils, dataloader, air_data = select_data_demo(config)
    config.model.A = torch.eye(config.model.V)

    n_heads = [4, 1]
    model = STDiff(config.model, n_utils, n_heads, config.model.V).to(config.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    loss_fn = torch.nn.MSELoss().to(config.device)

    out_dir = "outputs/demo_run"
    os.makedirs(out_dir, exist_ok=True)

    max_batches = 5
    metric_list = []
    t0 = timer()

    model.train()
    for i, batch in enumerate(dataloader):
        if i >= max_batches:
            break

        future, history, pos_w, pos_d, edge = batch
        future = torch.nan_to_num(future, nan=0.0).to(config.device)
        history = torch.nan_to_num(history, nan=0.0).to(config.device)
        pos_w = pos_w.to(config.device)
        pos_d = pos_d.to(config.device)

        out = model(history, pos_w, pos_d)
        pred = out[:, -config.T_h:]

        loss = loss_fn(future, pred)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        future12 = future[:, :12]
        pred12 = pred[:, :12]
        a, b, c, _ = future12.shape

        y_true = air_data.reverse_normalization(future12.detach().cpu().numpy()).reshape(a, b, -1)
        y_pred = air_data.reverse_normalization(pred12.detach().cpu().numpy()).reshape(a, b, -1)

        res = metric(y_true, y_pred)
        metric_list.append(res)

        print(f"[demo] batch {i+1}/{max_batches} loss={loss.item():.4f} "
              f"mae={res[0]:.4f} rmse={res[2]:.4f} smape={res[3]:.2f}")

    elapsed = timer() - t0
    arr = np.array(metric_list)

    torch.save(model.state_dict(), f"{out_dir}/demo_model.pth")
    np.save(f"{out_dir}/demo_metrics.npy", arr)

    with open(f"{out_dir}/summary.txt", "w", encoding="utf-8") as f:
        f.write(f"batches={len(metric_list)} time_sec={elapsed:.2f}\n")
        f.write(f"mae={arr[:,0].mean():.6f} mse={arr[:,1].mean():.6f} rmse={arr[:,2].mean():.6f} smape={arr[:,3].mean():.6f}\n")

    print(f"[demo] done. outputs saved to {out_dir}, time={elapsed:.2f}s")


if __name__ == "__main__":
    main()
