# Three-Arm Training Summary

| Arm | Status | Final Train Loss | Final Eval Loss | Train-Loss Last 10% | Grad-Norm Early (p0-p20) | Grad-Norm Late (p80-p100) | Runtime | Throughput |
|---|---|---|---|---|---|---|---|---|
| vanilla | [incomplete] | — | — | 1.711 | 2.02–2.22 | 1.48–1.86 | — | — |
| vanilla_zeroed | [incomplete] | — | — | 1.808 | 2.19–2.19 | 2.00–2.00 | — | — |
| rope_prov | [incomplete] | — | — | 4.376 | 12.88–16.25 | 9.94–11.19 | — | — |

## Interpretation guide

- **Compare `rope_prov` final eval loss vs `vanilla_zeroed`**, not vs `vanilla`. `vanilla_zeroed` is the architectural ceiling (T2b). If `rope_prov` ≤ `vanilla_zeroed`, the role signal is doing real work.
- **`vanilla_zeroed` ≈ `vanilla`** ⇒ high-freq RoPE pairs are dimensionally underutilized; rope_prov has no intrinsic capacity penalty. **`vanilla_zeroed` ≫ `vanilla`** ⇒ those pairs were structural; ceiling is constrained.
- **Late grad-norm** drifting toward 1-3 ⇒ training stabilized. Stuck at 10+ ⇒ unresolved instability; bump `max_grad_norm`.

## Raw

```json
[
  {
    "name": "vanilla",
    "log_path": "/tmp/vanilla_full.log",
    "incomplete": true,
    "final_train_loss": null,
    "final_eval_loss": null,
    "eval_losses": [],
    "train_runtime_s": null,
    "overall_rate_ex_s": null,
    "n_train_loss_points": 16,
    "n_grad_norm_points": 16
  },
  {
    "name": "vanilla_zeroed",
    "log_path": "/tmp/vanilla_zeroed_full.log",
    "incomplete": true,
    "final_train_loss": null,
    "final_eval_loss": null,
    "eval_losses": [],
    "train_runtime_s": null,
    "overall_rate_ex_s": null,
    "n_train_loss_points": 3,
    "n_grad_norm_points": 3
  },
  {
    "name": "rope_prov",
    "log_path": "/tmp/rope_prov_full.log",
    "incomplete": true,
    "final_train_loss": null,
    "final_eval_loss": null,
    "eval_losses": [],
    "train_runtime_s": null,
    "overall_rate_ex_s": null,
    "n_train_loss_points": 16,
    "n_grad_norm_points": 16
  }
]
```