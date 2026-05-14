# Three-Arm Training Summary

| Arm | Status | Final Train Loss | Final Eval Loss | Train-Loss Last 10% | Grad-Norm Early (p0-p20) | Grad-Norm Late (p80-p100) | Runtime | Throughput |
|---|---|---|---|---|---|---|---|---|
| vanilla | ok | 1.673 | 1.629 | 1.660 | 1.24–2.20 | 1.16–2.20 | 29.9 min | 31.7 ex/s |
| vanilla_zeroed | ok | 1.673 | 1.628 | 1.660 | 1.23–2.22 | 1.16–2.23 | 30.8 min | 30.8 ex/s |
| rope_prov | ok | 3.290 | 3.166 | 2.895 | 7.53–19.50 | 4.50–9.19 | 34.0 min | 27.9 ex/s |

## Interpretation guide

- **Compare `rope_prov` final eval loss vs `vanilla_zeroed`**, not vs `vanilla`. `vanilla_zeroed` is the architectural ceiling (T2b). If `rope_prov` ≤ `vanilla_zeroed`, the role signal is doing real work.
- **`vanilla_zeroed` ≈ `vanilla`** ⇒ the zeroed low-frequency RoPE pairs are cheap at this sequence length; rope_prov has no intrinsic capacity penalty. **`vanilla_zeroed` ≫ `vanilla`** ⇒ those pairs were structural; ceiling is constrained.
- **Late grad-norm** drifting toward 1-3 ⇒ training stabilized. Stuck at 10+ ⇒ unresolved instability; bump `max_grad_norm`.

## Raw

```json
[
  {
    "name": "vanilla",
    "log_path": "wandb/run-20260514_074429-99p44wbd/files/output.log",
    "incomplete": false,
    "final_train_loss": 1.6727153406486854,
    "final_eval_loss": 1.6287761926651,
    "eval_losses": [
      1.7065120935440063,
      1.6594570875167847,
      1.6411381959915161,
      1.6330255270004272,
      1.6302251815795898,
      1.6292632818222046,
      1.6282421350479126,
      1.6287761926651
    ],
    "train_runtime_s": 1794.9332,
    "overall_rate_ex_s": 31.69,
    "n_train_loss_points": 177,
    "n_grad_norm_points": 177
  },
  {
    "name": "vanilla_zeroed",
    "log_path": "wandb/run-20260514_071326-0lvji7jx/files/output.log",
    "incomplete": false,
    "final_train_loss": 1.6726698510281675,
    "final_eval_loss": 1.6282860040664673,
    "eval_losses": [
      1.7069637775421143,
      1.6589758396148682,
      1.6407980918884277,
      1.6329410076141357,
      1.6301504373550415,
      1.6289162635803223,
      1.628543496131897,
      1.6282860040664673
    ],
    "train_runtime_s": 1846.4751,
    "overall_rate_ex_s": 30.8,
    "n_train_loss_points": 177,
    "n_grad_norm_points": 177
  },
  {
    "name": "rope_prov",
    "log_path": "wandb/run-20260514_081441-cfnltgh3/files/output.log",
    "incomplete": false,
    "final_train_loss": 3.2901602672027037,
    "final_eval_loss": 3.1661410331726074,
    "eval_losses": [
      4.495491981506348,
      3.68415904045105,
      3.3471055030822754,
      3.2232470512390137,
      3.183598518371582,
      3.1708874702453613,
      3.16957426071167,
      3.1661410331726074
    ],
    "train_runtime_s": 2038.9637,
    "overall_rate_ex_s": 27.89,
    "n_train_loss_points": 177,
    "n_grad_norm_points": 177
  }
]
```