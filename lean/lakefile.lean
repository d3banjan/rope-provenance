import Lake
open Lake DSL

package rolegate where
  name := `rolegate

require mathlib from git
  "https://github.com/leanprover-community/mathlib4" @
  "8f9d9cff6bd728b17a24e163c9402775d9e6a365"

lean_lib RoleGateReachability where
  roots := #[`RoleGateReachability]
