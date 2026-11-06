# Integration and arm definitions

`Controller("stall_refresh")` is instantiated once per episode. Before each native inference call, pass the uint8 RGB `(224,224,3)`, public end-effector position `(3,)`, and finger positions `(2,)` to `before`. If it returns true, clear the authors' visual cache before that same model call. Pass native action, cropped image and reuse flag to `after`.

The authors' arm retains native VLA-Cache without this reset controller. The other controller names are historical alternatives, not the selected final method. `reference/runtime.py` records how native cache indices and diagnostics are bound. It requires the exact modified Transformers/processor integration, not an arbitrary current OpenVLA install; preserve that native-path contract before claiming parity. The completed 120-case experiment uses the original runner, not the later partial HarnessStudio migration.
