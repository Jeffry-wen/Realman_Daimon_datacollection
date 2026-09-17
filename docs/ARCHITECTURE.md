# Capture stack

The repository intentionally contains only the aligned capture layer. The
compatible LeRobot checkout remains a separate dependency selected through
`LEROBOT_ROOT`.

```text
scripts/run_capture.sh
  -> config/lab_5090.env
  -> scripts/capture_realman_x5_force_aligned_app.sh
  -> scripts/capture_realman_x5_force_app.sh
  -> conda run ... scripts/capture_realman_x5_force_aligned_app.py
       -> scripts/capture_realman_x5_force_app.py (compatibility patches)
       -> LEROBOT_ROOT/tools/bi_x5_capture_app.py
       -> LEROBOT_ROOT/src/lerobot/... (robot, gripper, force, X5 drivers)
```

The aligned application owns timestamp buffers and dataset writes. Wrist RGB,
four X5 tactile streams, joint state/action and left/right force samples are
selected against one delayed target timestamp. With the supplied profile,
missing or stale required samples abort the episode rather than silently
holding the previous value.

The following companion LeRobot paths are required:

- `tools/bi_x5_capture_app.py`
- `src/lerobot/robots/bi_realman_ugripper_notac_new/`
- `src/lerobot/teleoperators/bi_realman_rm75b_leader/`
- the dataset writer and video utilities used by that capture app

Captured datasets and generated media are excluded by `.gitignore` and are not
part of this source repository.
