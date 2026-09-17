# Operational safety

- The two-pedal software stop is disabled by default so single-pedal commands
  are dispatched immediately. Use the safety-rated physical emergency-stop
  button for emergencies and keep it within reach.
- If `FOOT_PEDAL_TWO_PEDAL_STOP_ENABLED=true` is explicitly restored, holding
  any two configured pedals requests `rm_set_arm_stop()` on each enabled arm,
  discards the unfinished episode, and shuts down. Test it with no payload and
  a clear workspace. It is still not a safety-rated emergency stop.
- `EPISODE_RIGHT_ARM_RESET_ENABLED=false` is the committed default. Do not
  enable automatic motion until the workspace is clear and an operator is at
  the emergency stop.
- When enabled, the pre-episode reset commands only the right arm. The left arm
  is not sent a reset target. Drag mode starts only after reset succeeds.
- `EPISODE_RETURN_TO_START_ENABLED` is a separate dynamic reset. The start
  command records the live right-follower joints; finish/save and discard then
  command the right follower back to that pose after drag is disabled. The
  committed template keeps it off, while this machine's local single-arm
  profile enables it at 5% speed.
- A dynamic return is real robot motion along a joint-space trajectory. Keep
  the full return path clear, stay at the physical emergency stop, and do not
  assume the tool follows a straight Cartesian path. A per-joint displacement
  above the configured maximum is rejected before motion.
- Dynamic return restores the seven right-arm joints. The initial gripper
  position remains in dataset metadata but is not automatically commanded;
  during READY the right follower gripper follows the leader again. Keep
  fingers and loose objects away from the gripper whenever the app is running.
- The configured right-arm target is
  `0.251,-0.385,5.442,90.016,0.627,89.769,0.167` degrees. A start pose more than
  45 degrees away on any joint is rejected before motion.
- Run `scripts/run_capture.sh --check` after every configuration change. This
  does not stop a running process and does not open hardware.
- Keep the capture UI bound to `127.0.0.1` unless network access is explicitly
  protected. The UI is not an authentication boundary.
- `scripts/stop_capture_app.sh` sends a graceful stop first, then escalates to
  TERM and KILL for matching capture processes. Do not run it while another
  intended collection job is active on the same host.
- Never place credentials, private SSH keys, datasets, videos or local `.env`
  files in Git.
