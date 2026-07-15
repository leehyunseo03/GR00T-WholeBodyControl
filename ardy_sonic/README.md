# ardy_sonic — Ardy plans, GEAR-SONIC tracks, with receding-horizon replanning

This package closes the loop between two systems that live in **different Python
environments**:

- **Ardy** (NVIDIA's diffusion motion planner) generates a kinematic G1 walk-to-goal
  path (root + 29 DOF). It runs on the **host** in `conda activate ardy`.
- **GEAR-SONIC** (the whole-body tracking policy) *physically* tracks that path in
  Isaac Lab. It runs inside the **`gear-sonic-base` container** via
  `/workspace/isaaclab/isaaclab.sh -p`.

The robot walks toward a destination (default: **5 m forward + a target 29‑DOF
pose**, matching `ardy/workspace/generate_g1_walk_5m.py`). Because the physically
tracked motion drifts from the kinematic plan, Ardy **re‑plans from the robot's
actual achieved pose** each segment until the robot arrives.

```
 HOST (conda ardy)                         shared mount                 CONTAINER (isaaclab.sh -p)
 ┌────────────────────────┐   runtime/requests/*.json   ┌───────────────────────────────────┐
 │ ardy_planner_server.py │◄───────────────────────────│ eval_agent_trl.py                 │
 │  Ardy G1 model (resident)                            │   + ArdyReplanCallback            │
 │  walk→goal, land on pose │──────────────────────────►│   install_live_qpos_segment()     │
 │  SE(2)→world qpos (T,36) │  runtime/responses/*.npz   │   SONIC policy tracks @ 50 Hz     │
 └────────────────────────┘                             └───────────────────────────────────┘
```
`/home/hslee/IsaacLab_ws` (host) is bind-mounted to `/workspace` (container), so the
two processes exchange plans as files under `ardy_sonic/runtime/`.

## Why two processes?

The `ardy` conda env (host miniforge) is **not** mounted into the container, and the
container's Isaac python cannot import Ardy (nor its gated Llama text encoder). So
Ardy stays on the host and SONIC stays in the container; they only share the
filesystem. The Ardy CSV (`root xyz + quat wxyz + 29 joints`, 25 fps, z‑up/x‑fwd) has
the **exact same 29‑joint order** as SONIC's MuJoCo qpos (verified against both G1
MJCFs), so a plan needs only an SE(2) world transform — no joint remapping — before it
is fed to `TrackingCommand.install_live_qpos_segment`.

## Files

| file | env | role |
|---|---|---|
| `protocol.py` | both | request/response schema, atomic file IO, SE(2) transform of a qpos trajectory (numpy only) |
| `ardy_planner_server.py` | host `ardy` | resident Ardy model; serves plans (mirrors `ardy/scripts/generate.py`) |
| `ardy_replan_callback.py` | container | `eval_step` callback: read pose → request plan → install segment → replan → detect arrival |
| `make_placeholder_motion.py` | any (numpy+joblib) | builds the long static placeholder clip the tracker launches with |
| `run_planner.sh` | host | launches the planner server in `conda activate ardy` |
| `run_ardy_sonic.sh` | container | launches `eval_agent_trl.py` with the callback registered |
| `tests/test_bridge_offline.py` | container | offline control-loop test (no Isaac, no Ardy) |
| `runtime/` | shared | `requests/`, `responses/`, `plans/` (debug), `placeholder_motion.pkl` |

## Prerequisites

1. **Ardy** installed in the host `ardy` conda env (see `ardy/workspace/README.md`),
   including Hugging Face auth for the Llama text encoder and a GPU-matching PyTorch.
2. **gear_sonic eval deps in the SONIC python** — the stock Isaac Sim python is
   missing several packages gear_sonic needs for *any* eval: `joblib` (motion_lib),
   `easydict`, `loguru`, `vector_quantize_pytorch` (FSQ policy), `accelerate`.
   plus **exactly `trl==0.28.0`** (newer trl removed `trl.trainer.ppo_trainer`, which
   gear_sonic extends — a wrong version imports but fails at eval).
   `run_ardy_sonic.sh` installs/pins whatever is missing automatically; by hand:
   ```bash
   /workspace/isaaclab/isaaclab.sh -p -m pip install joblib easydict loguru \
       vector_quantize_pytorch accelerate "trl==0.28.0"
   # (transformers>=4.56.2 and accelerate>=1.3.0 are also required; usually already present)
   ```
   Use `isaaclab.sh -p -m pip` (not bare `pip`) so it targets the right interpreter,
   and **do not** let it downgrade `numpy`/`scipy`: gear_sonic pins `numpy==1.26.4`
   but Isaac Sim ships numpy 2.x and the pin is only declarative — downgrading breaks
   Isaac Sim. If a later `ModuleNotFoundError` appears, install just that module the
   same way.
3. **A SONIC checkpoint** — defaults to `sonic_release/last.pt` (already present).

## How to run

### Step 0 (once): validate the Ardy planner on its own — HOST
```bash
conda activate ardy
cd /home/hslee/IsaacLab_ws/GR00T-WholeBodyControl
python ardy_sonic/ardy_planner_server.py --once --distance 5 --duration 10 --start-xy 0 0 --heading 0
# -> writes runtime/plans/once.npy and runtime/responses/once.npz (a world-frame (T,36) walk)
```
This confirms Ardy generation + constraints + terminal landing + world transform work
before you bring up Isaac. (First run downloads/loads the model; it is slow.)

### Step 1: start the planner server — HOST
```bash
bash ardy_sonic/run_planner.sh            # in-process resident model (fast replans)
# or:  ENGINE=subprocess bash ardy_sonic/run_planner.sh   # slower fallback, reloads model per plan
```
Leave it running; it serves every request the tracker makes.

### Step 2: start the tracker — CONTAINER (`gear-sonic-base`)
```bash
docker exec -it gear-sonic-base bash
cd /workspace/GR00T-WholeBodyControl
bash ardy_sonic/run_ardy_sonic.sh
# open the WebRTC viewer at  http://<server-ip>:8211/streaming/client/
```
The robot repeatedly: gets an Ardy plan for the residual path → SONIC tracks ~90 % of
it → re-plans from the real pose → stops when within `arrival_radius` of the goal.

### Tuning (env vars for `run_ardy_sonic.sh`)
| var | default | meaning |
|---|---|---|
| `FORWARD_METERS` | `5.0` | goal distance ahead of the start pose |
| `GOAL_MODE` | `forward` | `forward` (ahead of start heading) or `absolute` |
| `ARRIVAL_RADIUS` | `0.30` | stop when the pelvis is this close to the goal (m) |
| `MAX_PLAN_DISTANCE` | `6.0` | cap per-plan walk distance (chunk longer goals) |
| `SECONDS_PER_METER` | `2.0` | plan duration = max(min, distance × this) → ~0.5 m/s |
| `TRACK_FRACTION` | `0.9` | replan after tracking this fraction of a segment |
| `MAX_REPLANS` / `MAX_STEPS` | `12` / `6000` | safety caps |
| `CHECKPOINT` | `sonic_release/last.pt` | SONIC policy checkpoint |

Absolute goal example:
```bash
GOAL_MODE=absolute bash ardy_sonic/run_ardy_sonic.sh \
  # then pass the goal via the callback override in the script, e.g.
  # ++callbacks.ardy_replan.goal_xy=[3.0,2.0]
```
A specific terminal 29‑DOF pose (MuJoCo order) is passed with
`++callbacks.ardy_replan.target_joint_qpos=[q0,...,q28]`; the default is a neutral
standing pose.

## Design notes / limitations

- **Replan cadence.** Each Ardy plan is a full *walk‑and‑stop*; the tracker follows
  ~`TRACK_FRACTION` of it, then re-plans the residual from the robot's real pose. For a
  5 m goal that is typically 1–3 plans total (initial + drift correction). Physics is
  frozen while a plan is being generated (the eval loop blocks on the response file),
  which is fine — the robot just holds position.
- **Segment length cap.** `install_live_qpos_segment` truncates an injected segment to
  the placeholder clip's frame count. The default placeholder is 1500 frames (30 s @
  50 fps); keep `PLACEHOLDER_FRAMES` ≥ the longest plan you inject.
- **Early terminations disabled.** `run_ardy_sonic.sh` raises the tracking-error
  termination thresholds and `episode_length_s` so the episode never auto-resets
  mid-navigation; arrival is decided solely by the callback. A fall will therefore not
  auto-reset — tune thresholds back down if you want that.
- **Shared output folder (uid mismatch).** The container process runs as uid 1000
  while the mounted host files are owned by uid 1001, so eval cannot write under
  host-owned paths (`logs_eval/`, the checkpoint dir, ...). `run_ardy_sonic.sh`
  redirects *all* eval output into one world-writable folder `SHARED_IO`
  (default `/workspace/shared_io` == host `/home/hslee/IsaacLab_ws/shared_io`): the
  hydra run dir, eval logs, renderings, a symlink-staged copy of the checkpoint
  (so `experiment_dir = checkpoint.parent` and the `model_step_*.pt` it writes land
  in a writable place), **and the Ardy↔SONIC request/response dir**
  (`${SHARED_IO}/runtime`, since the default `ardy_sonic/runtime` is host-owned and
  not container-writable). Both `run_ardy_sonic.sh` and `run_planner.sh` point
  `ARDY_SONIC_RUNTIME` at that one physical dir (container path
  `/workspace/shared_io/runtime`, host path `<IsaacLab_ws>/shared_io/runtime`).
  `umask 000` makes everything 0777 so host and container can both read/write.
  Override the location with `SHARED_IO=/some/other/dir` (and matching
  `ARDY_SONIC_RUNTIME` on the planner side).
- **Heading discontinuity.** A fresh plan starts from Ardy's canonical standing pose at
  the robot's current xy, facing the goal. Large heading changes between segments show
  up as a turn in the reference; SONIC absorbs moderate turns. Very sharp turns are a
  known rough edge.

## Offline test (no Isaac, no Ardy)
```bash
# in the container:
/workspace/isaaclab/isaaclab.sh -p ardy_sonic/tests/test_bridge_offline.py
```
Drives the real callback + protocol against a fake planner and a fake drifting robot,
and asserts the loop converges to the goal.
