---
license: other
license_name: bones-seed-license
license_link: https://bones.studio/info/seed-license
task_categories:
  - robotics
  - text-to-video
  - video-text-to-text
tags:
  - motion-capture
  - humanoid-robotics
  - human-motion
  - physical-ai
  - whole-body-control
  - NVIDIA-SOMA
  - Unitree-G1
  - BVH
  - MuJoCo
  - language-to-action
  - locomotion
  - gesture
  - dance
  - object-interaction
  - multimodal
  - annotated
pretty_name: "BONES-SEED: Skeletal Everyday Embodiment Dataset"
size_categories:
  - 100K<n<1M
language:
  - en
configs:
  - config_name: default
    data_files:
      - split: metadata
        path: metadata/seed_metadata_v003.parquet
extra_gated_prompt: "Please provide the following information to access BONES-SEED. By checking the box below, you confirm that you have read the
  [BONES-SEED License](https://bones.studio/info/seed-license). For licensing inquiries, contact licensing@bones.studio."
extra_gated_fields:
  Name: text
  Surname: text
  Affiliation:
    type: select
    options:
      - Academia
      - Industry
  Company/institution: text
  Professional email: text
  Tell us how you are going to use the data: text
  I confirm that I agree with the BONES-SEED license and I qualify as an Academic User (non-commercial, publicly available research at a non-profit institution) or that my company's current annual gross revenue is less than 1,000,000 USD: checkbox
  I want to hear from Bones about new human motion data releases: checkbox

---

<img src="https://media.bones.studio/bones-seed-logo.png"></img>
<video src="https://media.bones.studio/BONES_SEED_HUM2ROBOT_Multicam_Mosaic_tkozpf.mp4" controls autoplay muted loop></video>


# BONES-SEED: Skeletal Everyday Embodiment Dataset
BONES-SEED is an open dataset of 142,220 annotated human motion animations for humanoid robotics. It provides motion capture data in [SOMA](https://github.com/NVlabs/SOMA-X) and Unitree G1 formats, with natural language descriptions, temporal segmentation, and detailed skeletal metadata.

- **Project website:** [bones.studio/datasets/seed](https://bones.studio/datasets/seed)
- **Interactive viewer:** [seed-viewer.bones.studio](https://seed-viewer.bones.studio/)
- **Associated code:** [github.com/bones-studio/seed-viewer](https://github.com/bones-studio/seed-viewer)

| | |
|---|---|
| **Total motions** | 142,220 (71,132 original + 71,088 mirrored) |
| **Total duration** | ~288 hours (@ 120 fps) |
| **Performers** | 522 actors (253 F / 269 M) |
| **Age range** | 17–71 years |
| **Height range** | 145–199 cm |
| **Weight range** | 38–145 kg |
| **Output formats** | SOMA Uniform · SOMA Proportional · Unitree G1 MuJoCo-compatible |
| **Annotation depth** | Up to 6 NL descriptions per motion + temporal segmentation + technical descriptions + skeletal metadata |

## Intended Uses

BONES-SEED is designed to support research and development in:

- **Humanoid whole-body control** — training language-conditioned policies for humanoid robots
- **Motion generation** — text-to-motion and action-to-motion synthesis
- **Motion retrieval** — natural language search over large motion libraries
- **Sim-to-real transfer** — leveraging MuJoCo-compatible G1 trajectories for simulation training
- **Imitation learning** — learning from diverse human demonstrations
- **Motion understanding** — temporal segmentation, style classification, and activity recognition

## Download

BONES-SEED is hosted on Hugging Face and can be downloaded using any of the methods below.

### Using the Hugging Face Hub

Browse and download files directly from the dataset repository:

> [https://huggingface.co/datasets/bones-studio/seed](https://huggingface.co/datasets/bones-studio/seed)

### Using Git LFS

```bash
# Make sure Git LFS is installed
git lfs install

# Clone the full dataset
git clone https://huggingface.co/datasets/bones-studio/seed
```

### Using the Hugging Face CLI

```bash
# Install the Hugging Face CLI if you haven't already
pip install huggingface_hub

# Download the full dataset
huggingface-cli download bones-studio/seed --repo-type dataset --local-dir ./bones-seed
```

### Using Python

```python
from huggingface_hub import snapshot_download

# Download the full dataset
snapshot_download(
    repo_id="bones-studio/seed",
    repo_type="dataset",
    local_dir="./bones-seed"
)
```

### Loading Metadata Only

```python
import pandas as pd

# Load directly from Hugging Face
df = pd.read_parquet(
    "hf://datasets/bones-studio/seed/metadata/seed_metadata_v002.parquet"
)
print(f"Total motions: {len(df)}")
print(f"Columns: {df.columns.tolist()}")
```

## Dataset Structure

After downloading and extracting, the dataset is organized as follows:

```
bones-seed/
├── metadata/
│   ├── seed_metadata_v003.parquet          # Main metadata (51 columns × 142,220 rows)
│   ├── seed_metadata_v003.csv              # Same metadata in CSV format
│   └── seed_metadata_v002_temporal_labels.jsonl  # Temporal segmentation labels
├── soma_uniform/
│   └── bvh/{date}/{motion_name}.bvh        # SOMA Uniform motion files
├── soma_proportional/
│   └── bvh/{date}/{motion_name}.bvh        # SOMA Proportional motion files
├── g1/
│   └── csv/{date}/{motion_name}.csv        # Unitree G1 MuJoCo-compatible joint trajectories
├── soma_shapes/
│   ├── soma_base_fit_mhr_params.npz        # Shared shape params (SOMA Uniform)
│   ├── soma_proportion_fit_mhr_params/
│   │   └── {actor_id}.npz                  # Per-actor shape params (SOMA Proportional)
│   └── soma_base_rig/
│       ├── soma_base_skel_minimal.bvh      # SOMA base skeleton definition (BVH)
│       └── soma_base_skel_minimal.usd      # SOMA base skeleton definition (USD)
└── LICENSE.md
```

### Unpacking

The motion data directories (`soma_uniform/`, `soma_proportional/`, `g1/`) are distributed as tar archives. After downloading, extract them into the dataset root:

```bash
tar -xzf soma_uniform.tar.gz
tar -xzf soma_proportional.tar.gz
tar -xzf g1.tar.gz
```

## Motion Categories

BONES-SEED spans a wide range of human activities organized into 8 top-level packages and 20 fine-grained categories.

### Packages

| Package | Motions | Description |
|---|---|---|
| Locomotion | 74,488 | Walking, jogging, jumping, climbing, crawling, turning, and transitions |
| Communication | 21,493 | Gestures, pointing, looking, and communicative body language |
| Interactions | 14,643 | Object manipulation, pick-and-place, carrying, and tool use |
| Dances | 11,006 | Full-body dance performances across multiple styles |
| Gaming | 8,700 | Game-inspired actions and dynamic movements |
| Everyday | 5,816 | Household tasks, consuming, sitting, reading, and daily activities |
| Sport | 3,993 | Athletic movements and sports-specific actions |
| Other | 2,081 | Stunts, martial arts, magic, and edge-case motions |

### Categories

| Category | Motions |
|---|---|
| Basic Locomotion Neutral | 33,430 |
| Baseline | 22,878 |
| Gestures | 17,590 |
| Object Manipulation | 11,620 |
| Dancing | 11,006 |
| Object Interaction | 10,817 |
| Basic Locomotion Styles | 10,746 |
| Advanced Locomotion | 6,036 |
| Sports | 3,973 |
| Communication | 3,723 |
| Unusual Locomotion | 3,242 |
| Other | 2,081 |
| Consuming | 1,388 |
| Household | 1,318 |
| Stunts | 858 |
| Environments | 614 |
| Complex Actions | 540 |
| Looking and Pointing | 180 |
| Magic | 160 |
| Martial Arts | 20 |

## Data Formats

Every motion is provided in three skeletal representations supporting two character models: [SOMA](https://github.com/NVlabs/SOMA-X) and Unitree G1 robot. SOMA is a canonical body topology and rig that acts as a universal pivot for parametric human body models.

### SOMA Proportional (BVH)

A per-actor skeleton that preserves the original performer's body proportions. Each actor has an individual shape file.

```
soma_proportional/bvh/{date}/{motion_name}.bvh
soma_shapes/soma_proportion_fit_mhr_params/{actor_id}.npz
```

### SOMA Uniform (BVH)

A standardized skeleton shared across all motions, enabling direct comparison and batch processing. Each motion file is paired with a single shared shape file. The base skeleton definition is provided in both BVH and USD formats.

```
soma_uniform/bvh/{date}/{motion_name}.bvh
soma_shapes/soma_base_fit_mhr_params.npz
soma_shapes/soma_base_rig/soma_base_skel_minimal.bvh
soma_shapes/soma_base_rig/soma_base_skel_minimal.usd
```

### Unitree G1 MuJoCo-compatible (CSV)

Joint-angle trajectories retargeted to the Unitree G1 humanoid robot.

```
g1/csv/{date}/{motion_name}.csv
```

## Annotations

Each motion in BONES-SEED comes with rich multimodal annotations designed for language-conditioned policy learning, motion retrieval, and motion generation.

### Natural Language Descriptions

Every motion includes up to **6 natural language descriptions** at varying levels of detail:

- **Natural descriptions (4):** Fluent, human-written descriptions from different perspectives
- **Technical description (1):** Precise biomechanical description of the motion
- **Short descriptions (2):** Concise labels for indexing and retrieval

**Example — `read_newspaper_sitting`:**

| Field | Text |
|---|---|
| `content_natural_desc_1` | character reading newspaper while sitting |
| `content_natural_desc_2` | person reads a newspaper while sitting |
| `content_natural_desc_3` | individual sits and reads a newspaper |
| `content_natural_desc_4` | A person sitting reads a newspaper, holding it with both hands, moving pages and folding the newspaper. |
| `content_technical_description` | reading a newspaper holding it with both hands while sitting, moving pages folding a newspaper |
| `content_short_description` | reading newspaper sitting |

### Temporal Segmentation Labels

Each motion includes temporal segmentation that breaks the full sequence into meaningful phases with precise timestamps and natural language descriptions. These labels were created by NVIDIA for the [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) project and are stored in `metadata/seed_metadata_v002_temporal_labels.jsonl` (one JSON object per line).

**Schema:**

| Field | Type | Description |
|---|---|---|
| `filename` | string | Motion filename (matches `filename` column in metadata) |
| `num_events` | int | Number of temporal segments |
| `events` | array | Ordered list of temporal segments |
| `events[].start_time` | float | Segment start time in seconds |
| `events[].end_time` | float | Segment end time in seconds |
| `events[].description` | string | Natural language description of the segment |

**Example — `inside_door_knob_left_side_open_R_002__A512`:**

```json
{
  "filename": "inside_door_knob_left_side_open_R_002__A512",
  "num_events": 3,
  "events": [
    {"start_time": 0.0, "end_time": 1.88, "description": "A person rotates the door knob with their right hand."},
    {"start_time": 1.88, "end_time": 3.53, "description": "A person opens the door outward from the inside, holding the knob and then lowers their hand."},
    {"start_time": 3.53, "end_time": 4.83, "description": "A person is standing idle and slightly moving their right hand."}
  ]
}
```

**Loading temporal labels:**

```python
import json

temporal_labels = {}
with open("metadata/seed_metadata_v002_temporal_labels.jsonl") as f:
    for line in f:
        entry = json.loads(line)
        temporal_labels[entry["filename"]] = entry["events"]

# Look up segments for a specific motion
events = temporal_labels["inside_door_knob_left_side_open_R_002__A512"]
for event in events:
    print(f"[{event['start_time']:.2f}s - {event['end_time']:.2f}s] {event['description']}")
```

### Motion Properties

Each motion is tagged with structured metadata for filtering and analysis:

| Field | Description | Example Values |
|---|---|---|
| `content_type_of_movement` | Primary movement type | walking, jogging, gesture, dancing, jumping |
| `content_body_position` | Starting/primary body position | standing, sitting on floor, crouching, crawling |
| `content_uniform_style` | Performance style | neutral, injured leg, injured torso, hurry, old |
| `content_horizontal_move` | Horizontal displacement flag | 0 or 1 |
| `content_vertical_move` | Vertical displacement flag | 0 or 1 |
| `content_props` | Involves props/objects | 0 or object descriptor |
| `content_complex_action` | Multi-phase complex action | 0 or 1 |
| `content_repeated_action` | Contains repeated cycles | 0 or 1 |

## Metadata Schema

The metadata parquet file contains **51 columns** organized into five groups.

### Motion Identity

| Column | Type | Description |
|---|---|---|
| `move_name` | string | Unique motion identifier |
| `filename` | string | Base filename (without extension) |
| `move_duration_frames` | int | Duration in frames (@ 120 fps) |
| `package` | string | Top-level category (Locomotion, Communication, etc.) |
| `category` | string | Fine-grained category |
| `is_neutral` | float | Whether the motion uses a neutral performance style |
| `is_mirror` | bool | Whether the motion is a left-right mirror |

### File Paths

| Column | Type | Description |
|---|---|---|
| `move_soma_uniform_path` | string | Path to SOMA Uniform BVH file |
| `move_soma_uniform_shape_path` | string | Path to SOMA Uniform shape parameters |
| `move_soma_proportional_path` | string | Path to SOMA Proportional BVH file |
| `move_soma_proportional_shape_path` | string | Path to SOMA Proportional shape parameters |
| `move_g1_mujoco_path` | string | Path to Unitree G1 MuJoCo-compatible CSV file |

### Capture Session

| Column | Type | Description |
|---|---|---|
| `take_name` | string | Capture session identifier |
| `take_actor` | string | Actor identifier for this take |
| `take_org_name` | string | Original take name |
| `take_date` | int | Capture date (YYMMDD format) |
| `take_day_part` | string | Part of capture day |

### Content Annotations

| Column | Type | Description |
|---|---|---|
| `content_name` | string | Semantic motion name |
| `content_natural_desc_1` | string | Natural language description 1 |
| `content_natural_desc_2` | string | Natural language description 2 |
| `content_natural_desc_3` | string | Natural language description 3 |
| `content_natural_desc_4` | string | Natural language description 4 |
| `content_technical_description` | string | Technical/biomechanical description |
| `content_short_description` | string | Short description 1 |
| `content_short_description_2` | string | Short description 2 |
| `content_all_rigplay_styles` | string | All performance styles applied |
| `content_uniform_style` | string | Normalized style label |
| `content_type_of_movement` | string | Movement type classification |
| `content_body_position` | string | Body position classification |
| `content_horizontal_move` | int | Horizontal displacement flag |
| `content_vertical_move` | int | Vertical displacement flag |
| `content_props` | string | Props/objects involved |
| `content_complex_action` | int | Complex action flag |
| `content_repeated_action` | int | Repeated action flag |

### Actor Biometrics

| Column | Type | Description |
|---|---|---|
| `actor_uid` | string | Unique actor identifier |
| `actor_height` | string | Height category (S / M / T) |
| `actor_height_cm` | int | Height in centimeters |
| `actor_foot_cm` | int | Foot length in cm |
| `actor_collarbone_height_cm` | int | Collarbone height in cm |
| `actor_collarbone_span_cm` | int | Collarbone span in cm |
| `actor_elbow_span_cm` | int | Elbow span in cm |
| `actor_wrist_span_cm` | int | Wrist span in cm |
| `actor_shoulder_span_cm` | int | Shoulder span in cm |
| `actor_hips_height_cm` | int | Hips height in cm |
| `actor_hips_bones_span_cm` | int | Hips bone span in cm |
| `actor_knee_height_cm` | int | Knee height in cm |
| `actor_ankle_height_cm` | int | Ankle height in cm |
| `actor_weight_kg` | int | Weight in kilograms |
| `actor_age_yr` | int | Age in years |
| `actor_gender` | string | Gender (F / M) |
| `actor_profession` | string | Performer background (actor, dancer, stuntman, general, professional) |

## About Bones Studio

With over 5 years of experience, [Bones Studio](https://bones.studio) builds enterprise-grade, multimodal datasets of human behavior and motion for AI and robotics. BONES-SEED represents a curated subset of Bones Studio's broader motion capture library, with expanded datasets available for commercial licensing.

Learn more: [bones.studio/datasets](https://bones.studio/datasets)

## Acknowledgments

Thanks to NVIDIA for providing the [SOMA](https://github.com/NVlabs/SOMA-X) and G1 retargets, and for creating the temporal segmentation labels as part of the [Kimodo](https://research.nvidia.com/labs/sil/projects/kimodo/) project.
---