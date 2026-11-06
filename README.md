# My harness for VLA-Cache — stall-triggered refresh

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2502.02175). The evaluated behavior is in [`my_harness.py`](my_harness.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | 83.8% LIBERO-Spatial |

| Our run of the authors' harness | 103/120 (85.83%) |

| **My harness** | **104/120 (86.67%)** |


Ten reported Spatial tasks, twelve official starts each (indices 20–31); bare-policy control 102/120. Separate development index 40.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Monitor public EEF/finger motion and the policy's recently executed movement commands.

2. After eight motion commands with little observed movement, bypass visual-token reuse for three native policy calls.

3. Return to the authors' caching with a bounded cooldown; never replace actions or add a model call.


Categories: `execution_monitoring`, `contact_feedback`, `uncertainty_compute`.

## Models and inputs

- openvla/openvla-7b-finetuned-libero-spatial at 962318cec55ac10993ff0f5f43eda9a270b4c873, original BF16 weights.

- Exact processor source, modified Transformers revision and LIBERO revisions are recorded in provenance.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/siyuhsu/vla-cache) before installing its model stack.
2. Controller dependency: numpy==1.26.4
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 -m pip install "numpy==1.26.4"
python3 tools/verify_export.py
python -m unittest discover -s tests -v
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

Instantiate Controller("stall_refresh"). Call before(rgb224, public_eef_position, public_fingers) before native inference; when True, clear the authors' visual cache. Feed the native action/cropped image/reuse flag back through after(). reference/runtime.py contains the exact integration.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

220 controlled actions plus ten native setup no-ops; no multiplier.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `my_harness.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- Only one helped case and zero harmed; McNemar p=1.0. No statistically established gain.

- Whole-policy time was 87.44 versus 87.84 minutes; no measured speed improvement.

- The completed 120-case results use the original runner. The later HarnessStudio migration is a separate, incomplete cohort.

- Fresh states on known tasks do not establish unseen-task generalization.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

2 CPU tests passed. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
