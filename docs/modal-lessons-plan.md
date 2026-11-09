# Run each lesson on Modal

Use `modal_lessons.py` as a thin set of named Modal functions over the existing
numbered Python lessons. Reuse the original pinned CUDA image and named volume.
CPU functions cover reading/data/prompt/reward/metrics lessons; BF16 model execution
and GPU preflight use the original B300 resources.

All experiment files live at `/data/runs/<run-name>`; reload the mounted volume
before resolving commands and commit it after child exit, including failures.
Run each lesson in a subprocess so its argv and model resources do not leak into
subsequent invocations. Select the latest complete checkpoint automatically and
reuse the saved scheduler horizon and baseline evaluation size.

- [x] Test command routing, persistent paths, resume selection, invalid arguments
  and commit-on-failure behavior using the standard-library test suite.
- [x] Add the runner, Modal functions and a runnable performance-reading lesson.
- [x] Document setup, all per-lesson commands and artifact download/resume.
- [x] Run offline tests and SDK CLI help when available; do not launch paid jobs.
