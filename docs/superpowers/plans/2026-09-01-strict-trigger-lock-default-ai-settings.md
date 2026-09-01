# Strict Trigger Lock and Recommended AI Defaults Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the balanced AI settings the safe defaults and guarantee that one raw-Trigger press performs at most one target acquisition, failing closed without replacement until a new Trigger press.

**Architecture:** Add a separate immutable strict-lock policy beside the legacy pure tracker, then let `AiService` apply it only when the production UI supplies a runtime Trigger-epoch provider. The Tk main thread owns eligible raw-Trigger epochs and immediate invalidation; the inference worker owns per-frame association while the service owns the cross-generation one-claim record. Overlay visualization, Adaptive Zoom refinement, and Makcu movement publication remain separate so loss suppresses AI movement without suppressing Jitter or accepted boxes.

**Tech Stack:** Python 3, `dataclasses`, NumPy, Tkinter, ONNX Runtime DirectML, DXCam, Makcu, and `unittest`; no new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-01-strict-trigger-lock-default-ai-settings-design.md`

## Global Constraints

- Windows-only standalone Jitter repository; do not import or read EverFall Jitter source or configuration.
- Keep Tk widget and Tk-variable access on the main thread; worker providers read immutable snapshots under short locks.
- Keep strict association, target selection, Adaptive Zoom geometry, and combined movement pure and independent of Tkinter and Makcu.
- Preserve generation identifiers, targeting revisions, daemon workers, stop events, and immediate STOP/disconnect/source-change cancellation.
- One eligible raw-Trigger press may claim at most one target; zero or non-unique continuation latches `LOST` until raw Trigger-up followed by Trigger-down.
- The detector has no identity field. Promise no reacquisition/no observable replacement, not unknowable physical identity through merged output.
- Modifier changes never create a Trigger epoch. Test 3s uses one synthetic epoch per AI-involving run.
- Keep both approved ONNX contracts, both capture modes, bundled `models/all_games_320.onnx`, 1,000 Hz time-based servo, 150 ms freshness, and existing Adaptive Zoom cooldown.
- Use defaults `confidence=.25`, `aim_strength=.35`, `smoothing=.58`, `max_step=18`, and curve `(0,.16,.38,.68,.95)`; keep `Center 320` as startup capture mode.
- Keep schema 5. Do not migrate valid schema-5 values, create schema 6, persist runtime lock state, or edit local `config.json`.
- Do not add Torch, Ultralytics, OpenCV, Pillow, Pystray, alternate runtimes, training, profiles, downloads, copying, tray behavior, or additional bundled models.
- Preserve the five untracked external `.onnx` files; never stage, edit, delete, copy, or package them.
- Use `apply_patch` for source/document edits and TDD for every behavior change.
- Do not run Nuitka unless the user explicitly requests a packaged build.

## File and interface map

- `jitter_app/ai/targeting.py`: owns the new scalar and response-curve defaults only; stateless acquisition helpers remain compatible.
- `jitter_app/ai/tracking.py`: retains `TrackerState`/`observe_detections` unchanged and adds `StrictTriggerLockState`, `StrictTriggerLockObservation`, and `observe_strict_trigger_lock`.
- `jitter_app/ai/zoom.py`: makes multiple compatible refinement candidates fail back to the locked base result.
- `jitter_app/ai/service.py`: adds optional managed Trigger epochs, atomic epoch claim/invalidation, strict worker integration, and separate movement/Overlay publication.
- `jitter_app/presentation/ui.py`: creates eligible raw-Trigger epochs, exposes a thread-safe provider, creates synthetic Test epochs, and invalidates every lifecycle edge.
- `tests/test_ai_targeting.py`, `tests/test_settings.py`, `tests/test_ui.py`: verify defaults, persistence compatibility, and UI startup controls.
- `tests/test_ai_tracking.py`: verifies the pure strict state machine independently of threads and hardware.
- `tests/test_ai_zoom.py`, `tests/test_ai_service.py`: verify refinement ambiguity, worker races, managed/unmanaged behavior, and generation safety.
- `AGENTS.md`, `README.md`: replace the superseded stateless movement rule and document new defaults/Trigger behavior.

## Subagent execution map

- This map applies when the user selects the recommended Subagent-Driven handoff. The required plan header continues to offer Inline Execution; if that option is selected, `superpowers:executing-plans` runs the same task barriers serially and does not dispatch this map.
- Use `superpowers:subagent-driven-development` for its ledger, task briefs, fresh task contexts, fix loops, per-task reviews, and final broad review. The user's explicit three-subagent speed request overrides only that skill's serial-implementer rule for the initial independent file sets below; use `superpowers:dispatching-parallel-agents` for that one parallel wave.
- Interpret “three subagents” as three concurrent worker seats: start exactly three in the initial wave and never exceed three active subagents. Fresh implementers and reviewers required by the SDD protocol rotate through freed seats, so the plan preserves fresh task context without exceeding the requested parallelism.
- “Worker A/B/C” names implementation lanes and file ownership, not permission to reuse contaminated context: Task 4 and Task 5 receive fresh agents even when they occupy a prior lane's worktree.
- Worker A owns Task 1 (defaults/config) and later Task 5 (active documentation). Worker B owns Task 2 (pure strict lock). Worker C owns Task 3 (service/zoom integration) and Task 4's files.
- Start three fresh implementers together: Worker A executes Task 1, Worker B executes Task 2, and Worker C executes only Task 3 Steps 1-4 to create and confirm its failing service/zoom tests. Worker C must not edit production service/zoom code until Task 2 is reviewed and integrated.
- Review and integrate Tasks 1 and 2 independently. Rebase Worker C's test-only branch onto that exact integrated head, then resume it for Task 3 Steps 5-10. Task 4 starts only after Task 3 review/integration. Task 5 uses a fresh worktree from the reviewed Task 4 head.
- Each task's fresh reviewer must return both a requirements/spec verdict and a code-quality verdict before integration. Task 6 uses one fresh whole-branch reviewer. Review agents may occupy free slots but never raise active subagents above three.
- Corrections return to the lane owning the production file: A for defaults/config/docs, B for pure tracking, C for service/zoom/UI. The coordinating agent does not make implementation corrections directly.

| Gate | Implementer lane | Fresh review required | Integration/dependency barrier |
| --- | --- | --- | --- |
| Task 0 | Coordinator, read-only setup | Preflight table in SDD ledger | No implementation dispatch before baseline/worktrees pass |
| Task 1 | A | Spec verdict + quality verdict | Cherry-pick only after both pass |
| Task 2 | B | Spec verdict + quality verdict | Cherry-pick only after both pass; Task 3 production waits |
| Task 3 | C | Spec verdict + quality verdict | Fast-forward only after both pass; Task 4 waits |
| Task 4 | Fresh agent in C lane | Spec verdict + quality verdict | Fast-forward only after both pass; Task 5 waits |
| Task 5 | Fresh agent in A lane | Spec verdict + quality verdict | Fast-forward `codex/strict-lock-docs` only after both pass |
| Task 6 | Coordinator runs commands | Fresh whole-branch reviewer | Any correction returns to A/B/C ownership and receives scoped re-review |

---

### Task 0: Controller preflight, worktrees, and protected-model baseline

**Files:**
- Verify tracked: `docs/superpowers/plans/2026-09-01-strict-trigger-lock-default-ai-settings.md`
- Create through the SDD skill: `.superpowers/sdd/2026-09-01-strict-trigger-lock-default-ai-settings/progress.md`
- Read only: the five exact protected external model paths listed below
- Do not copy into worktrees: any protected external model

**Interfaces:**
- Consumes: the committed approved spec/plan and the current integration `HEAD`.
- Produces: one recorded integration base, one protected-model integrity baseline, three isolated absolute worktree paths, and reviewed commit SHAs for deterministic handoffs.

- [ ] **Step 1: Verify the plan is tracked and record the integration base**

Run from `C:\Users\User\Desktop\Jitter`:

```powershell
git ls-files --error-unmatch -- docs/superpowers/plans/2026-09-01-strict-trigger-lock-default-ai-settings.md
git rev-parse HEAD
git status --short --branch
```

Expected: the plan path is tracked; record the exact `HEAD` as `INTEGRATION_BASE` in this plan's SDD ledger. The planning session normally satisfies this with the docs-only commit `docs: plan strict trigger target lock` before offering execution. If `git ls-files` fails, do not dispatch implementation: from the primary checkout stage only this exact plan, verify the staged list, commit it, and then record the resulting `HEAD`:

```powershell
git add -- docs/superpowers/plans/2026-09-01-strict-trigger-lock-default-ai-settings.md
git diff --cached --name-only
git diff --cached --check
git commit -m "docs: plan strict trigger target lock"
git rev-parse HEAD
```

The staged list must contain exactly `docs/superpowers/plans/2026-09-01-strict-trigger-lock-default-ai-settings.md`. After that precondition, the only untracked paths are the five protected models. If any other source/doc/test path is dirty, preserve it and resolve scope before creating worktrees.

- [ ] **Step 2: Record an exact protected-model integrity baseline**

Use this exact read-only PowerShell command:

```powershell
$protectedModels = @(
    'models/Apex_20k_pictures_640.onnx',
    'models/all_games.onnx',
    'models/all_games_128.onnx',
    'models/all_games_256.onnx',
    'models/all_games_640.onnx'
)
$protectedModels | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    $hash = Get-FileHash -LiteralPath $_ -Algorithm SHA256
    [pscustomobject]@{
        Path = $_
        Length = $item.Length
        LastWriteTimeUtc = $item.LastWriteTimeUtc.ToString('O')
        SHA256 = $hash.Hash
    }
} | ConvertTo-Json
```

Copy the exact emitted JSON into the ignored SDD ledger using `apply_patch`. Never use `git clean` in the primary checkout, never stage a protected model, and never copy one into a task worktree. Re-run the same command after every integration and at Task 6; exact paths, lengths, timestamps, and hashes must match the ledger baseline. If they do not, stop and report the external change without restoring or overwriting it.

- [ ] **Step 3: Create the three isolated worktrees through the required skill**

Invoke `superpowers:using-git-worktrees`, then create/verify these exact branches and absolute paths from `INTEGRATION_BASE`:

```text
Worker A: branch codex/strict-lock-defaults
          C:\Users\User\Desktop\Jitter-worktrees\strict-lock-defaults
Worker B: branch codex/strict-lock-tracking
          C:\Users\User\Desktop\Jitter-worktrees\strict-lock-tracking
Worker C: branch codex/strict-lock-service
          C:\Users\User\Desktop\Jitter-worktrees\strict-lock-service
```

Verify every exact path:

```powershell
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-defaults' status --short --branch
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-defaults' rev-parse HEAD
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-tracking' status --short --branch
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-tracking' rev-parse HEAD
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-service' status --short --branch
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-service' rev-parse HEAD
```

Every worktree must be clean and equal the `INTEGRATION_BASE` recorded in the ledger. The protected external models remain only in the primary checkout.

- [ ] **Step 4: Dispatch the initial three-agent wave and enforce review barriers**

Generate one SDD task brief/report path per assignment. Dispatch Task 1 to Worker A, Task 2 to Worker B, and Task 3 Steps 1-4 only to Worker C. Each implementer must commit only its enumerated files and report the exact red/green commands it ran.

As A and B finish, generate their review packages from `INTEGRATION_BASE` to their reported heads. Dispatch fresh reviewers and require both verdicts. Route findings back through the SDD fix loop. Do not integrate an unreviewed commit and do not integrate Worker C's deliberately red tests.

- [ ] **Step 5: Integrate reviewed A/B commits and hand the exact head to Worker C**

In the primary integration checkout, cherry-pick only reviewed Task 1 commit SHAs, then only reviewed Task 2 commit SHAs. After each cherry-pick run its focused green suite, the protected-model command from Step 2, and `git status --short --branch`.

Record the resulting `TASK_1_2_HEAD`, then run:

```powershell
$task12Head = (git rev-parse HEAD).Trim()
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-service' rebase $task12Head
git -C 'C:\Users\User\Desktop\Jitter-worktrees\strict-lock-service' diff --name-only $task12Head HEAD
```

The diff list must contain only `tests/test_ai_service.py` and `tests/test_ai_zoom.py` before permitting Task 3 Steps 5-10. After Task 3 passes review, the primary branch at `TASK_1_2_HEAD` integrates Worker C with:

```powershell
git merge --ff-only codex/strict-lock-service
```

- [ ] **Step 6: Create downstream fresh contexts from reviewed integration heads**

After Task 3 integration, dispatch a fresh Task 4 implementer in the clean Worker C worktree at that exact head. After Task 4 review, integrate with `git merge --ff-only codex/strict-lock-service`.

Then use `superpowers:using-git-worktrees` to create a fresh Task 5 branch `codex/strict-lock-docs` at:

```text
C:\Users\User\Desktop\Jitter-worktrees\strict-lock-docs
```

Its base must equal the reviewed Task 4 integration head. Dispatch a fresh Worker A-context implementer for Task 5. Task 6 runs only after Task 5 review/integration.

After Task 5 receives both review verdicts, integrate it from the primary checkout and recheck the protected-model baseline:

```powershell
git merge --ff-only codex/strict-lock-docs
git status --short --branch
```

---

### Task 1: Adopt the balanced AI defaults without migrating schema 5

**Files:**
- Modify: `jitter_app/ai/targeting.py:16-18,63-69`
- Modify: `tests/test_ai_targeting.py:26-74,150-370`
- Modify: `tests/test_settings.py:199-355,420-466`
- Modify: `tests/test_ui.py:662-712,2150-2425`

**Interfaces:**
- Consumes: existing `AimSettings`, `DEFAULT_RESPONSE_CURVE`, `aim_settings_from_mapping`, `aim_settings_to_mapping`, and schema-5 `ConfigStore` behavior.
- Produces: exact `AimSettings()` defaults `(0.25, 0.35, 0.58, 18, (0.0, 0.16, 0.38, 0.68, 0.95), "head")`; no schema/API change.

- [ ] **Step 1: Write failing exact-default and threshold tests**

Add to `AimSettingsTests` in `tests/test_ai_targeting.py`:

```python
def test_recommended_ai_defaults_are_exact(self):
    self.assertEqual(DEFAULT_RESPONSE_CURVE, (0.0, 0.16, 0.38, 0.68, 0.95))
    self.assertEqual(
        AimSettings(),
        AimSettings(
            confidence=0.25,
            aim_strength=0.35,
            smoothing=0.58,
            max_step=18,
            response_curve=(0.0, 0.16, 0.38, 0.68, 0.95),
            target_area="head",
        ),
    )

def test_default_confidence_boundary_is_inclusive(self):
    accepted = select_target(
        (Detection(150, 150, 170, 170, 0.25, 7),),
        AimSettings(), sequence=1, captured_at=1.0,
    )
    rejected = select_target(
        (Detection(150, 150, 170, 170, 0.249, 7),),
        AimSettings(), sequence=2, captured_at=2.0,
    )
    self.assertIsNotNone(accepted)
    self.assertIsNone(rejected)
```

Change `Detection(0, 0, 10, 10, 0.34, 7)` in `TargetSelectionTests.test_rejects_below_confidence` to confidence `0.24` so it still tests filtering rather than the old default.

- [ ] **Step 2: Write a failing valid-schema-5 non-migration regression**

Add to `tests/test_settings.py`:

```python
def test_schema_five_explicit_old_defaults_are_not_migrated(self):
    old_ai = {
        "confidence": "0.35",
        "aim_strength": "0.35",
        "smoothing": "0.65",
        "max_step": "20",
        "response_curve": ["0", "0.12", "0.35", "0.68", "1"],
    }
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "config.json"
        path.write_text(
            json.dumps({"schema_version": 5, "ai": old_ai}),
            encoding="utf-8",
        )
        restored = ConfigStore(path).load().config.ai

    self.assertEqual(
        restored,
        AimSettings(0.35, 0.35, 0.65, 20, (0.0, 0.12, 0.35, 0.68, 1.0)),
    )
```

Add a UI startup assertion using a fresh `AppConfig()`:

```python
def test_ai_controls_show_recommended_defaults(self):
    self.assertEqual(self.app.ai_vars["confidence"].get(), "0.25")
    self.assertEqual(self.app.ai_vars["aim_strength"].get(), "0.35")
    self.assertEqual(self.app.ai_vars["smoothing"].get(), "0.58")
    self.assertEqual(self.app.ai_vars["max_step"].get(), "18")
    self.assertEqual(
        tuple(self.app.get_ai_settings().response_curve),
        (0.0, 0.16, 0.38, 0.68, 0.95),
    )
```

- [ ] **Step 3: Run the focused tests and confirm the expected failures**

Run:

```powershell
python -m unittest tests.test_ai_targeting.AimSettingsTests.test_recommended_ai_defaults_are_exact -v
python -m unittest tests.test_ai_targeting.AimSettingsTests.test_default_confidence_boundary_is_inclusive -v
python -m unittest tests.test_settings.ConfigStoreTests.test_schema_five_explicit_old_defaults_are_not_migrated -v
python -m unittest tests.test_ui.JitterLayoutTests.test_ai_controls_show_recommended_defaults -v
```

Expected: the exact-default and UI tests fail against `0.35/0.65/20/(0,.12,.35,.68,1)`; the explicit-schema test already passes and protects the no-migration rule.

- [ ] **Step 4: Change only the canonical defaults**

Update `jitter_app/ai/targeting.py`:

```python
DEFAULT_RESPONSE_CURVE = (0.0, 0.16, 0.38, 0.68, 0.95)

@dataclass(frozen=True)
class AimSettings:
    confidence: float = 0.25
    aim_strength: float = 0.35
    smoothing: float = 0.58
    max_step: int = 18
    response_curve: tuple[float, float, float, float, float] = DEFAULT_RESPONSE_CURVE
    target_area: str = "head"
```

Do not change `AIM_LIMITS`, `SCHEMA_VERSION`, `ConfigStore`, capture defaults, or local `config.json`.

- [ ] **Step 5: Update expectations that intentionally serialize or display defaults**

Use these exact replacements:

```python
# compact default curve
["0", "0.16", "0.38", "0.68", "0.95"]

# schema-three boolean field fallbacks
AimSettings(0.25, 1.25, 0.58, 30)

# default curve entry percentages
{1: "16", 2: "38", 3: "68", 4: "95"}

# curve edit from a scalar-explicit AimSettings using its default curve
AimSettings(0.5, 0.6, 0.7, 30, (0.0, 0.16, 0.42, 0.68, 0.95))

# exact zero/hundred boundary edit
(0.0, 0.0, 0.38, 0.68, 1.0)

# ordered manual edits of nodes 1 and 3
(0.0, 0.3, 0.38, 0.7, 0.95)

# lower drag clamp for node 2
0.16
```

Keep the old tuple in interpolation tests that deliberately exercise an arbitrary valid curve. Keep explicit non-default settings unchanged.

- [ ] **Step 6: Run the complete default/config/UI-focused files**

Run:

```powershell
python -m unittest tests.test_ai_targeting tests.test_settings tests.test_ui -v
```

Expected: all tests pass. If a failure asserts an old value while claiming to test defaults, update it; if it supplies explicit values, fix the production regression instead of rewriting the assertion.

- [ ] **Step 7: Commit Task 1**

```powershell
git add -- jitter_app/ai/targeting.py tests/test_ai_targeting.py tests/test_settings.py tests/test_ui.py
git diff --cached --check
git commit -m "feat: adopt balanced AI aim defaults"
```

---

### Task 2: Add the pure strict Trigger-lock state machine

**Files:**
- Modify: `jitter_app/ai/tracking.py:1-586`
- Modify: `tests/test_ai_tracking.py:1`

**Interfaces:**
- Consumes: `AimSettings`, `Detection`, `DetectionAnalysis`, `DetectionFrameSnapshot`, `TargetSnapshot`, `analyze_detections`, `detection_aim_point`, and the existing conservative geometry constants.
- Produces:
  - `StrictTriggerLockState(epoch: int | None = None, mode: str = "idle", confirmed_detection: Detection | None = None, confirmed_target: TargetSnapshot | None = None, preceding_target: TargetSnapshot | None = None, confidence: float | None = None, target_area: str = "head")`
  - `StrictTriggerLockObservation(state: StrictTriggerLockState, analysis: DetectionAnalysis)`
  - `observe_strict_trigger_lock(state, detections, settings, *, trigger_epoch, sequence, captured_at, frame_width=320, frame_height=320, output_width=None, output_height=None, capture_left=0, capture_top=0) -> StrictTriggerLockObservation`
- Preserves: `TrackerState`, `TrackingObservation`, and every existing `observe_detections` behavior and test.

- [ ] **Step 1: Add failing acquisition, loss-latch, and new-epoch tests**

Before adding the new class, change the existing `head_box(120, confidence=0.34)` fixture in `ConservativeTrackingTests.test_malformed_and_unaccepted_detections_are_not_published` to `confidence=0.24`. That value remains below both the old and new defaults, so Worker B's legacy suite passes independently before Task 1 is integrated.

Import the new API and add a separate `StrictTriggerLockTests` class:

```python
from jitter_app.ai.tracking import (
    StrictTriggerLockState,
    observe_strict_trigger_lock,
)

class StrictTriggerLockTests(unittest.TestCase):
    def observe(self, state, detections, sequence, *, epoch=1, settings=None,
                frame_width=320, frame_height=320, captured_at=None):
        return observe_strict_trigger_lock(
            state,
            detections,
            settings or AimSettings(),
            trigger_epoch=epoch,
            sequence=sequence,
            captured_at=(
                sequence / 120 if captured_at is None else captured_at
            ),
            frame_width=frame_width,
            frame_height=frame_height,
        )

    def test_ambiguous_continuation_latches_lost_until_new_epoch(self):
        acquired = self.observe(
            StrictTriggerLockState(), (head_box(120),), 1,
        )
        self.assertEqual(acquired.state.mode, "tracking")
        self.assertEqual(acquired.analysis.target.aim_x, 120)

        lost = self.observe(
            acquired.state, (head_box(126), head_box(114)), 2,
        )
        self.assertEqual(lost.state.mode, "lost")
        self.assertIsNone(lost.analysis.target)
        self.assertIsNone(lost.analysis.frame.selected_index)

        reappeared = self.observe(lost.state, (head_box(128),), 3)
        self.assertEqual(reappeared.state.mode, "lost")
        self.assertIsNone(reappeared.analysis.target)

        reacquired = self.observe(
            reappeared.state, (head_box(128),), 4, epoch=2,
        )
        self.assertEqual(reacquired.state.mode, "tracking")
        self.assertEqual(reacquired.analysis.target.aim_x, 128)
```

Add these exact behavioral tests:

```python
def test_empty_first_frame_consumes_epoch_into_lost(self):
    result = self.observe(StrictTriggerLockState(), (), 1)
    self.assertEqual(result.state.mode, "lost")
    self.assertEqual(result.state.epoch, 1)
    self.assertIsNone(result.analysis.target)
    self.assertIsNone(result.analysis.frame.selected_index)

def test_equal_distance_first_frame_fails_closed_instead_of_detector_order(self):
    result = self.observe(
        StrictTriggerLockState(), (head_box(140), head_box(180)), 1,
    )
    self.assertEqual(result.state.mode, "lost")
    self.assertIsNone(result.analysis.target)
    self.assertIsNone(result.analysis.frame.selected_index)
    self.assertEqual(len(result.analysis.frame.detections), 2)

def test_unique_continuation_survives_detector_order_change(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(120),), 1,
    )
    continued = self.observe(
        acquired.state, (head_box(220), head_box(122)), 2,
    )
    self.assertEqual(continued.state.mode, "tracking")
    self.assertEqual(continued.analysis.target.aim_x, 122)
    self.assertEqual(continued.analysis.frame.selected_index, 1)

def test_one_frame_disappearance_never_recovers_same_epoch(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(120),), 1,
    )
    missing = self.observe(acquired.state, (), 2)
    returned = self.observe(missing.state, (head_box(121),), 3)
    self.assertEqual(missing.state.mode, "lost")
    self.assertEqual(returned.state.mode, "lost")
    self.assertIsNone(returned.analysis.target)

def test_idle_epoch_none_keeps_stateless_overlay_analysis(self):
    result = self.observe(
        StrictTriggerLockState(), (head_box(150),), 1, epoch=None,
    )
    self.assertEqual(result.state.mode, "idle")
    self.assertEqual(result.analysis.target.aim_x, 150)
    self.assertEqual(result.analysis.frame.selected_index, 0)

def test_acquisition_uses_the_one_unique_nearest_candidate(self):
    result = self.observe(
        StrictTriggerLockState(), (head_box(80), head_box(150)), 1,
    )
    self.assertEqual(result.state.mode, "tracking")
    self.assertEqual(result.analysis.target.aim_x, 150)
    self.assertEqual(result.analysis.frame.selected_index, 1)

def test_acquisition_considers_heads_and_players_together(self):
    player = player_box(160, aim_y=160)
    head = head_box(300, 300)
    result = self.observe(
        StrictTriggerLockState(), (head, player), 1,
    )
    self.assertEqual(result.state.mode, "tracking")
    self.assertEqual(result.analysis.target.target_class, "player")
    self.assertEqual(result.analysis.frame.selected_index, 1)

def test_exact_and_near_overlapping_continuations_fail_closed(self):
    for contenders in (
        (head_box(121), head_box(121)),
        (head_box(121), head_box(121.001)),
    ):
        with self.subTest(contenders=contenders):
            acquired = self.observe(
                StrictTriggerLockState(), (head_box(120),), 1,
            )
            result = self.observe(acquired.state, contenders, 2)
            self.assertEqual(result.state.mode, "lost")
            self.assertIsNone(result.analysis.target)
            self.assertIsNone(result.analysis.frame.selected_index)

def test_state_and_observation_are_deeply_immutable(self):
    result = self.observe(
        StrictTriggerLockState(), (head_box(150),), 1,
    )
    with self.assertRaises(FrozenInstanceError):
        result.state.mode = "lost"
    with self.assertRaises(FrozenInstanceError):
        result.analysis.frame.selected_index = None
```

- [ ] **Step 2: Add failing identity-signature and native-geometry tests**

```python
def test_confidence_change_loses_active_epoch(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(120, confidence=.9),), 1,
        settings=AimSettings(confidence=.25),
    )
    changed = self.observe(
        acquired.state, (head_box(121, confidence=.9),), 2,
        settings=AimSettings(confidence=.30),
    )
    self.assertEqual(changed.state.mode, "lost")
    self.assertIsNone(changed.analysis.target)

def test_full_display_uses_native_center_and_canonical_match_radius(self):
    acquired = self.observe(
        StrictTriggerLockState(),
        (head_rectangle(800, 540, 60, 90),),
        1, frame_width=1920, frame_height=1080,
    )
    continued = self.observe(
        acquired.state,
        (head_rectangle(830, 540, 60, 90),),
        2, frame_width=1920, frame_height=1080,
    )
    self.assertEqual(continued.state.mode, "tracking")
    self.assertEqual(continued.analysis.target.frame_width, 1920)
    self.assertEqual(continued.analysis.target.frame_height, 1080)

def test_target_area_change_latches_lost(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(120),), 1,
        settings=AimSettings(target_area="head"),
    )
    changed = self.observe(
        acquired.state, (head_box(121),), 2,
        settings=AimSettings(target_area="chest"),
    )
    self.assertEqual(changed.state.mode, "lost")
    self.assertIsNone(changed.analysis.target)

def test_frame_geometry_change_latches_lost(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(120),), 1,
    )
    changed = self.observe(
        acquired.state, (head_box(240, 200, size=20),), 2,
        frame_width=640, frame_height=400,
    )
    self.assertEqual(changed.state.mode, "lost")
    self.assertIsNone(changed.analysis.target)

def test_target_class_change_latches_lost(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(120),), 1,
    )
    changed = self.observe(
        acquired.state, (player_box(121, aim_y=100),), 2,
    )
    self.assertEqual(changed.state.mode, "lost")
    self.assertIsNone(changed.analysis.target)
    self.assertEqual(changed.analysis.frame.detections[0].class_id, 0)

def test_invalid_and_non_finite_geometry_fail_closed(self):
    invalid_detections = (
        Detection(120, 95, 120, 105, .9, 7),
        Detection(float("nan"), 95, 125, 105, .9, 7),
        Detection(115, 95, float("inf"), 105, .9, 7),
    )
    for detection in invalid_detections:
        with self.subTest(detection=detection):
            result = self.observe(
                StrictTriggerLockState(), (detection,), 1,
            )
            self.assertEqual(result.state.mode, "lost")
            self.assertIsNone(result.analysis.target)
            self.assertIsNone(result.analysis.frame.selected_index)

def test_non_finite_capture_time_fails_closed(self):
    result = self.observe(
        StrictTriggerLockState(),
        (head_box(120),),
        1,
        captured_at=float("nan"),
    )
    self.assertEqual(result.state.mode, "lost")
    self.assertIsNone(result.analysis.target)

def test_area_ratio_boundaries_are_inclusive_and_excess_loses(self):
    for ratio, expected_mode in (
        (0.4, "tracking"),
        (2.5, "tracking"),
        (0.399, "lost"),
        (2.501, "lost"),
    ):
        with self.subTest(ratio=ratio):
            acquired = self.observe(
                StrictTriggerLockState(),
                (head_rectangle(120, 100, 10, 10),),
                1,
            )
            side = 10 * math.sqrt(ratio)
            result = self.observe(
                acquired.state,
                (head_rectangle(121, 100, side, side),),
                2,
            )
            self.assertEqual(result.state.mode, expected_mode)
            if expected_mode == "lost":
                self.assertIsNone(result.analysis.target)

def test_canonical_displacement_boundary_scales_to_full_display(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(600, 540),), 1,
        frame_width=1920, frame_height=1080,
    )
    exact = self.observe(
        acquired.state, (head_box(888, 540),), 2,
        frame_width=1920, frame_height=1080,
    )
    self.assertEqual(exact.state.mode, "tracking")

    acquired = self.observe(
        StrictTriggerLockState(), (head_box(600, 540),), 1,
        frame_width=1920, frame_height=1080,
    )
    outside = self.observe(
        acquired.state, (head_box(888.001, 540),), 2,
        frame_width=1920, frame_height=1080,
    )
    self.assertEqual(outside.state.mode, "lost")

def test_full_display_velocity_cap_scales_from_canonical_800_pps(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(600, 540),), 1,
        frame_width=1920, frame_height=1080, captured_at=0.0,
    )
    continued = self.observe(
        acquired.state, (head_box(840, 540),), 2,
        frame_width=1920, frame_height=1080, captured_at=0.010,
    )
    capped = self.observe(
        continued.state, (head_box(600, 540),), 3,
        frame_width=1920, frame_height=1080, captured_at=0.020,
    )
    self.assertEqual(capped.state.mode, "tracking")
    self.assertEqual(capped.analysis.target.aim_x, 600)

def test_prediction_horizon_is_capped_at_one_hundred_milliseconds(self):
    acquired = self.observe(
        StrictTriggerLockState(), (head_box(100),), 1,
        captured_at=0.0,
    )
    continued = self.observe(
        acquired.state, (head_box(140),), 2,
        captured_at=0.050,
    )
    capped = self.observe(
        continued.state, (head_box(172),), 3,
        captured_at=0.160,
    )
    self.assertEqual(capped.state.mode, "tracking")
    self.assertEqual(capped.analysis.target.aim_x, 172)
```

- [ ] **Step 3: Run strict tests and confirm missing-interface failures**

Run:

```powershell
python -m unittest tests.test_ai_tracking.StrictTriggerLockTests -v
```

Expected: import/error failures because the strict API does not exist.

- [ ] **Step 4: Define immutable strict state and full-geometry analysis helpers**

Add beside, not in place of, the legacy tracker:

```python
STRICT_LOCK_IDLE = "idle"
STRICT_LOCK_TRACKING = "tracking"
STRICT_LOCK_LOST = "lost"

@dataclass(frozen=True)
class StrictTriggerLockState:
    epoch: int | None = None
    mode: str = STRICT_LOCK_IDLE
    confirmed_detection: Detection | None = None
    confirmed_target: TargetSnapshot | None = None
    preceding_target: TargetSnapshot | None = None
    confidence: float | None = None
    target_area: str = "head"

@dataclass(frozen=True)
class StrictTriggerLockObservation:
    state: StrictTriggerLockState
    analysis: DetectionAnalysis
```

Use `analyze_detections` to create the complete accepted frame and viewport. Add a pure helper that returns a copy with `target=None` and `selected_index=None` without dropping detections or geometry:

```python
def _strict_without_target(analysis: DetectionAnalysis) -> DetectionAnalysis:
    return DetectionAnalysis(
        None,
        replace(analysis.frame, selected_index=None),
    )

def _strict_with_candidate(
    analysis: DetectionAnalysis,
    candidate: _Candidate,
) -> DetectionAnalysis:
    return DetectionAnalysis(
        candidate.target,
        replace(analysis.frame, selected_index=candidate.index),
    )
```

Build strict candidates separately from the Overlay tuple. Preserve each accepted tuple index in `_Candidate`. A candidate is valid only when `captured_at` is finite and non-negative, all box coordinates and confidence are finite, width and height are positive, `detection_aim_point` returns finite coordinates for the configured Target Area, and the resulting `TargetSnapshot` carries `base.frame.frame_width` and `base.frame.frame_height`. Invalid boxes may remain in the immutable Overlay tuple, but they can never acquire or continue a lock.

- [ ] **Step 5: Implement one-shot acquisition and irreversible loss**

Implement this exact control order in `observe_strict_trigger_lock`:

```python
base = analyze_detections(
    detections,
    settings,
    sequence=sequence,
    captured_at=captured_at,
    frame_width=frame_width,
    frame_height=frame_height,
    output_width=output_width,
    output_height=output_height,
    capture_left=capture_left,
    capture_top=capture_top,
)
area = validated_target_area(settings.target_area)

if trigger_epoch is None:
    return StrictTriggerLockObservation(StrictTriggerLockState(), base)
if type(trigger_epoch) is not int or trigger_epoch <= 0:
    lost = StrictTriggerLockState(epoch=None, mode=STRICT_LOCK_LOST)
    return StrictTriggerLockObservation(lost, _strict_without_target(base))
if state.epoch != trigger_epoch:
    candidates = _strict_candidates(base, settings)
    candidate = _strict_unique_initial_candidate(
        candidates,
        base.frame.frame_width,
        base.frame.frame_height,
    )
    if candidate is None:
        lost = StrictTriggerLockState(
            epoch=trigger_epoch,
            mode=STRICT_LOCK_LOST,
            confidence=settings.confidence,
            target_area=area,
        )
        return StrictTriggerLockObservation(lost, _strict_without_target(base))
    tracking = StrictTriggerLockState(
        epoch=trigger_epoch,
        mode=STRICT_LOCK_TRACKING,
        confirmed_detection=candidate.detection,
        confirmed_target=candidate.target,
        confidence=settings.confidence,
        target_area=area,
    )
    return StrictTriggerLockObservation(
        tracking,
        _strict_with_candidate(base, candidate),
    )
if state.mode == STRICT_LOCK_LOST:
    return StrictTriggerLockObservation(state, _strict_without_target(base))
```

`_strict_unique_initial_candidate` computes squared distance from every valid strict candidate to `(frame_width / 2.0, frame_height / 2.0)`, rejects an empty set, and rejects every exact nearest-distance tie. It returns the one `_Candidate` only when the minimum distance occurs exactly once; detector order must not resolve the tie.

- [ ] **Step 6: Implement unique canonical-scaled continuation**

For the tracking branch:

1. Reject a Confidence/Target Area change or frame-width/height change into `LOST`.
2. Construct current candidates from `base.frame.detections` with `detection_aim_point` and native `TargetSnapshot` geometry.
3. Require the same `target_class` as the confirmed target.
4. Require area ratio `0.4 <= current_area / confirmed_area <= 2.5`.
5. Predict from `preceding_target -> confirmed_target` for at most `0.100` seconds; clamp velocity to `800 * max(frame_width, frame_height) / 320` source pixels/second.
6. Let `scale = max(frame_width, frame_height) / 320`. Require distance to prediction no larger than `max(48 * scale, 1.5 * confirmed_box_diagonal)`.
7. Reuse `_intersection_over_union` and the legacy weighted association score `0.60 * distance_ratio + 0.25 * (1.0 - iou) + 0.15 * min(1.0, abs(log(area_ratio)))`; require the score to be at most `1.0 - AMBIGUITY_MARGIN`. This makes overlap constrain borderline displacement without requiring overlap for every legitimate fast frame.
8. Treat every candidate satisfying all gates as plausible. Require `len(plausible) == 1`; zero or multiple candidates enter irreversible `LOST`. Do not use the score to choose between two plausible candidates.
9. Update `preceding_target` from the old confirmed target and publish only the unique candidate.

Use the candidate's index in the immutable accepted tuple:

```python
analysis = DetectionAnalysis(
    target,
    replace(base.frame, selected_index=index),
)
next_state = StrictTriggerLockState(
    epoch=trigger_epoch,
    mode=STRICT_LOCK_TRACKING,
    confirmed_detection=detection,
    confirmed_target=target,
    preceding_target=state.confirmed_target,
    confidence=settings.confidence,
    target_area=area,
)
return StrictTriggerLockObservation(next_state, analysis)
```

- [ ] **Step 7: Run strict and legacy tracker tests**

Run:

```powershell
python -m unittest tests.test_ai_tracking.StrictTriggerLockTests -v
python -m unittest tests.test_ai_tracking -v
python -m unittest tests.test_ai_targeting -v
```

Expected: strict tests pass and every legacy tracker/stateless test remains unchanged.

- [ ] **Step 8: Commit Task 2**

```powershell
git add -- jitter_app/ai/tracking.py tests/test_ai_tracking.py
git diff --cached --check
git commit -m "feat: add strict per-trigger target lock"
```

---

### Task 3: Integrate managed Trigger epochs into AI service and Adaptive Zoom

**Files:**
- Modify: `jitter_app/ai/service.py:153-640`
- Modify: `jitter_app/ai/zoom.py:264-346`
- Modify: `tests/test_ai_service.py:1`
- Modify: `tests/test_ai_zoom.py:380-650`

**Interfaces:**
- Consumes: Task 2 `StrictTriggerLockState`, `StrictTriggerLockObservation`, strict mode constants, and `observe_strict_trigger_lock`.
- Produces:
  - `AiService.start(settings_provider, zoom_gate_provider=None, trigger_epoch_provider=None, *, model_path=None, capture_mode=CENTER_320) -> int | None`
  - `AiService.invalidate_trigger_lock(trigger_epoch: int | None) -> int`
  - omitted `trigger_epoch_provider` preserves legacy stateless target publication;
  - supplied provider returning `None` publishes stateless Overlay detection frames but no movement snapshot.

- [ ] **Step 1: Add a failing Adaptive Zoom ambiguity test**

In `tests/test_ai_zoom.py`, use the existing `base_player()` fixture and its established crop transform so both crop-local heads map inside the selected base player:

```python
def test_multiple_compatible_refinements_fail_back_to_base(self):
    result = compose_zoom_refinement(
        self.base_player(),
        (
            Detection(70, 35, 90, 55, .92, 7),
            Detection(74, 39, 94, 59, .93, 7),
        ),
        ZoomTransform(80, 40, 160, 160, 320, 320, 2.0),
        AimSettings(confidence=.35),
    )
    self.assertIsNone(result)
```

Keep `test_refinement_stays_with_selected_base_target_in_crowded_crop`, but move its `nearer_crosshair` fixture outside the selected seed's expanded association bounds while leaving it nearer the screen crosshair:

```python
matching_base = Detection(35, 35, 45, 45, 0.92, 7)
nearer_crosshair = Detection(85, 85, 95, 95, 0.95, 7)
```

With `ZoomTransform(60, 60, 160, 160, 320, 320, 2.0)`, that distractor maps to aim point `(150, 150)`, outside the seed's maximum compatible coordinate `142` while the matching base maps to `(100, 100)`. Preserve the existing successful matching-base assertions. This keeps the old test's unrelated-distractor guarantee without contradicting the new multiple-compatible fallback.

- [ ] **Step 2: Add failing managed/unmanaged service tests**

Use `ControlledCapture` so each assertion corresponds to one released frame:

```python
def test_managed_idle_keeps_overlay_selection_but_no_movement_target(self):
    epoch = {"value": None}
    capture = ControlledCapture([rgb_frame(150)])
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: FakeDetector(),
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)
    service.start(
        AimSettings,
        lambda: False,
        lambda: epoch["value"],
    )
    capture.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_detection_snapshot() is not None))
    self.assertIsNone(service.latest_snapshot())
    self.assertEqual(service.latest_detection_snapshot().selected_index, 0)

def test_omitted_epoch_provider_preserves_legacy_stateless_publication(self):
    capture = ControlledCapture([rgb_frame(150)])
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: FakeDetector(),
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)
    service.start(AimSettings, lambda: False)
    capture.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))
```

Use direct `AiService` construction like the existing test file; do not add a production-only test factory.

- [ ] **Step 3: Add failing epoch claim, loss, invalidation, and race tests**

Add this deterministic test capture beside `ControlledCapture`. Its event fires only when the worker asks for another frame after completely handling the one delivered frame:

```python
class FrameCompletionCapture(ControlledCapture):
    def __init__(self, frames, **kwargs):
        frame_count = len(frames)
        super().__init__(frames, **kwargs)
        self._delivered_count = 0
        self.processed = [threading.Event() for _ in range(frame_count)]

    def read(self):
        if self._delivered_count:
            self.processed[self._delivered_count - 1].set()
        frame = super().read()
        if frame is not None:
            self._delivered_count += 1
        return frame
```

Add a small test-only constructor inside `AiServiceTests`:

```python
def start_managed_sequence(self, outputs, epoch):
    capture = ControlledCapture([rgb_frame(index + 1) for index in range(len(outputs))])
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: SequenceDetector(outputs),
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)
    service.start(AimSettings, lambda: False, lambda: epoch["value"])
    return service, capture
```

Then add these controlled tests:

```python
def test_epoch_acquires_once_and_missing_frame_latches_lost(self):
    target = (Detection(145, 145, 165, 165, .9, 7),)
    epoch = {"value": 1}
    service, capture = self.start_managed_sequence(
        (target, (), target, target), epoch,
    )
    capture.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))
    capture.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_snapshot() is None))
    capture.release_frame()
    self.assertTrue(wait_until(
        lambda: service.latest_detection_snapshot().sequence >= 3
    ))
    self.assertIsNone(service.latest_snapshot())
    epoch["value"] = 2
    capture.release_frame()
    self.assertTrue(wait_until(
        lambda: service.latest_snapshot() is not None
        and service.latest_snapshot().sequence == 4
    ))

def test_same_epoch_cannot_be_claimed_by_successor_generation(self):
    target = (Detection(145, 145, 165, 165, .9, 7),)
    epoch = {"value": 1}
    capture_one = ControlledCapture([rgb_frame(1)])
    capture_two = ControlledCapture([rgb_frame(2)])
    capture_three = ControlledCapture([rgb_frame(3)])
    captures = [capture_one, capture_two, capture_three]
    detectors = [SequenceDetector([target]) for _ in range(3)]
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: detectors.pop(0),
        capture_factory=lambda _mode: captures.pop(0),
    )
    self.addCleanup(service.close)

    service.start(AimSettings, lambda: False, lambda: epoch["value"])
    capture_one.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))
    service.stop("generation_one_complete")
    self.assertTrue(wait_until(lambda: not service.worker_active))

    service.start(AimSettings, lambda: False, lambda: epoch["value"])
    capture_two.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_detection_snapshot() is not None))
    self.assertIsNone(service.latest_snapshot())
    service.stop("same_epoch_rejected")
    self.assertTrue(wait_until(lambda: not service.worker_active))

    epoch["value"] = 2
    service.start(AimSettings, lambda: False, lambda: epoch["value"])
    capture_three.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_snapshot() is not None))
```

Add the remaining race and validation tests with these exact event boundaries:

```python
def test_release_during_blocking_base_inference_discards_result(self):
    epoch = {"value": 1}
    capture = FrameCompletionCapture([rgb_frame(1)])
    detector = BlockingDetector()
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: detector,
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)
    service.start(AimSettings, lambda: False, lambda: epoch["value"])
    capture.release_frame()
    self.assertTrue(detector.entered.wait(1.0))
    epoch["value"] = None
    detector.release.set()
    self.assertTrue(capture.processed[0].wait(1.0))
    self.assertIsNone(service.latest_snapshot())
    self.assertIsNone(service.latest_detection_snapshot())

def test_invalidate_unclaimed_epoch_prevents_later_acquisition(self):
    target = (Detection(145, 145, 165, 165, .9, 7),)
    epoch = {"value": 1}
    service, capture = self.start_managed_sequence((target,), epoch)
    service.invalidate_trigger_lock(1)
    capture.release_frame()
    self.assertTrue(wait_until(lambda: service.latest_detection_snapshot() is not None))
    self.assertIsNone(service.latest_snapshot())

def test_invalidation_during_blocking_inference_discards_result(self):
    epoch = {"value": 1}
    capture = FrameCompletionCapture([rgb_frame(1)])
    detector = BlockingDetector()
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: detector,
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)
    service.start(AimSettings, lambda: False, lambda: epoch["value"])
    capture.release_frame()
    self.assertTrue(detector.entered.wait(1.0))
    service.invalidate_trigger_lock(1)
    detector.release.set()
    self.assertTrue(capture.processed[0].wait(1.0))
    self.assertIsNone(service.latest_snapshot())
    self.assertIsNone(service.latest_detection_snapshot())

def test_older_epoch_cannot_be_reclaimed_after_a_newer_claim(self):
    target = (Detection(145, 145, 165, 165, .9, 7),)
    epoch = {"value": 1}
    service, capture = self.start_managed_sequence(
        (target, target, target), epoch,
    )
    capture.release_frame()
    self.assertTrue(wait_until(
        lambda: service.latest_snapshot() is not None
        and service.latest_snapshot().sequence == 1
    ))
    epoch["value"] = 2
    capture.release_frame()
    self.assertTrue(wait_until(
        lambda: service.latest_snapshot() is not None
        and service.latest_snapshot().sequence == 2
    ))
    epoch["value"] = 1
    self.release_and_wait(capture, service, 3)
    self.assertIsNone(service.latest_snapshot())
    self.assertIsNone(service.latest_detection_snapshot().selected_index)

def test_release_during_second_inference_preserves_prior_frame_only(self):
    target = (Detection(145, 145, 165, 165, .9, 7),)
    entered = threading.Event()
    release = threading.Event()

    def block_second_call():
        entered.set()
        release.wait(1.0)

    detector = SequentialDetector(
        (target, target), second_call_hook=block_second_call,
    )
    capture = FrameCompletionCapture([rgb_frame(1), rgb_frame(2)])
    epoch = {"value": 1}
    service = AiService(
        lambda _event: None,
        detector_factory=lambda _path: detector,
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)
    service.start(AimSettings, lambda: False, lambda: epoch["value"])

    capture.release_frame()
    self.assertTrue(capture.processed[0].wait(1.0))
    before = service.latest_detection_snapshot()
    self.assertEqual(before.sequence, 1)
    self.assertEqual(before.selected_index, 0)

    capture.release_frame()
    self.assertTrue(entered.wait(1.0))
    epoch["value"] = None
    release.set()
    self.assertTrue(capture.processed[1].wait(1.0))
    after = service.latest_detection_snapshot()
    self.assertEqual(after.sequence, before.sequence)
    self.assertEqual(after.detections, before.detections)
    self.assertIsNone(after.selected_index)
    self.assertIsNone(service.latest_snapshot())

def test_invalid_bool_zero_and_negative_epochs_publish_no_movement(self):
    target = (Detection(145, 145, 165, 165, .9, 7),)
    for invalid in (True, 0, -1):
        with self.subTest(epoch=invalid):
            epoch = {"value": invalid}
            service, capture = self.start_managed_sequence((target,), epoch)
            with self.assertLogs(
                "jitter_app.ai.service", level="WARNING"
            ):
                capture.release_frame()
                self.assertTrue(wait_until(
                    lambda: service.latest_detection_snapshot() is not None
                ))
            self.assertIsNone(service.latest_snapshot())

def test_trigger_epoch_provider_exception_uses_worker_error_boundary(self):
    events = []
    capture = ControlledCapture([rgb_frame(1)])
    service = AiService(
        events.append,
        detector_factory=lambda _path: FakeDetector(),
        capture_factory=lambda _mode: capture,
    )
    self.addCleanup(service.close)

    def failing_provider():
        raise RuntimeError("epoch provider failed")

    service.start(AimSettings, lambda: False, failing_provider)
    with self.assertLogs("jitter_app.ai.service", level="ERROR"):
        capture.release_frame()
        self.assertTrue(capture.closed.wait(1.0))
    self.assertEqual(service.status, "error")
    self.assertIsNone(service.latest_snapshot())
    self.assertTrue(any(event.kind == "error" for event in events))
```

Use the existing `make_zoom_service` helper for the refinement race:

```python
def test_release_during_refinement_discards_result(self):
    epoch = {"value": 1}
    capture = FrameCompletionCapture([rgb_frame(1)])
    detector = SequentialDetector(
        (
            (Detection(140, 80, 180, 160, .9, 0),),
            (Detection(144, 135, 174, 165, .95, 7),),
        ),
        second_call_hook=lambda: epoch.update(value=None),
    )
    service, _events = self.make_zoom_service(detector, capture=capture)
    service.start(
        AimSettings,
        lambda: True,
        lambda: epoch["value"],
    )
    capture.release_frame()
    self.assertTrue(capture.processed[0].wait(1.0))
    self.assertEqual(len(detector.frames), 2)
    self.assertIsNone(service.latest_snapshot())
    self.assertIsNone(service.latest_detection_snapshot())

def test_refined_box_never_becomes_next_frame_lock_identity(self):
    selected_base = self.small_head(160.0)
    refined_crop_local = Detection(112, 100, 124, 112, .95, 7)
    next_base = self.small_head(115.0)
    detector = SequentialDetector(
        ((selected_base,), (refined_crop_local,), (next_base,))
    )
    capture = ControlledCapture([rgb_frame(1), rgb_frame(2)])
    gate = {"active": True}
    epoch = {"value": 1}
    clock = MutableClock(10.0)
    service, _events = self.make_zoom_service(
        detector, capture=capture, clock=clock,
    )
    service.start(
        AimSettings,
        lambda: gate["active"],
        lambda: epoch["value"],
    )

    self.release_and_wait(capture, service, 1)
    self.assertNotEqual(
        service.latest_detection_snapshot().detections[0],
        selected_base,
    )
    gate["active"] = False
    clock.set(10.01)
    self.release_and_wait(capture, service, 2)
    self.assertEqual(service.latest_snapshot().aim_x, 115.0)
    self.assertEqual(
        service.latest_detection_snapshot().detections,
        (next_base,),
    )
```

The new-epoch-after-release assertion is the fourth-frame portion of `test_epoch_acquires_once_and_missing_frame_latches_lost`.

- [ ] **Step 4: Run focused service/zoom tests to confirm failures**

Run:

```powershell
python -m unittest tests.test_ai_zoom -v
python -m unittest tests.test_ai_service -v
```

Expected: the new refinement assertion fails because the nearest compatible result wins, and service tests fail because the provider/claim API is absent.

- [ ] **Step 5: Make refinement compatibility unique**

Change `compose_zoom_refinement` after building `compatible`:

```python
if len(compatible) != 1:
    return None
selected_refined, selected_point = compatible[0]
```

Do not update any next-frame strict-lock state from the refined box. Existing service behavior already keeps base analysis separate; preserve that separation.

- [ ] **Step 6: Add service-level epoch validation and atomic claim/invalidation**

Initialize one cross-generation record in `AiService.__init__`:

```python
self._claimed_trigger_epoch: int | None = None
```

Add lock-protected helpers. Invalid values never become claims:

```python
@staticmethod
def _validated_trigger_epoch(raw: object) -> int | None:
    return raw if type(raw) is int and raw > 0 else None

def _claim_trigger_epoch(self, epoch: int) -> bool:
    with self._lock:
        if (
            self._claimed_trigger_epoch is not None
            and epoch <= self._claimed_trigger_epoch
        ):
            return False
        self._claimed_trigger_epoch = epoch
        return True

def invalidate_trigger_lock(self, trigger_epoch: int | None) -> int:
    with self._lock:
        epoch = self._validated_trigger_epoch(trigger_epoch)
        if (
            epoch is not None
            and (
                self._claimed_trigger_epoch is None
                or epoch > self._claimed_trigger_epoch
            )
        ):
            self._claimed_trigger_epoch = epoch
        self._targeting_revision += 1
        self._latest = None
        if self._latest_detection is not None:
            self._latest_detection = replace(
                self._latest_detection,
                selected_index=None,
            )
        return self._targeting_revision
```

Factor the duplicated snapshot-clearing body shared with `reset_targeting` into a private `_reset_targeting_locked()` method so both public methods remain atomic.

- [ ] **Step 7: Extend `start` without breaking compatibility callers**

Use the exact signature:

```python
def start(
    self,
    settings_provider: Callable[[], AimSettings],
    zoom_gate_provider: Callable[[], bool] | None = None,
    trigger_epoch_provider: Callable[[], int | None] | None = None,
    *,
    model_path: Path | str | None = None,
    capture_mode: str = CENTER_320,
) -> int | None:
    managed_trigger_lock = trigger_epoch_provider is not None
    resolved_trigger_epoch_provider = trigger_epoch_provider
    if resolved_trigger_epoch_provider is None:
        resolved_trigger_epoch_provider = lambda: None
```

Merge those first three statements into the existing `start` body before its
capture-mode validation; do not replace the rest of that body. Pass both new
values to the worker in this exact order:

```python
args=(
    generation,
    stop_event,
    settings_provider,
    zoom_gate_provider,
    resolved_trigger_epoch_provider,
    managed_trigger_lock,
    model_path,
    generation_capture_mode,
)
```

Use the matching worker signature:

```python
def _worker(
    self,
    generation: int,
    stop_event: threading.Event,
    settings_provider: Callable[[], AimSettings],
    zoom_gate_provider: Callable[[], bool],
    trigger_epoch_provider: Callable[[], int | None],
    managed_trigger_lock: bool,
    generation_model_path: Path | str | None,
    generation_capture_mode: str,
) -> None:
    lock_state = StrictTriggerLockState()
    active_trigger_epoch: int | None = None
```

The last two assignments initialize generation-local strict state before the
existing capture loop. In managed mode, log a warning once per distinct
invalid provider value per generation and treat it as no epoch; do not claim
booleans, zero, negative integers, strings, or other objects.

- [ ] **Step 8: Integrate strict analysis into the worker**

At generation start initialize:

```python
lock_state = StrictTriggerLockState()
active_trigger_epoch: int | None = None
```

For every captured frame:

1. Read/validate the Trigger epoch and read the targeting revision before base inference when managed.
2. If the epoch differs from `active_trigger_epoch`, set the new epoch and the current revision as the worker baselines, then atomically claim the epoch. A successful new claim starts a fresh idle state for acquisition; a failed claim creates `StrictTriggerLockState(epoch=epoch, mode=STRICT_LOCK_LOST)`. A `None` epoch resets worker-local state to idle without changing the service claim record.
3. Only when the epoch is unchanged, compare the current targeting revision with its worker baseline. A revision change for that same claimed epoch updates the baseline and forces irreversible `LOST` before analysis. This prevents the reset performed immediately before exposing a genuinely new epoch from invalidating that new epoch.
4. Run `observe_strict_trigger_lock` in managed mode; retain `analyze_detections` unchanged in unmanaged mode.
5. Re-read the provider after base inference, before refinement, after refinement, and immediately before publication. Also compare the targeting revision captured at frame start immediately before publication. If either differs, discard the entire newly inferred frame: do not advance `_latest_detection.sequence`, do not replace its detections, and do not emit a zoom transition from that frame. Atomically set `_latest=None` and clear only the selected index on the previously published detection frame, if one exists.
6. Run Adaptive Zoom only when the strict analysis has a target and the existing zoom gate is true.
7. In managed mode publish `_latest` only for active `TRACKING`; with no epoch publish only `_latest_detection`. In unmanaged mode keep `_latest = published.target`.
8. On a normally observed `LOST` frame whose epoch/revision remains current, atomically store `_latest=None` and that current frame with `selected_index=None`; never retain a stale target for 150 ms. This is distinct from Step 5's stale in-flight frame, which is never published.

Keep FPS/provider/zoom events, capture timestamps, stop events, and generation checks unchanged.

- [ ] **Step 9: Run service, zoom, and movement integration tests**

Run:

```powershell
python -m unittest tests.test_ai_zoom tests.test_ai_service tests.test_makcu_service tests.test_motion -v
```

Expected: all pass, including every old caller that omits the provider.

- [ ] **Step 10: Commit Task 3**

```powershell
git add -- jitter_app/ai/service.py jitter_app/ai/zoom.py tests/test_ai_service.py tests/test_ai_zoom.py
git diff --cached --check
git commit -m "feat: enforce strict trigger lock in AI runtime"
```

---

### Task 4: Own Trigger epochs in the Tk lifecycle

**Files:**
- Modify: `jitter_app/presentation/ui.py:340-454,2327-2350,3728-3897,4058-4124,4201-4389,4471-4490,4746-4804,4959-5007,5056-5092`
- Modify: `tests/test_ui.py:219-339,560-730,4423-4510,5350-5645`

**Interfaces:**
- Consumes: Task 3 `AiService.start(settings_provider, zoom_gate_provider=None, trigger_epoch_provider=None, *, model_path=None, capture_mode=CENTER_320)`, `AiService.reset_targeting()`, and `AiService.invalidate_trigger_lock(epoch)`.
- Produces:
  - `JitterApp.get_trigger_lock_epoch() -> int | None`, safe for the inference thread;
  - a monotonic `_trigger_lock_counter`, runtime `_trigger_lock_epoch`, and `_trigger_lock_owner` (`"normal"`, `"test"`, or `None`);
  - a main-thread `_physical_buttons_down` set that survives `TriggerGate.clear()` and deduplicates raw device edges;
  - one synthetic epoch for each AI Test 3s run;
  - immediate reset/invalidation ordering on every raw Trigger and lifecycle edge.

- [ ] **Step 1: Extend UI test doubles without changing existing call tuple indexes**

In `StubAiService` and `StrictDuplicateStartAiService`, accept `trigger_epoch_provider=None`, keep `start_calls` as the existing four-tuple `(settings_provider, zoom_gate_provider, model_path, capture_mode)`, and record providers separately:

```python
self.trigger_epoch_providers = []
self.invalidated_trigger_epochs = []

def start(self, settings_provider, zoom_gate_provider=None,
          trigger_epoch_provider=None, *, model_path=None,
          capture_mode=CENTER_320):
    self.trigger_epoch_providers.append(trigger_epoch_provider)
    self.start_calls.append(
        (settings_provider, zoom_gate_provider, model_path, capture_mode)
    )

def invalidate_trigger_lock(self, epoch):
    self.invalidated_trigger_epochs.append(epoch)
    return self.reset_targeting()
```

This preserves dozens of existing `[2]` model-path and `[3]` capture-mode assertions.

- [ ] **Step 2: Add failing provider and raw-edge tests**

Add near existing Adaptive Zoom gate tests:

```python
def test_ai_runtime_receives_thread_safe_trigger_epoch_provider(self):
    self.service.connected = True
    self.app.toggle_ai_source()
    self.app.set_master(True)
    provider = self.ai.trigger_epoch_providers[-1]
    self.assertIs(provider.__self__, self.app)
    self.assertIs(provider.__func__, self.app.get_trigger_lock_epoch.__func__)
    self.assertIsNone(provider())

def test_raw_trigger_release_and_repress_creates_new_epoch(self):
    self.prepare_armed_sources(MotionSources(False, True))
    self.handle_current_ai_event(AiEvent("ready", "DmlExecutionProvider"))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    first = self.app.get_trigger_lock_epoch()
    self.assertIsInstance(first, int)
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertEqual(self.app.get_trigger_lock_epoch(), first)
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertGreater(self.app.get_trigger_lock_epoch(), first)
```

- [ ] **Step 3: Add failing modifier, eligibility, lifecycle, and Test tests**

Add a test helper after `prepare_armed_sources`:

```python
def begin_ai_trigger_epoch(self):
    self.prepare_armed_sources(MotionSources(False, True))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    epoch = self.app.get_trigger_lock_epoch()
    self.assertIsInstance(epoch, int)
    return epoch
```

Add these exact tests:

```python
def test_modifier_release_repress_keeps_raw_trigger_epoch(self):
    self.app.modifier_var.set("Right")
    self.app.on_bindings_changed()
    self.prepare_armed_sources(MotionSources(False, True))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    epoch = self.app.get_trigger_lock_epoch()
    self.app.handle_service_event(ServiceEvent("button", ("Right", True)))
    self.app.handle_service_event(ServiceEvent("button", ("Right", False)))
    self.app.handle_service_event(ServiceEvent("button", ("Right", True)))
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_press_while_master_disabled_requires_new_raw_press_after_enable(self):
    self.service.connected = True
    self.app.ai_selected = True
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.set_master(True)
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsInstance(self.app.get_trigger_lock_epoch(), int)

def test_press_while_ai_unselected_requires_new_press_after_source_addition(self):
    self.prepare_armed_sources(MotionSources(True, False))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.toggle_ai_source()
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsInstance(self.app.get_trigger_lock_epoch(), int)

def test_press_while_disconnected_cannot_acquire_after_reconnect(self):
    self.prepare_armed_sources(MotionSources(False, True))
    self.service.connected = False
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.service.connected = True
    self.app.handle_service_event(ServiceEvent("reconnected", "Makcu"))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_stop_clears_trigger_epoch_before_runtime_cancellation(self):
    self.begin_ai_trigger_epoch()
    self.app.emergency_stop("test stop")
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_disconnect_clears_trigger_epoch(self):
    self.begin_ai_trigger_epoch()
    self.app.handle_service_event(ServiceEvent("disconnected", "lost"))
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_binding_change_clears_trigger_epoch(self):
    self.begin_ai_trigger_epoch()
    self.app.trigger_var.set("Mouse4")
    self.app.on_bindings_changed()
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_ai_source_removal_clears_trigger_epoch(self):
    self.begin_ai_trigger_epoch()
    self.app.toggle_ai_source()
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_adding_or_removing_jitter_clears_normal_ai_epoch(self):
    epoch = self.begin_ai_trigger_epoch()
    self.app.toggle_jitter_source()
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))

    self.app.close_app()
    self.make_app()
    self.prepare_armed_sources(MotionSources(True, True))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsInstance(self.app.get_trigger_lock_epoch(), int)
    self.app.toggle_jitter_source()
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_confidence_and_target_area_changes_invalidate_active_epoch(self):
    epoch = self.begin_ai_trigger_epoch()
    self.app.ai_vars["confidence"].set("0.30")
    self.app._ai_changed("confidence")
    self.assertEqual(self.ai.invalidated_trigger_epochs[-1], epoch)
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)
    self.app.target_area_var.set("Chest")
    self.app._target_area_changed()
    self.assertEqual(self.ai.invalidated_trigger_epochs[-1], epoch)
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)

def test_motion_only_ai_settings_do_not_invalidate_active_epoch(self):
    epoch = self.begin_ai_trigger_epoch()
    baseline = list(self.ai.invalidated_trigger_epochs)
    for key, value in (
        ("aim_strength", "0.40"),
        ("smoothing", "0.60"),
        ("max_step", "19"),
    ):
        self.app.ai_vars[key].set(value)
        self.app._ai_changed(key)
    self.app._reset_ai_curve()
    self.assertEqual(self.ai.invalidated_trigger_epochs, baseline)
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)

def test_ai_test_run_creates_one_synthetic_epoch_and_clears_it(self):
    self.prepare_armed_sources(MotionSources(False, True))
    self.app.start_test_run()
    epoch = self.app.get_trigger_lock_epoch()
    self.assertIsInstance(epoch, int)
    self.assertFalse(self.app._begin_test_motion(self.app._test_generation))
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)
    self.app._restore_after_test()
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_trigger_held_through_test_completion_requires_release_repress(self):
    self.prepare_armed_sources(MotionSources(False, True))
    self.app.start_test_run()
    synthetic = self.app.get_trigger_lock_epoch()
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertEqual(self.app.get_trigger_lock_epoch(), synthetic)

    self.app._restore_after_test()
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", False)))
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertGreater(self.app.get_trigger_lock_epoch(), synthetic)

def test_target_reset_finishes_before_new_epoch_is_exposed(self):
    self.prepare_armed_sources(MotionSources(False, True))
    observed_epochs = []
    motion_observations = []
    original_reset = self.ai.reset_targeting
    original_motion_start = self.service.start_composite_motion_source

    def recording_reset():
        observed_epochs.append(self.app.get_trigger_lock_epoch())
        return original_reset()

    def recording_motion_start(*args, **kwargs):
        motion_observations.append(
            (
                self.app.get_trigger_lock_epoch(),
                self.ai.snapshot,
            )
        )
        return original_motion_start(*args, **kwargs)

    self.ai.reset_targeting = recording_reset
    self.service.start_composite_motion_source = recording_motion_start
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertEqual(observed_epochs, [None])
    self.assertIsInstance(self.app.get_trigger_lock_epoch(), int)
    self.assertEqual(
        motion_observations,
        [(self.app.get_trigger_lock_epoch(), None)],
    )

def test_master_and_hotkey_disable_clear_active_epoch(self):
    self.begin_ai_trigger_epoch()
    self.app.set_master(False)
    self.assertIsNone(self.app.get_trigger_lock_epoch())

    self.app.close_app()
    self.make_app()
    self.begin_ai_trigger_epoch()
    self.app._cancel_after("_ui_pump_after_id")
    self.app._hotkey_pressed()
    self.drain_ui_queue()
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_ai_error_and_close_clear_active_epoch(self):
    self.begin_ai_trigger_epoch()
    with self.assertLogs(level="ERROR"):
        self.handle_current_ai_event(
            AiEvent("error", "RuntimeError: detector failed")
        )
    self.assertIsNone(self.app.get_trigger_lock_epoch())

    self.app.close_app()
    self.make_app()
    self.begin_ai_trigger_epoch()
    self.app.close_app()
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_reconnect_during_held_trigger_cannot_reacquire(self):
    self.begin_ai_trigger_epoch()
    self.app.reconnect()
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.service.connected = True
    self.app.handle_service_event(ServiceEvent("reconnected", "Makcu"))
    self.app.set_master(True)
    self.assertIsNone(self.app.get_trigger_lock_epoch())
    self.app.handle_service_event(ServiceEvent("button", ("Left", True)))
    self.assertIsNone(self.app.get_trigger_lock_epoch())

def test_capture_switch_invalidates_but_keeps_exposed_epoch(self):
    epoch = self.begin_ai_trigger_epoch()
    self.app.capture_mode_var.set("Full Display")
    self.app._capture_mode_changed()
    self.assertEqual(self.ai.invalidated_trigger_epochs[-1], epoch)
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)

def test_model_switch_invalidates_but_keeps_exposed_epoch(self):
    epoch = self.begin_ai_trigger_epoch()
    self.begin_custom_model_switch("strict-lock.onnx")
    self.assertEqual(self.ai.invalidated_trigger_epochs[-1], epoch)
    self.assertEqual(self.app.get_trigger_lock_epoch(), epoch)
```

Update two existing revision-count assertions so they measure the lifecycle operation under test rather than assuming Trigger down never reset targeting:

```python
# test_motion_worker_receives_live_ai_targeting_epoch_provider
before_switch = provider()
self.app.capture_mode_var.set("Full Display")
self.app._capture_mode_changed()
self.assertGreater(provider(), before_switch)
self.assertEqual(provider(), self.ai.targeting_revision)

# test_active_switch_stops_motion_and_ai_then_restarts_candidate_on_validation
resets_before_switch = self.ai.reset_targeting_calls
self.app.model_browse_button.invoke()
self.assertEqual(
    self.ai.reset_targeting_calls,
    resets_before_switch + 1,
)
```

- [ ] **Step 4: Run the new UI tests and confirm missing-provider/state failures**

Run:

```powershell
python -m unittest tests.test_ui.JitterLayoutTests.test_ai_runtime_receives_thread_safe_trigger_epoch_provider -v
python -m unittest tests.test_ui.JitterLayoutTests.test_raw_trigger_release_and_repress_creates_new_epoch -v
python -m unittest tests.test_ui -v
```

Expected: new tests fail because the UI does not expose epochs or pass the provider; existing tests remain the baseline.

- [ ] **Step 5: Add runtime-only epoch state and provider**

Initialize under the existing AI snapshot lock:

```python
self._ai_lock = threading.RLock()
self._ai_snapshot: AimSettings = (
    self.config.ai
    if self.config.ai.target_area == "head"
    else replace(self.config.ai, target_area="head")
)
self._adaptive_zoom_gate = False
self._trigger_lock_counter = 0
self._trigger_lock_epoch: int | None = None
self._trigger_lock_owner: str | None = None
self._physical_buttons_down: set[str] = set()
```

Add main-thread helpers and a worker-safe reader:

```python
def get_trigger_lock_epoch(self) -> int | None:
    with self._ai_lock:
        return self._trigger_lock_epoch

def _publish_trigger_lock_epoch(
    self,
    epoch: int | None,
    owner: str | None,
) -> None:
    if (epoch is None) != (owner is None) or owner not in {
        None, "normal", "test",
    }:
        raise ValueError("Trigger-lock epoch and owner must agree")
    with self._ai_lock:
        self._trigger_lock_epoch = epoch
        self._trigger_lock_owner = owner

def _clear_trigger_lock_epoch(
    self,
    expected_owner: str | None = None,
) -> int | None:
    with self._ai_lock:
        if (
            expected_owner is not None
            and self._trigger_lock_owner != expected_owner
        ):
            return None
        epoch = self._trigger_lock_epoch
        self._trigger_lock_epoch = None
        self._trigger_lock_owner = None
        return epoch

def _next_trigger_lock_epoch(self) -> int:
    with self._ai_lock:
        self._trigger_lock_counter += 1
        return self._trigger_lock_counter
```

Do not read Tk variables in `get_trigger_lock_epoch`.

- [ ] **Step 6: Implement ordered begin/end/invalidate helpers**

```python
def _raw_trigger_press_eligible(self) -> bool:
    return bool(
        not self._closing
        and self.service.connected
        and self.master_armed
        and self.ai_selected
        and self._motion_mode is None
    )

def _reset_targeting_for_trigger_lock(self, context: str) -> bool:
    try:
        revision = self.ai_service.reset_targeting()
    except Exception:
        logging.exception("AI targeting reset failed during %s", context)
        self._clear_trigger_lock_epoch()
        self.footer_var.set("AI target lock stopped; check app.log")
        return False
    self._ai_targeting_revision = revision
    return True

def _begin_trigger_lock_epoch(self) -> None:
    with self._ai_lock:
        if self._trigger_lock_owner is not None:
            return
    if not self._raw_trigger_press_eligible():
        return
    if not self._reset_targeting_for_trigger_lock("Trigger press"):
        return
    self._publish_trigger_lock_epoch(
        self._next_trigger_lock_epoch(), "normal"
    )

def _end_trigger_lock_epoch(self) -> None:
    epoch = self._clear_trigger_lock_epoch("normal")
    if epoch is None:
        return
    self._reset_targeting_for_trigger_lock("Trigger release")

def _retire_owned_trigger_lock_epoch(self) -> None:
    epoch = self._clear_trigger_lock_epoch()
    if epoch is None:
        return
    self._reset_targeting_for_trigger_lock("Trigger lock retirement")

def _invalidate_trigger_lock_epoch(self) -> None:
    epoch = self.get_trigger_lock_epoch()
    try:
        revision = self.ai_service.invalidate_trigger_lock(epoch)
    except Exception:
        logging.exception("AI Trigger lock invalidation failed")
        self._clear_trigger_lock_epoch()
        self.footer_var.set("AI target lock stopped; check app.log")
        return
    self._ai_targeting_revision = revision
```

The two exception paths above are the exact UI boundary: both log the detailed
diagnostic, clear the exposed epoch before returning, and leave the concise
footer action for the user. They must not publish a new epoch after a failed
reset or let an exception escape the Tk callback.

- [ ] **Step 7: Detect actual raw Trigger edges before the combined gate**

In the `button` event branch, deduplicate physical button edges independently of `TriggerGate`, then update the normal movement gate:

```python
button = str(button)
pressed = bool(pressed)
was_physically_down = button in self._physical_buttons_down
if pressed:
    self._physical_buttons_down.add(button)
else:
    self._physical_buttons_down.discard(button)
self.trigger_gate.update_button(button, pressed)
is_trigger_button = button == self.trigger_gate.trigger
if self._motion_mode not in _TEST_MOTION_MODES:
    if is_trigger_button and pressed and not was_physically_down:
        self._begin_trigger_lock_epoch()
    elif is_trigger_button and not pressed and was_physically_down:
        self._end_trigger_lock_epoch()
self._sync_adaptive_zoom_gate()
```

Only after this ordering may the existing normal-motion start/stop logic run. `TriggerGate.clear()` never clears `_physical_buttons_down`; only an actual button-up callback removes that physical latch. Duplicate down events after Master/source/reconnect/STOP state changes therefore never allocate another epoch. Modifier edges never call either helper. During every Test motion/loading mode, physical Trigger events may update `_physical_buttons_down` and `TriggerGate`, but they never begin or end the synthetic epoch.

- [ ] **Step 8: Pass the provider to every production AI generation**

Change `_start_ai_runtime`:

```python
generation = self.ai_service.start(
    self.get_ai_settings,
    self.get_adaptive_zoom_gate,
    self.get_trigger_lock_epoch,
    model_path=choice.path,
    capture_mode=generation_capture_mode,
)
```

All deferred starts, capture switches, model rollback, and Overlay demand flow through this method and therefore receive the same provider.

- [ ] **Step 9: Integrate normal lifecycle invalidation**

Apply these exact rules:

- Master disable, STOP, disconnect, hotkey disable, shutdown, and AI error: call `_retire_owned_trigger_lock_epoch()` before stopping/canceling movement. It clears either normal or Test ownership and resets targeting once.
- Every motion-source set change calls `_end_trigger_lock_epoch()` before changing the source booleans, including adding/removing Jitter while AI remains selected and adding/removing AI. Controls remain unavailable during Test, so this can retire only normal ownership.
- Binding changes call `_end_trigger_lock_epoch()` before `TriggerGate.configure()`. Keep `_physical_buttons_down` intact so a Trigger already held under either the old or new binding cannot become a new edge until its actual up callback.
- Confidence or Target Area change: call `_invalidate_trigger_lock_epoch()` after replacing valid settings and keep the same exposed epoch so the service renders `LOST`/`NONE`.
- Strength, Smoothing, Max Step, and response-curve changes: update settings without invalidation.
- Capture/model switch while an epoch is exposed: call `_invalidate_trigger_lock_epoch()` before retiring the worker so a successor generation cannot claim it.
- Raw Trigger release: clear provider first, then reset targeting, then stop motion.
- Never clear `_physical_buttons_down` from Master, source, reconnect, STOP, disconnect, AI-error, or Test cleanup paths. Those paths may clear `TriggerGate` for motion safety; the independent physical set is released only by actual button-up events.

Use the owner-aware helpers instead of scattering direct `_trigger_lock_epoch` or `_trigger_lock_owner` assignments.

- [ ] **Step 10: Add one synthetic epoch to AI Test 3s**

In `start_test_run`, call `_end_trigger_lock_epoch()` before its existing `trigger_gate.clear()` so a normal held-Trigger epoch is retired before Test owns movement. This is a release of the normal request, not the synthetic allocation. Do not clear `_physical_buttons_down`.

In `_begin_test_motion`, before `_request_motion_start` when `sources.ai`:

```python
if not self._reset_targeting_for_trigger_lock("AI Test 3s start"):
    return False
self._publish_trigger_lock_epoch(
    self._next_trigger_lock_epoch(), "test"
)
```

At the start of `_restore_after_test`, clear only Test ownership and use the
same exception-safe reset helper when it returned an epoch:

```python
test_epoch = self._clear_trigger_lock_epoch("test")
if test_epoch is not None:
    self._reset_targeting_for_trigger_lock("AI Test 3s cleanup")
```

Always call `trigger_gate.clear()` before restoring normal Master state so a Trigger pressed during Test cannot start normal motion after Test; retain `_physical_buttons_down`, requiring its actual release and a new down edge. `_abort_test_run` already flows through `_restore_after_test`. STOP, disconnect, and AI-error paths use `_retire_owned_trigger_lock_epoch()` and therefore also destroy Test ownership. Jitter-only Test 3s keeps the provider `None`. Do not enable Adaptive Zoom during tests.

- [ ] **Step 11: Run the complete UI and service interaction suites**

Run:

```powershell
python -m unittest tests.test_ui tests.test_ai_service tests.test_makcu_service -v
```

Expected: all tests pass without Tk access from worker providers, stale targets, duplicate epochs, or changed four-tuple indexes in `StubAiService.start_calls`.

- [ ] **Step 12: Commit Task 4**

```powershell
git add -- jitter_app/presentation/ui.py tests/test_ui.py
git diff --cached --check
git commit -m "feat: bind AI target locks to trigger presses"
```

---

### Task 5: Update the active repository contract and user documentation

**Files:**
- Modify: `AGENTS.md:35-40,148-153,181-190`
- Modify: `README.md:150-214`

**Interfaces:**
- Consumes: completed Tasks 1-4 behavior and the approved design spec.
- Produces: one current, non-contradictory repository contract and accurate user-facing defaults; historical completed specs/plans remain unchanged.

- [ ] **Step 1: Replace the superseded stateless production rule in `AGENTS.md`**

Update the tracking layout description to state that the module owns both the legacy compatibility tracker and pure Strict Trigger Lock.

Replace the current-frame-only movement paragraph with this exact requirement text:

```text
Outside an eligible raw-Trigger epoch, base selection remains current-frame
nearest for Overlay visualization and initial acquisition. During an eligible
raw-Trigger press, perform at most one acquisition, follow only one unique
same-class geometrically plausible base continuation, and latch LOST on no
match or multiple plausible matches. LOST publishes no AI movement and no
selected Overlay index until Trigger-up followed by Trigger-down creates a new
epoch. Modifier cycling is not a new epoch. Never describe box association as
guaranteed physical identity.
```

Update the Adaptive Zoom bullets so stability observes the strict locked base target during an epoch, refinement never changes next-frame identity state, and stateless base selection applies only outside an active epoch.

- [ ] **Step 2: Update README defaults and Trigger behavior**

Use exact displayed defaults:

```text
Confidence 0.25
Aim Strength 0.35
Smoothing 0.58
Max Step 18
Curve 0% / 16% / 38% / 68% / 95%
```

Add a concise Trigger Lock paragraph explaining that a missing/ambiguous target stops AI assistance for the remainder of the press and that releasing/repressing Trigger starts a new acquisition. Explain that Modifier release alone does not choose a new target. Do not claim face/person identification.

- [ ] **Step 3: Check active docs for contradictory current claims**

Run:

```powershell
rg -n "stateless|current-frame|replacement|0\.65|Max Step.*20|12%.*35%.*68%.*100%" AGENTS.md README.md
```

Expected: no active statement claims movement always switches current-frame during a Trigger epoch, and no active defaults retain the old values. Dated historical specs/plans are allowed to retain their original decisions.

- [ ] **Step 4: Run documentation-adjacent contract tests**

Run:

```powershell
python -m unittest tests.test_package_layout tests.test_entrypoints tests.test_distribution_metadata -v
```

Expected: all pass; no source/package layout, dependency, or release-material changes.

- [ ] **Step 5: Commit Task 5**

```powershell
git add -- AGENTS.md README.md
git diff --cached --check
git commit -m "docs: describe strict trigger target locking"
```

---

### Task 6: Cross-task review and complete verification

**Files:**
- Review: every file committed by Tasks 1-5
- Do not modify: `models/*.onnx`, `config.json`, `config.json.bak`, `app.log`, build outputs, caches, or distribution artifacts

**Interfaces:**
- Consumes: the complete integrated implementation.
- Produces: evidence that the source compiles, all tests pass, approved dependencies import, DirectML self-check passes, distribution review passes, and only intended files are tracked.

- [ ] **Step 1: Review the implementation against every spec section**

Check these invariants directly in code and tests:

```text
one claim per epoch
first missing/non-unique continuation -> irreversible LOST
raw Trigger release/repress is the only normal reacquisition boundary
Modifier does not allocate an epoch
successor generation cannot reclaim a held epoch
managed idle publishes Overlay frame but no movement target
unmanaged AiService callers preserve stateless publication
refined boxes never update next-frame association
Jitter continues when AI target is None
schema 5 and valid user settings remain unchanged
no Tk variable access from worker providers
```

If an invariant lacks a deterministic test, record a finding in the SDD ledger and dispatch it to the lane owning that production file. That implementer adds the red test first, records the expected failure, makes the smallest correction, runs the owning focused suite, commits only its enumerated files, and enters the normal scoped review gate before Task 6 resumes.

- [ ] **Step 2: Run syntax compilation**

```powershell
$jitterSources = @('main.py', 'distribution_metadata.py') + @(Get-ChildItem -LiteralPath 'jitter_app' -Recurse -Filter '*.py' | Sort-Object FullName | ForEach-Object { $_.FullName })
python -m py_compile @jitterSources
```

Expected: exit code 0 with no output.

- [ ] **Step 3: Run the complete unit/integration-style suite**

```powershell
python -m unittest discover -s tests -v
```

Expected: every test passes. Do not hide failures, weaken strict-lock assertions, or raise arbitrary timeouts.

- [ ] **Step 4: Verify the pinned runtime imports**

```powershell
python -c "import makcu, serial, pygame, onnxruntime, dxcam, comtypes, numpy"
```

Expected: exit code 0.

- [ ] **Step 5: Run the DirectML AI runtime self-check**

```powershell
python .\main.py --ai-runtime-self-check
```

Expected: success using `DmlExecutionProvider` and the bundled startup model.

- [ ] **Step 6: Review the distribution plan without building**

```powershell
python .\distribution_metadata.py --review-json
```

Expected: valid JSON review with the canonical Nuitka config, DirectML self-check step, and complete release-material plan. Do not run `gen.bat`.

- [ ] **Step 7: Audit Git scope and untracked models**

```powershell
git status --short --branch
git diff --check
git log --oneline -8
$protectedModels = @(
    'models/Apex_20k_pictures_640.onnx',
    'models/all_games.onnx',
    'models/all_games_128.onnx',
    'models/all_games_256.onnx',
    'models/all_games_640.onnx'
)
$protectedModels | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    $hash = Get-FileHash -LiteralPath $_ -Algorithm SHA256
    [pscustomobject]@{
        Path = $_
        Length = $item.Length
        LastWriteTimeUtc = $item.LastWriteTimeUtc.ToString('O')
        SHA256 = $hash.Hash
    }
} | ConvertTo-Json
```

Expected: only intentional source/test/docs commits are present; the exact five user-owned external `.onnx` paths remain untracked, and every length, timestamp, and SHA-256 value matches Task 0's ledger baseline.

- [ ] **Step 8: Run the fresh whole-branch review gate**

Invoke `superpowers:requesting-code-review`. Generate the SDD review package from `INTEGRATION_BASE` to current `HEAD` and dispatch one fresh whole-branch reviewer with the approved spec, this plan, the package, and every deferred/ruled ledger entry. Require explicit spec-compliance and code-quality verdicts covering Tasks 1-5.

Critical/Important findings return to the owning A/B/C lane, use the SDD fix loop, run their exact focused suites, and receive a scoped re-review. The coordinator never edits implementation code. Do not proceed with an open load-bearing finding.

- [ ] **Step 9: Record hardware verification as pending unless Makcu is connected**

With hardware, verify AI-only and combined movement, Trigger/Modifier cycling, crossing/loss/new-Trigger reacquisition, both capture modes, Adaptive Zoom, Test 3s, Overlay, model/capture switch, reconnect, STOP, hotkey, and shutdown. Without hardware, report these checks explicitly as not performed; do not claim hardware success from mocks.

- [ ] **Step 10: Commit only reviewed corrections through their owning lane**

No omnibus `git add -- jitter_app tests` is allowed. A corrective implementer may stage only the exact subset of its ownership list:

```text
Lane A: jitter_app/ai/targeting.py
        tests/test_ai_targeting.py
        tests/test_settings.py
        tests/test_ui.py (default-control assertions only)
        AGENTS.md
        README.md
Lane B: jitter_app/ai/tracking.py
        tests/test_ai_tracking.py
Lane C: jitter_app/ai/service.py
        jitter_app/ai/zoom.py
        jitter_app/presentation/ui.py
        tests/test_ai_service.py
        tests/test_ai_zoom.py
        tests/test_ui.py (Trigger/runtime lifecycle assertions only)
```

Before every corrective commit run:

```powershell
git diff --cached --name-only
git diff --cached --check
```

The staged list must equal the exact files named in that finding's fix report and be a subset of its lane above. After scoped re-review/integration, rerun Steps 2-8 and the protected-model comparison. If no correction exists, do not create an empty commit.

- [ ] **Step 11: Complete only after evidence is current**

Invoke `superpowers:verification-before-completion` before any success claim, then `superpowers:finishing-a-development-branch` for the handoff. Do not push unless the user explicitly requests it.
