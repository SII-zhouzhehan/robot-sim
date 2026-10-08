# Changelog

## v0.2.0 — 2026-10-08

- Use MuJoCo as the only physics backend for the robot, A/B/C scenes, and Gymnasium environment.
- Add a native viewer, browser control panel, model converter, scripted/PPO evaluator, and PPO trainer.
- Share Python 3.11 dependencies across Windows and Apple Silicon macOS.
- Remove redundant entry points and scripts tied to historical experiment directories; document retained scripts.
- Normalize source line endings and keep private assets, weights, outputs, and IDE settings out of Git.
- Validate with 26 automated tests plus one subtest, dependency checks, and native macOS viewer/headless runs.

Historical results from earlier simulator versions are not MuJoCo benchmark results.
