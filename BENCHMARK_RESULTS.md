# Four-Obstacle Headless Simulation Results

Evaluation date: 2026-09-21. This is a source-only, simulation-only research record.
Robot geometry, private logs, trajectories, weights, and hardware drawings are not included.

## Latest Fixed Batch

**79/80 successful trials (98.75%).**
Per-obstacle counts, uncertainty, and every failure are listed below.
The latest batch uses new seeds after development; no additional trials were appended
to dilute failures. The target is a per-class sample rate >=95%, not strictly >95%
in every class and not a confidence guarantee on the underlying probability.

## Scope and Acceptance

Each obstacle is tested in isolation using the analytical Phi decoder, scripted brain,
MPC, and swerve execution. This is not a compound-course, learned-policy, or physical-robot result.
The obstacle geometry and success criteria are unchanged between the randomized batches.

The nominal smoke batch is a development check. Randomized batches perturb initial x/y
by up to 0.03 m, initial yaw by up to 3 degrees, and wheel/ground friction over [0.9,1.1].
All batch sizes and seeds are written before running; failures remain in the denominator.

Success requires full-body clearance, no corridor bypass, goal distance <0.20 m,
actual seven-joint error <=0.05 rad relative to the required morphology, and at least
two ground-contacting wheels for ten consecutive 0.05 s control steps. Contact force >1 N
or penetration >2 mm with the obstacle at any 240 Hz physics tick fails the trial.
The timeout is 40 simulated seconds. Platform contact is forbidden in this straddling task.
These criteria do not certify general self-collision safety or hardware stability.

The headless policy pauses motion during shape changes and uses a geometric heading
estimate near Phi=3. It uses finite motor commands, without the GUI demo's per-tick
joint-state-reset patch. The revised policy additionally aligns the ring configuration
sideways before platform approach and morphing. The original required Phi=2 is retained.

## four_obstacles_smoke_20260921

Trials: 4. Nominal only: True. Platform pre-alignment: False. Platform Phi: 2.

| Obstacle | Successes / trials | Sample rate | 95% Wilson interval | Failures |
| --- | ---: | ---: | ---: | --- |
| narrow_gate | 1/1 | 100.0% | [20.7%, 100.0%] | None |
| low_platform | 0/1 | 0.0% | [0.0%, 79.3%] | obstacle_contact: 1 |
| bridge | 1/1 | 100.0% | [20.7%, 100.0%] | None |
| cylinders | 1/1 | 100.0% | [20.7%, 100.0%] | None |

At least 95% sample success in **every** obstacle class: **NO**.

A 100% sample rate from 20 trials does not establish a true success probability
of at least 95% at 95% confidence. Rates are not pooled across controller versions.

Record fingerprints (raw records retained privately):

- `protocol.json` SHA-256: `e3ec144536785c62074b42ebdbc180eda0aa4b28517639f4868f7c3c4faa5b2b`
- `summary.json` SHA-256: `af2d44be7ae6e4baea7d102e4fa68dbeebfe29a09678a16473c107b7db420595`
- Evaluated `scripts/eval_four_obstacles.py` SHA-256: `09b1f260736cfa9bfdfa9b57ddf3ce75997b97008a84fa4020ab5d42939c1352`
- Evaluated `src/evaluation/obstacle_protocol.py` SHA-256: `251f0d7b477db04d49c675764afcfd8832671118f73182b4ea69ecd9a32cdde9`

<details>
<summary>All trial outcomes (sanitized scalar fields)</summary>

| Obstacle | Seed | Outcome | Simulated seconds |
| --- | ---: | --- | ---: |
| narrow_gate | 2026092100 | success | 15.10 |
| low_platform | 2026102100 | obstacle_contact | 8.15 |
| bridge | 2026112100 | success | 14.75 |
| cylinders | 2026122100 | success | 11.65 |

</details>

## four_obstacles_batch_v1_20260921

Trials: 80. Nominal only: False. Platform pre-alignment: False. Platform Phi: 2.

| Obstacle | Successes / trials | Sample rate | 95% Wilson interval | Failures |
| --- | ---: | ---: | ---: | --- |
| narrow_gate | 20/20 | 100.0% | [83.9%, 100.0%] | None |
| low_platform | 0/20 | 0.0% | [0.0%, 16.1%] | obstacle_contact: 15, timeout: 5 |
| bridge | 20/20 | 100.0% | [83.9%, 100.0%] | None |
| cylinders | 20/20 | 100.0% | [83.9%, 100.0%] | None |

At least 95% sample success in **every** obstacle class: **NO**.

A 100% sample rate from 20 trials does not establish a true success probability
of at least 95% at 95% confidence. Rates are not pooled across controller versions.

Record fingerprints (raw records retained privately):

- `protocol.json` SHA-256: `5351cb62bf200aedef727ce38cfc3ab57a9e08a3828aea404b7d471411c1d620`
- `summary.json` SHA-256: `cd027e772b8a8b7214c799b75d4697ae59c17e49306331a4559a1df9240184b2`
- Evaluated `scripts/eval_four_obstacles.py` SHA-256: `898970681fdf28ace223f8bbbf5dc864a49360afb39af75246a3ea3e52b166f9`
- Evaluated `src/evaluation/obstacle_protocol.py` SHA-256: `251f0d7b477db04d49c675764afcfd8832671118f73182b4ea69ecd9a32cdde9`

<details>
<summary>All trial outcomes (sanitized scalar fields)</summary>

| Obstacle | Seed | Outcome | Simulated seconds |
| --- | ---: | --- | ---: |
| narrow_gate | 2026092100 | success | 15.20 |
| narrow_gate | 2026092101 | success | 15.00 |
| narrow_gate | 2026092102 | success | 15.15 |
| narrow_gate | 2026092103 | success | 15.05 |
| narrow_gate | 2026092104 | success | 15.20 |
| narrow_gate | 2026092105 | success | 15.05 |
| narrow_gate | 2026092106 | success | 15.05 |
| narrow_gate | 2026092107 | success | 15.05 |
| narrow_gate | 2026092108 | success | 15.15 |
| narrow_gate | 2026092109 | success | 15.15 |
| narrow_gate | 2026092110 | success | 15.10 |
| narrow_gate | 2026092111 | success | 15.15 |
| narrow_gate | 2026092112 | success | 15.20 |
| narrow_gate | 2026092113 | success | 15.10 |
| narrow_gate | 2026092114 | success | 15.15 |
| narrow_gate | 2026092115 | success | 15.05 |
| narrow_gate | 2026092116 | success | 15.15 |
| narrow_gate | 2026092117 | success | 15.05 |
| narrow_gate | 2026092118 | success | 15.10 |
| narrow_gate | 2026092119 | success | 15.00 |
| low_platform | 2026102100 | obstacle_contact | 9.10 |
| low_platform | 2026102101 | obstacle_contact | 7.20 |
| low_platform | 2026102102 | obstacle_contact | 6.95 |
| low_platform | 2026102103 | timeout | 40.00 |
| low_platform | 2026102104 | obstacle_contact | 6.95 |
| low_platform | 2026102105 | timeout | 40.00 |
| low_platform | 2026102106 | obstacle_contact | 7.65 |
| low_platform | 2026102107 | obstacle_contact | 6.80 |
| low_platform | 2026102108 | obstacle_contact | 8.00 |
| low_platform | 2026102109 | obstacle_contact | 7.25 |
| low_platform | 2026102110 | obstacle_contact | 12.65 |
| low_platform | 2026102111 | obstacle_contact | 7.95 |
| low_platform | 2026102112 | obstacle_contact | 7.00 |
| low_platform | 2026102113 | timeout | 40.00 |
| low_platform | 2026102114 | timeout | 40.00 |
| low_platform | 2026102115 | timeout | 40.00 |
| low_platform | 2026102116 | obstacle_contact | 7.20 |
| low_platform | 2026102117 | obstacle_contact | 38.50 |
| low_platform | 2026102118 | obstacle_contact | 6.95 |
| low_platform | 2026102119 | obstacle_contact | 10.35 |
| bridge | 2026112100 | success | 14.75 |
| bridge | 2026112101 | success | 14.75 |
| bridge | 2026112102 | success | 14.65 |
| bridge | 2026112103 | success | 14.65 |
| bridge | 2026112104 | success | 14.70 |
| bridge | 2026112105 | success | 14.70 |
| bridge | 2026112106 | success | 14.75 |
| bridge | 2026112107 | success | 14.75 |
| bridge | 2026112108 | success | 14.70 |
| bridge | 2026112109 | success | 14.80 |
| bridge | 2026112110 | success | 14.75 |
| bridge | 2026112111 | success | 14.80 |
| bridge | 2026112112 | success | 14.70 |
| bridge | 2026112113 | success | 14.65 |
| bridge | 2026112114 | success | 14.80 |
| bridge | 2026112115 | success | 14.80 |
| bridge | 2026112116 | success | 14.70 |
| bridge | 2026112117 | success | 14.70 |
| bridge | 2026112118 | success | 14.70 |
| bridge | 2026112119 | success | 14.70 |
| cylinders | 2026122100 | success | 12.30 |
| cylinders | 2026122101 | success | 12.25 |
| cylinders | 2026122102 | success | 11.50 |
| cylinders | 2026122103 | success | 11.70 |
| cylinders | 2026122104 | success | 11.50 |
| cylinders | 2026122105 | success | 11.75 |
| cylinders | 2026122106 | success | 11.60 |
| cylinders | 2026122107 | success | 11.45 |
| cylinders | 2026122108 | success | 12.00 |
| cylinders | 2026122109 | success | 11.50 |
| cylinders | 2026122110 | success | 11.70 |
| cylinders | 2026122111 | success | 11.80 |
| cylinders | 2026122112 | success | 11.70 |
| cylinders | 2026122113 | success | 11.75 |
| cylinders | 2026122114 | success | 11.40 |
| cylinders | 2026122115 | success | 11.60 |
| cylinders | 2026122116 | success | 12.30 |
| cylinders | 2026122117 | success | 11.60 |
| cylinders | 2026122118 | success | 12.65 |
| cylinders | 2026122119 | success | 12.85 |

</details>

## four_obstacles_batch_v2_20260921

Trials: 80. Nominal only: False. Platform pre-alignment: True. Platform Phi: 2.

| Obstacle | Successes / trials | Sample rate | 95% Wilson interval | Failures |
| --- | ---: | ---: | ---: | --- |
| narrow_gate | 20/20 | 100.0% | [83.9%, 100.0%] | None |
| low_platform | 20/20 | 100.0% | [83.9%, 100.0%] | None |
| bridge | 20/20 | 100.0% | [83.9%, 100.0%] | None |
| cylinders | 19/20 | 95.0% | [76.4%, 99.1%] | timeout: 1 |

At least 95% sample success in **every** obstacle class: **YES**.

A 100% sample rate from 20 trials does not establish a true success probability
of at least 95% at 95% confidence. Rates are not pooled across controller versions.

Record fingerprints (raw records retained privately):

- `protocol.json` SHA-256: `124bbcdc1a0a7151ae4097cc9d8f92a8c897a1056a34b43d3603eb19e3eded2f`
- `summary.json` SHA-256: `5a35b5476358f82d8857842317c47d50fca26f2aab0255c0bb0465712f7e8759`
- Evaluated `scripts/eval_four_obstacles.py` SHA-256: `af6d3e420dff379895b7a3f6fb869c66a020655f9389670318236cae83c907f3`
- Evaluated `src/evaluation/obstacle_protocol.py` SHA-256: `251f0d7b477db04d49c675764afcfd8832671118f73182b4ea69ecd9a32cdde9`

<details>
<summary>All trial outcomes (sanitized scalar fields)</summary>

| Obstacle | Seed | Outcome | Simulated seconds |
| --- | ---: | --- | ---: |
| narrow_gate | 2026093100 | success | 15.15 |
| narrow_gate | 2026093101 | success | 15.05 |
| narrow_gate | 2026093102 | success | 15.10 |
| narrow_gate | 2026093103 | success | 15.10 |
| narrow_gate | 2026093104 | success | 15.10 |
| narrow_gate | 2026093105 | success | 17.05 |
| narrow_gate | 2026093106 | success | 15.05 |
| narrow_gate | 2026093107 | success | 15.10 |
| narrow_gate | 2026093108 | success | 15.00 |
| narrow_gate | 2026093109 | success | 15.15 |
| narrow_gate | 2026093110 | success | 15.05 |
| narrow_gate | 2026093111 | success | 15.05 |
| narrow_gate | 2026093112 | success | 15.00 |
| narrow_gate | 2026093113 | success | 15.10 |
| narrow_gate | 2026093114 | success | 15.10 |
| narrow_gate | 2026093115 | success | 15.00 |
| narrow_gate | 2026093116 | success | 15.05 |
| narrow_gate | 2026093117 | success | 15.15 |
| narrow_gate | 2026093118 | success | 14.95 |
| narrow_gate | 2026093119 | success | 15.10 |
| low_platform | 2026103100 | success | 19.30 |
| low_platform | 2026103101 | success | 19.55 |
| low_platform | 2026103102 | success | 19.20 |
| low_platform | 2026103103 | success | 19.30 |
| low_platform | 2026103104 | success | 19.65 |
| low_platform | 2026103105 | success | 19.15 |
| low_platform | 2026103106 | success | 19.35 |
| low_platform | 2026103107 | success | 19.45 |
| low_platform | 2026103108 | success | 19.05 |
| low_platform | 2026103109 | success | 19.15 |
| low_platform | 2026103110 | success | 19.05 |
| low_platform | 2026103111 | success | 19.55 |
| low_platform | 2026103112 | success | 19.50 |
| low_platform | 2026103113 | success | 19.00 |
| low_platform | 2026103114 | success | 19.25 |
| low_platform | 2026103115 | success | 19.40 |
| low_platform | 2026103116 | success | 19.35 |
| low_platform | 2026103117 | success | 19.45 |
| low_platform | 2026103118 | success | 19.10 |
| low_platform | 2026103119 | success | 19.25 |
| bridge | 2026113100 | success | 14.60 |
| bridge | 2026113101 | success | 14.70 |
| bridge | 2026113102 | success | 14.75 |
| bridge | 2026113103 | success | 14.65 |
| bridge | 2026113104 | success | 14.75 |
| bridge | 2026113105 | success | 14.75 |
| bridge | 2026113106 | success | 14.75 |
| bridge | 2026113107 | success | 14.70 |
| bridge | 2026113108 | success | 14.80 |
| bridge | 2026113109 | success | 14.80 |
| bridge | 2026113110 | success | 14.70 |
| bridge | 2026113111 | success | 14.70 |
| bridge | 2026113112 | success | 14.75 |
| bridge | 2026113113 | success | 14.70 |
| bridge | 2026113114 | success | 14.65 |
| bridge | 2026113115 | success | 14.70 |
| bridge | 2026113116 | success | 14.85 |
| bridge | 2026113117 | success | 14.70 |
| bridge | 2026113118 | success | 14.60 |
| bridge | 2026113119 | success | 14.80 |
| cylinders | 2026123100 | success | 11.85 |
| cylinders | 2026123101 | success | 11.75 |
| cylinders | 2026123102 | success | 11.60 |
| cylinders | 2026123103 | success | 12.55 |
| cylinders | 2026123104 | success | 12.90 |
| cylinders | 2026123105 | success | 11.40 |
| cylinders | 2026123106 | success | 11.50 |
| cylinders | 2026123107 | timeout | 40.00 |
| cylinders | 2026123108 | success | 11.80 |
| cylinders | 2026123109 | success | 11.90 |
| cylinders | 2026123110 | success | 11.70 |
| cylinders | 2026123111 | success | 11.65 |
| cylinders | 2026123112 | success | 11.50 |
| cylinders | 2026123113 | success | 11.75 |
| cylinders | 2026123114 | success | 11.80 |
| cylinders | 2026123115 | success | 11.80 |
| cylinders | 2026123116 | success | 11.60 |
| cylinders | 2026123117 | success | 11.85 |
| cylinders | 2026123118 | success | 11.70 |
| cylinders | 2026123119 | success | 12.20 |

</details>

## Development Diagnostics

The following nominal parameter checks informed the revised controller. They are
not held-out trials and are excluded from every randomized success-rate denominator.

| Check | Seed | Platform Phi | Heading (deg) | Pre-align | Outcome |
| --- | ---: | ---: | ---: | --- | --- |
| platform_development_20260921 | 2026092200 | 1.8 | 0 | False | obstacle_contact |
| platform_development_20260921 | 2026092201 | 1.8 | 90 | False | obstacle_contact |
| platform_development_20260921 | 2026092202 | 1.9 | 0 | False | obstacle_contact |
| platform_development_20260921 | 2026092203 | 1.9 | 90 | False | timeout |
| platform_development_20260921 | 2026092204 | 2 | 0 | False | obstacle_contact |
| platform_development_20260921 | 2026092205 | 2 | 90 | False | obstacle_contact |
| platform_development_20260921 | 2026092206 | 2.1 | 0 | False | obstacle_contact |
| platform_development_20260921 | 2026092207 | 2.1 | 90 | False | obstacle_contact |
| platform_alignment_development_20260921 | 2026092300 | 1.85 | 90 | True | success |
| platform_alignment_development_20260921 | 2026092301 | 1.9 | 90 | True | success |
| platform_alignment_development_20260921 | 2026092302 | 1.95 | 90 | True | success |
| platform_alignment_development_20260921 | 2026092303 | 2 | 90 | True | success |

## Reproduction

Private local robot assets and the documented Python environment are required.
See the README for protocol details. Each output directory must be new.

```bash
python scripts/eval_four_obstacles.py --trials 20 --workers 4 --seed 2026092100 --output experiments/reproduce_v1
python scripts/eval_four_obstacles.py --trials 20 --workers 4 --seed 2026093100 --align-platform --output experiments/reproduce_v2
python -m unittest discover -s tests -v
```

The repository is evidence of implemented evaluation and reported results, not an
independently reproducible open-geometry benchmark. Hardware validation remains pending.
