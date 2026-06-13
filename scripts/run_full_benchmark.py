"""Full-scale benchmark launcher (70 jobs: 14 algos x V1.1-V1.4 generalization
+ the V1.4 E2 misalignment study). Runs each driver command via `zsh -c`
with a concurrency cap, per-job logs, and a live progress file; records
DONE/FAILED by exit code. Portable (paths derived from this file).

Run from the repo root:
    .venv/bin/python scripts/run_full_benchmark.py
"""
import subprocess, time, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOGDIR = ROOT / "results/full_benchmark/logs"
LOGDIR.mkdir(parents=True, exist_ok=True)
PROGRESS = LOGDIR / "full_progress.log"
MAXPROC = 6

TAB = ["q_learning", "sarsa", "expected_sarsa", "spie_q"]
DEEP = ["dqn", "ddqn", "dueling_dqn", "dueling_ddqn_per",
        "noisy_dueling_ddqn_per", "spie_dqn", "spie_ddqn", "spie_dueling_dqn",
        "spie_dueling_ddqn_per", "spie_noisy_dueling_ddqn_per"]
VERS = ["v1_1", "v1_2", "v1_3", "v1_4"]
VMAP = {"v1_1": "V1.1", "v1_2": "V1.2", "v1_3": "V1.3", "v1_4": "V1.4"}
PY = ".venv/bin/python scripts/train_axiom_forge.py"
PRE = "OMP_NUM_THREADS=2 MKL_NUM_THREADS=2"
SEEDS = "--seeds 0 1 2 3 4 --train-configs 16 --heldout-configs 12"

jobs = []  # (tag, command)


def add(tag, body):
    log = LOGDIR / f"{tag}.log"
    jobs.append((tag, f"{PRE} {PY} {body} > {log} 2>&1"))


for v in VERS:
    for a in TAB:
        add(f"{v}_{a}", f"--version {v} --algo {a} --episodes 3000 {SEEDS} "
            f"--demo-configs 16 --demo-sweeps 4 --epsilon-start 0.1 "
            f"--results-dir results/full_benchmark/{VMAP[v]}/{a}")
for v in VERS:
    for a in DEEP:
        add(f"{v}_{a}", f"--version {v} --algo {a} --episodes 200 {SEEDS} "
            f"--demo-configs 16 --pretrain-steps 1000 --epsilon-start 0.2 "
            f"--buffer-size 50000 --device cpu "
            f"--results-dir results/full_benchmark/{VMAP[v]}/{a}")
for a in TAB:
    add(f"e2_{a}", f"--version v1_4 --algo {a} --episodes 3000 {SEEDS} "
        f"--demo-configs 0 --e2 --epsilon-start 1.0 "
        f"--results-dir results/full_benchmark/V1.4_e2/{a}")
for a in DEEP:
    add(f"e2_{a}", f"--version v1_4 --algo {a} --episodes 400 {SEEDS} "
        f"--demo-configs 0 --e2 --epsilon-start 1.0 --buffer-size 50000 "
        f"--device cpu --results-dir results/full_benchmark/V1.4_e2/{a}")

# Deep jobs first-ish? Keep generation order; pool handles balance.
start = time.time()
running = {}   # Popen -> tag
done, failed = [], []
queue = list(jobs)
total = len(queue)


def log_progress():
    with open(PROGRESS, "w") as f:
        f.write(f"elapsed {int(time.time()-start)}s | total {total} | "
                f"done {len(done)} | failed {len(failed)} | "
                f"running {len(running)} | queued {len(queue)}\n")
        for t in done:
            f.write(f"DONE {t}\n")
        for t in failed:
            f.write(f"FAILED {t}\n")


while queue or running:
    while queue and len(running) < MAXPROC:
        tag, cmd = queue.pop(0)
        p = subprocess.Popen(["zsh", "-c", cmd], cwd=str(ROOT))
        running[p] = tag
    time.sleep(2)
    for p in list(running):
        rc = p.poll()
        if rc is not None:
            tag = running.pop(p)
            (done if rc == 0 else failed).append(tag)
            log_progress()

log_progress()
with open(PROGRESS, "a") as f:
    f.write(f"\nALL COMPLETE in {int(time.time()-start)}s | "
            f"done {len(done)}/{total} | failed {failed}\n")
print(f"done {len(done)}/{total}, failed {failed}")
