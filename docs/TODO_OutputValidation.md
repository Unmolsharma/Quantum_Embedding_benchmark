# Logging Implementation Guide

---

## Directory Structure

    ember_results/
    ├── results/
    ├── logs/
    │   ├── runs/        # per-run logs: {algorithm}__{graph_id}__{trial}__{seed}.log
    │   └── runner/      # runner-level logs: {batch_id}.log
    └── metadata/

Create at batch start:

    def _init_log_dirs(output_dir: Path):
        for d in ['results', 'logs/runs', 'logs/runner', 'metadata']:
            (output_dir / d).mkdir(parents=True, exist_ok=True)

---

## Per-Run Log Files

One file per `(algorithm, graph_id, trial, seed)`. Open before the algorithm runs,
close after. Capture stdout, stderr, and runner diagnostics for that run only.

    import contextlib, io, logging

    def _run_log_path(log_dir: Path, algo: str, graph_id: str,
                      trial: int, seed: int) -> Path:
        return log_dir / 'runs' / f"{algo}__{graph_id}__{trial}__{seed}.log"

    @contextlib.contextmanager
    def _capture_run_output(log_path: Path):
        """Redirect stdout and stderr to log file for the duration of the block."""
        with open(log_path, 'w') as f:
            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                yield f

Usage in `benchmark_one()`:

    log_path = _run_log_path(log_dir, algo_name, graph_id, trial, seed)
    with _capture_run_output(log_path) as log_file:
        try:
            raw = algo.embed(source_graph, target_graph, timeout=timeout, seed=seed)
        except Exception as e:
            log_file.write(f"\n[RUNNER] CRASH: {traceback.format_exc()}\n")
            raw = {'embedding': {}, 'success': False, 'status': 'CRASH',
                   'error': traceback.format_exc()}
        
        # Runner appends its own diagnostics to the same file
        log_file.write(f"\n[RUNNER] status={raw.get('status')} "
                       f"success={raw.get('success')} "
                       f"wall_time={raw.get('time', 'N/A')}\n")

---

## Runner-Level Logger

One log file per batch. Records batch lifecycle, suspensions, and high-rate anomalies.
Always writes to stderr for CRASH and high INVALID_OUTPUT rates.

    def _setup_runner_logger(log_dir: Path, batch_id: str) -> logging.Logger:
        logger = logging.getLogger(f'ember.runner.{batch_id}')
        logger.setLevel(logging.DEBUG)

        # File handler — full detail
        fh = logging.FileHandler(log_dir / 'runner' / f'{batch_id}.log')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))

        # Stderr handler — warnings and above only
        sh = logging.StreamHandler()
        sh.setLevel(logging.WARNING)
        sh.setFormatter(logging.Formatter('[EMBER] %(levelname)s: %(message)s'))

        logger.addHandler(fh)
        logger.addHandler(sh)
        return logger

Log key events:

    logger.info(f"Batch {batch_id} started: {n_runs} runs planned")
    logger.warning(f"{algo_name} CRASH rate {rate:.0%} on {graph_class} — suspending")
    logger.warning(f"{algo_name} INVALID_OUTPUT rate {rate:.0%} on {graph_class} — suspending")
    logger.info(f"Batch {batch_id} complete: {n_success}/{n_runs} SUCCESS")

---

## Immediate Stderr Surfacing

CRASH and high-rate INVALID_OUTPUT must reach the terminal in real time,
not just the log file:

    SUSPENSION_THRESHOLD = 0.10   # suspend after >10% crash or invalid rate

    def _check_suspension(runner_logger, algo_name, graph_class,
                          status_counts: dict, n_runs: int):
        crash_rate   = status_counts.get('CRASH', 0) / max(n_runs, 1)
        invalid_rate = status_counts.get('INVALID_OUTPUT', 0) / max(n_runs, 1)

        if crash_rate > SUSPENSION_THRESHOLD:
            runner_logger.warning(
                f"{algo_name} CRASH rate {crash_rate:.0%} on {graph_class} "
                f"({status_counts['CRASH']}/{n_runs}) — suspending from this class"
            )
            return True

        if invalid_rate > SUSPENSION_THRESHOLD:
            runner_logger.warning(
                f"{algo_name} INVALID_OUTPUT rate {invalid_rate:.0%} on {graph_class} "
                f"({status_counts['INVALID_OUTPUT']}/{n_runs}) — suspending from this class"
            )
            return True

        return False

---

## Log Retrieval

    def get_run_log(output_dir: Path, algo: str, graph_id: str,
                    trial: int, seed: int) -> str:
        path = _run_log_path(output_dir / 'logs', algo, graph_id, trial, seed)
        if not path.exists():
            return f"No log found for {algo} / {graph_id} / trial {trial} / seed {seed}"
        return path.read_text()

Expose via CLI:

    # ember logs --algorithm gf_bolt --graph erdos_renyi_20 --trial 3 --seed 42
    @cli.command()
    def logs(algorithm, graph, trial, seed, output_dir):
        print(get_run_log(Path(output_dir), algorithm, graph, trial, seed))

---

## Retention Policy

Run at batch completion:

    def _cleanup_logs(log_dir: Path, results: list[EmbeddingResult]):
        keep_statuses = {'CRASH', 'INVALID_OUTPUT', 'TIMEOUT'}

        for result in results:
            if result.status in keep_statuses:
                continue   # keep — active debugging artifact

            path = _run_log_path(
                log_dir, result.algorithm_name,
                result.graph_id, result.trial, result.seed
            )
            if path.exists():
                path.unlink()

Runner logs are never auto-deleted. Per-run logs for SUCCESS, FAILURE,
SKIPPED, and INVALID_INPUT are deleted after the batch completes and
results are written to the database.

---

## What Algorithms Must Not Do

- Print to stdout or stderr directly — use the `error` field in the return dict
  for any diagnostic information that must survive the run
- Assume their output will be visible during benchmark execution
- Configure the root logger — use `logging.getLogger(__name__)` at DEBUG level only