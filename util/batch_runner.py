from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set

import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
DONE_MARKER_NAME = ".batch_done.json"
SUMMARY_DIR_NAME = "_batch"
GENERATED_CONFIGS_DIR_NAME = "generated_configs"

TRAINING_ENTRYPOINTS = {
    "corrmatch.py",
    "dwl.py",
    "fixmatch.py",
    "fixmatch_pascal.py",
    "fixmatch_peft.py",
    "fixmatch_rgcr.py",
    "fixmatch_rgcrv2.py",
    "fixmatch_rgcrv3.py",
    "fixmatch_rgcrv4.py",
    "fixmatch_rgcrv5.py",
    "fixmatch_rgcrv6.py",
    "fixmatch_rvsc.py",
    "fixmatch_rvsc_moe.py",
    "rankmatch.py",
    "ranpaste.py",
    "scalematch.py",
    "scalematch_peft.py",
    "segmind.py",
    "supervised.py",
    "unimatch.py",
    "unimatch_v2.py",
    "unimatch_v2_rgcr.py",
    "unimatchv2_peft.py",
    "wscl.py",
}


class ManifestError(ValueError):
    """Raised when the batch manifest is invalid."""


@dataclass(frozen=True)
class BatchGlobalConfig:
    nproc_per_node: int
    save_root: Path
    port_base: int
    continue_on_error: bool = True
    max_retries: int = 1
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchJob:
    name: str
    script: Path
    config: Path
    config_overrides: Dict[str, Any]
    generated_config_path: Optional[Path]
    labeled_id_path: Path
    unlabeled_id_path: Path
    save_subdir: Path
    save_path: Path
    port: int
    extra_args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BatchManifest:
    manifest_path: Path
    global_config: BatchGlobalConfig
    jobs: List[BatchJob]


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def shell_join(parts: Sequence[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SemiFT training jobs unattended from a YAML manifest."
    )
    parser.add_argument("--manifest", required=True, type=str, help="Path to a YAML job manifest.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print commands without executing them.",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help="Comma-separated job names to execute from the manifest.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Override global.save_root from the manifest.",
    )
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Keep polling the manifest and execute newly appended jobs incrementally.",
    )
    parser.add_argument(
        "--poll-seconds",
        type=float,
        default=30.0,
        help="Polling interval in seconds when --watch is enabled.",
    )
    parser.add_argument(
        "--max-idle-polls",
        type=int,
        default=None,
        help="Stop watch mode after this many consecutive polls with no new jobs.",
    )
    return parser.parse_args(argv)


def resolve_path(value: str, *, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path

    repo_candidate = (REPO_ROOT / path).resolve()
    manifest_candidate = (base_dir / path).resolve()
    if repo_candidate.exists() or not manifest_candidate.exists():
        return repo_candidate
    return manifest_candidate


def load_yaml_mapping(path: Path, *, description: str) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ManifestError(f"{description} must be a mapping: {path}")
    return data


def require_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ManifestError(f"'{field_name}' must be a mapping.")
    return value


def require_str_list(value: Any, field_name: str) -> List[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ManifestError(f"'{field_name}' must be a list of strings.")
    return list(value)


def require_env_mapping(value: Any, field_name: str) -> Dict[str, str]:
    mapping = require_mapping(value, field_name)
    env: Dict[str, str] = {}
    for key, val in mapping.items():
        if not isinstance(key, str) or not isinstance(val, (str, int, float, bool)):
            raise ManifestError(
                f"'{field_name}' must map string keys to string-like values."
            )
        env[key] = str(val)
    return env


def require_overrides_mapping(value: Any, field_name: str) -> Dict[str, Any]:
    mapping = require_mapping(value, field_name)
    return deepcopy(mapping)


def ensure_existing_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise ManifestError(f"{description} does not exist: {path}")
    return path


def ensure_training_entrypoint(path: Path) -> Path:
    if path.name not in TRAINING_ENTRYPOINTS:
        allowed = ", ".join(sorted(TRAINING_ENTRYPOINTS))
        raise ManifestError(
            f"Unsupported training script '{path.name}'. Expected one of: {allowed}"
        )
    if path.parent != REPO_ROOT:
        raise ManifestError(f"Training script must live at the repo root: {path}")
    return ensure_existing_file(path, "Training script")


def parse_only_names(raw_only: Optional[str]) -> Optional[Set[str]]:
    if raw_only is None:
        return None
    names = {name.strip() for name in raw_only.split(",") if name.strip()}
    if not names:
        raise ManifestError("--only was provided but no valid job names were found.")
    return names


def sanitize_name(value: str) -> str:
    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return sanitized or "job"


def generated_config_dir(manifest: BatchManifest) -> Path:
    manifest_name = sanitize_name(manifest.manifest_path.stem)
    return (
        manifest.global_config.save_root
        / SUMMARY_DIR_NAME
        / GENERATED_CONFIGS_DIR_NAME
        / manifest_name
    )


def effective_config_path(job: BatchJob) -> Path:
    return job.generated_config_path or job.config


def deep_merge_dicts(base: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result = deepcopy(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def build_effective_config(job: BatchJob) -> Dict[str, Any]:
    base_config = load_yaml_mapping(job.config, description=f"Config for job '{job.name}'")
    if not job.config_overrides:
        return base_config
    return deep_merge_dicts(base_config, job.config_overrides)


def materialize_job_config(job: BatchJob) -> Path:
    if not job.config_overrides:
        return job.config
    if job.generated_config_path is None:
        raise ManifestError(f"Generated config path is missing for job '{job.name}'.")

    effective_config = build_effective_config(job)
    job.generated_config_path.parent.mkdir(parents=True, exist_ok=True)
    job.generated_config_path.write_text(
        yaml.safe_dump(effective_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return job.generated_config_path


def load_manifest(
    manifest_path: Path,
    *,
    save_root_override: Optional[Path] = None,
    only_names: Optional[Set[str]] = None,
    allow_no_jobs: bool = False,
    allow_missing_only_names: bool = False,
) -> BatchManifest:
    manifest_path = manifest_path.resolve()
    data = load_yaml_mapping(manifest_path, description="Manifest")
    manifest_dir = manifest_path.parent

    global_data = require_mapping(data.get("global"), "global")
    jobs_data = data.get("jobs")
    if not isinstance(jobs_data, list):
        raise ManifestError("'jobs' must be a list.")
    if not jobs_data and not allow_no_jobs:
        raise ManifestError("'jobs' must be a non-empty list.")

    missing_global = [
        key for key in ("nproc_per_node", "save_root", "port_base") if key not in global_data
    ]
    if missing_global:
        raise ManifestError(
            f"Missing required global fields: {', '.join(missing_global)}"
        )

    save_root = (
        save_root_override.resolve()
        if save_root_override is not None
        else resolve_path(str(global_data["save_root"]), base_dir=manifest_dir)
    )
    global_config = BatchGlobalConfig(
        nproc_per_node=int(global_data["nproc_per_node"]),
        save_root=save_root,
        port_base=int(global_data["port_base"]),
        continue_on_error=bool(global_data.get("continue_on_error", True)),
        max_retries=int(global_data.get("max_retries", 1)),
        env=require_env_mapping(global_data.get("env"), "global.env"),
    )
    if global_config.nproc_per_node < 1:
        raise ManifestError("global.nproc_per_node must be >= 1")
    if global_config.port_base < 1:
        raise ManifestError("global.port_base must be >= 1")
    if global_config.max_retries < 0:
        raise ManifestError("global.max_retries must be >= 0")

    jobs: List[BatchJob] = []
    seen_names: Set[str] = set()
    seen_save_subdirs: Set[Path] = set()
    generated_root = save_root / SUMMARY_DIR_NAME / GENERATED_CONFIGS_DIR_NAME / sanitize_name(manifest_path.stem)

    for index, raw_job in enumerate(jobs_data):
        if not isinstance(raw_job, dict):
            raise ManifestError(f"jobs[{index}] must be a mapping.")
        missing_job = [
            key
            for key in (
                "name",
                "script",
                "config",
                "labeled_id_path",
                "unlabeled_id_path",
                "save_subdir",
            )
            if key not in raw_job
        ]
        if missing_job:
            raise ManifestError(
                f"jobs[{index}] is missing required fields: {', '.join(missing_job)}"
            )

        name = str(raw_job["name"])
        if only_names is not None and name not in only_names:
            continue
        if name in seen_names:
            raise ManifestError(f"Duplicate job name: {name}")
        seen_names.add(name)

        save_subdir = Path(str(raw_job["save_subdir"]))
        if save_subdir in seen_save_subdirs:
            raise ManifestError(f"Duplicate save_subdir: {save_subdir}")
        seen_save_subdirs.add(save_subdir)

        script_path = ensure_training_entrypoint(
            resolve_path(str(raw_job["script"]), base_dir=manifest_dir)
        )
        config_path = ensure_existing_file(
            resolve_path(str(raw_job["config"]), base_dir=manifest_dir),
            f"Config for job '{name}'",
        )
        _ = load_yaml_mapping(config_path, description=f"Config for job '{name}'")
        config_overrides = require_overrides_mapping(
            raw_job.get("config_overrides"), f"jobs[{index}].config_overrides"
        )
        if config_overrides:
            merged_preview = deep_merge_dicts(
                load_yaml_mapping(config_path, description=f"Config for job '{name}'"),
                config_overrides,
            )
            if not isinstance(merged_preview, dict):
                raise ManifestError(f"Effective config for job '{name}' must be a mapping.")
            generated_config_path: Optional[Path] = (
                generated_root / f"{sanitize_name(name)}.yaml"
            ).resolve()
        else:
            generated_config_path = None

        labeled_id_path = ensure_existing_file(
            resolve_path(str(raw_job["labeled_id_path"]), base_dir=manifest_dir),
            f"Labeled split file for job '{name}'",
        )
        unlabeled_id_path = ensure_existing_file(
            resolve_path(str(raw_job["unlabeled_id_path"]), base_dir=manifest_dir),
            f"Unlabeled split file for job '{name}'",
        )
        port = global_config.port_base + len(jobs)
        save_path = (global_config.save_root / save_subdir).resolve()

        job = BatchJob(
            name=name,
            script=script_path,
            config=config_path,
            config_overrides=config_overrides,
            generated_config_path=generated_config_path,
            labeled_id_path=labeled_id_path,
            unlabeled_id_path=unlabeled_id_path,
            save_subdir=save_subdir,
            save_path=save_path,
            port=port,
            extra_args=require_str_list(raw_job.get("extra_args"), f"jobs[{index}].extra_args"),
            env=require_env_mapping(raw_job.get("env"), f"jobs[{index}].env"),
        )
        jobs.append(job)

    if only_names is not None:
        missing_names = sorted(only_names - seen_names)
        if missing_names and not allow_missing_only_names:
            raise ManifestError(
                f"Requested job(s) not found in manifest: {', '.join(missing_names)}"
            )
    if not jobs and not allow_no_jobs:
        raise ManifestError("No jobs remain after applying filters.")

    return BatchManifest(
        manifest_path=manifest_path,
        global_config=global_config,
        jobs=jobs,
    )


def build_job_command(manifest: BatchManifest, job: BatchJob) -> List[str]:
    return [
        "torchrun",
        f"--nproc_per_node={manifest.global_config.nproc_per_node}",
        f"--master_port={job.port}",
        str(job.script),
        "--config",
        str(effective_config_path(job)),
        "--labeled-id-path",
        str(job.labeled_id_path),
        "--unlabeled-id-path",
        str(job.unlabeled_id_path),
        "--save-path",
        str(job.save_path),
        "--port",
        str(job.port),
        *job.extra_args,
    ]


def done_marker_path(job: BatchJob) -> Path:
    return job.save_path / DONE_MARKER_NAME


def checkpoint_path(job: BatchJob) -> Path:
    return job.save_path / "latest.pth"


def summary_paths(manifest: BatchManifest) -> tuple[Path, Path]:
    summary_dir = manifest.global_config.save_root / SUMMARY_DIR_NAME
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    base_name = manifest.manifest_path.stem
    run_path = summary_dir / f"{timestamp}_{base_name}.json"
    latest_path = summary_dir / f"{base_name}_latest.json"
    return run_path, latest_path


def prepare_log_file(job: BatchJob) -> Path:
    job.save_path.mkdir(parents=True, exist_ok=True)
    return job.save_path / "out.log"


def merged_env(global_env: Dict[str, str], job_env: Dict[str, str]) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(global_env)
    env.update(job_env)
    return env


def run_subprocess(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Dict[str, str],
    log_path: Path,
) -> int:
    with log_path.open("a", encoding="utf-8") as handle:
        try:
            process = subprocess.Popen(
                list(command),
                cwd=str(cwd),
                env=env,
                stdout=handle,
                stderr=subprocess.STDOUT,
                text=True,
            )
        except OSError as exc:
            handle.write(f"\n[batch_runner] failed to launch command: {exc}\n")
            return 127
        return process.wait()


def write_done_marker(job: BatchJob, *, command: Sequence[str], attempts: int) -> None:
    payload = {
        "job": job.name,
        "save_path": str(job.save_path),
        "base_config": str(job.config),
        "effective_config": str(effective_config_path(job)),
        "config_overrides": job.config_overrides,
        "completed_at": utc_now_iso(),
        "attempts": attempts,
        "command": list(command),
    }
    done_marker_path(job).write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )


def print_job_banner(log_path: Path, *, job: BatchJob, command: Sequence[str], attempt: int, was_resume: bool) -> None:
    lines = [
        "",
        f"=== [{datetime.now().isoformat(timespec='seconds')}] Batch runner launch ===",
        f"job={job.name}",
        f"attempt={attempt}",
        f"resume={str(was_resume).lower()}",
        f"base_config={job.config}",
        f"effective_config={effective_config_path(job)}",
        f"save_path={job.save_path}",
        f"port={job.port}",
        f"command={shell_join(command)}",
        "",
    ]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def execute_job(manifest: BatchManifest, job: BatchJob) -> Dict[str, Any]:
    materialize_job_config(job)
    command = build_job_command(manifest, job)
    log_path = prepare_log_file(job)
    done_path = done_marker_path(job)
    latest_path = checkpoint_path(job)
    was_resume = latest_path.exists() and not done_path.exists()

    if done_path.exists():
        return {
            "name": job.name,
            "status": "skipped",
            "attempts": 0,
            "exit_code": 0,
            "was_resume": latest_path.exists(),
            "command": command,
            "port": job.port,
            "base_config": str(job.config),
            "effective_config": str(effective_config_path(job)),
            "config_overrides": deepcopy(job.config_overrides),
            "save_path": str(job.save_path),
            "out_log": str(log_path),
            "started_at": None,
            "ended_at": utc_now_iso(),
        }

    env = merged_env(manifest.global_config.env, job.env)
    started_at = utc_now_iso()
    attempts = 0
    exit_code = 1

    for attempt in range(1, manifest.global_config.max_retries + 2):
        attempts = attempt
        print_job_banner(log_path, job=job, command=command, attempt=attempt, was_resume=was_resume)
        exit_code = run_subprocess(command, cwd=REPO_ROOT, env=env, log_path=log_path)
        if exit_code == 0:
            write_done_marker(job, command=command, attempts=attempts)
            break

    ended_at = utc_now_iso()
    status = "resumed" if exit_code == 0 and was_resume else "succeeded" if exit_code == 0 else "failed"
    return {
        "name": job.name,
        "status": status,
        "attempts": attempts,
        "exit_code": exit_code,
        "was_resume": was_resume,
        "command": command,
        "port": job.port,
        "base_config": str(job.config),
        "effective_config": str(effective_config_path(job)),
        "config_overrides": deepcopy(job.config_overrides),
        "save_path": str(job.save_path),
        "out_log": str(log_path),
        "started_at": started_at,
        "ended_at": ended_at,
    }


def write_summary_files(manifest: BatchManifest, payload: Dict[str, Any]) -> tuple[Path, Path]:
    run_path, latest_path = summary_paths(manifest)
    run_path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    run_path.write_text(serialized, encoding="utf-8")
    latest_path.write_text(serialized, encoding="utf-8")
    return run_path, latest_path


def print_dry_run(manifest: BatchManifest) -> None:
    print(f"Manifest: {manifest.manifest_path}")
    print(f"Save root: {manifest.global_config.save_root}")
    for index, job in enumerate(manifest.jobs, start=1):
        command = build_job_command(manifest, job)
        print(f"[{index}/{len(manifest.jobs)}] {job.name}")
        print(f"  base_config: {job.config}")
        print(f"  effective_config: {effective_config_path(job)}")
        if job.config_overrides:
            rendered = json.dumps(job.config_overrides, ensure_ascii=False, sort_keys=True)
            print(f"  config_overrides: {rendered}")
        print(f"  save_path: {job.save_path}")
        print(f"  port: {job.port}")
        print(f"  command: {shell_join(command)}")


def print_run_summary(summary: Dict[str, Any]) -> None:
    print("\nBatch summary")
    print("-------------")
    for result in summary["results"]:
        print(
            f"{result['status'].upper():9s} {result['name']} | port={result['port']} | "
            f"save_path={result['save_path']} | log={result['out_log']}"
        )
    print(f"Summary JSON: {summary['summary_path']}")


def build_summary_payload(
    manifest: BatchManifest,
    *,
    results: List[Dict[str, Any]],
    dry_run: bool,
    queue_mode: bool = False,
    poll_seconds: Optional[float] = None,
    max_idle_polls: Optional[int] = None,
    idle_polls: Optional[int] = None,
) -> Dict[str, Any]:
    payload = {
        "manifest": str(manifest.manifest_path),
        "save_root": str(manifest.global_config.save_root),
        "generated_at": utc_now_iso(),
        "dry_run": dry_run,
        "continue_on_error": manifest.global_config.continue_on_error,
        "max_retries": manifest.global_config.max_retries,
        "results": results,
    }
    if queue_mode:
        payload.update(
            {
                "queue_mode": True,
                "poll_seconds": poll_seconds,
                "max_idle_polls": max_idle_polls,
                "idle_polls": idle_polls,
            }
        )
    return payload


def persist_summary(manifest: BatchManifest, payload: Dict[str, Any]) -> Dict[str, Any]:
    run_path, latest_path = write_summary_files(manifest, payload)
    payload["summary_path"] = str(run_path)
    payload["latest_summary_path"] = str(latest_path)
    return payload


def ensure_watch_compatible(reference: BatchManifest, candidate: BatchManifest) -> None:
    ref = reference.global_config
    cur = candidate.global_config
    incompatible_fields: List[str] = []
    if cur.save_root != ref.save_root:
        incompatible_fields.append("global.save_root")
    if cur.port_base != ref.port_base:
        incompatible_fields.append("global.port_base")
    if cur.nproc_per_node != ref.nproc_per_node:
        incompatible_fields.append("global.nproc_per_node")
    if incompatible_fields:
        raise ManifestError(
            "Watch mode requires stable queue identity; incompatible changes detected in "
            + ", ".join(incompatible_fields)
        )


def reload_manifest_for_watch(
    manifest_path: Path,
    *,
    save_root_override: Optional[Path],
    only_names: Optional[Set[str]],
) -> BatchManifest:
    return load_manifest(
        manifest_path,
        save_root_override=save_root_override,
        only_names=only_names,
        allow_no_jobs=True,
        allow_missing_only_names=True,
    )


def run_batch(manifest: BatchManifest, *, dry_run: bool = False) -> Dict[str, Any]:
    if dry_run:
        print_dry_run(manifest)
        return {
            "manifest": str(manifest.manifest_path),
            "save_root": str(manifest.global_config.save_root),
            "dry_run": True,
            "results": [],
        }

    results: List[Dict[str, Any]] = []
    for job in manifest.jobs:
        result = execute_job(manifest, job)
        results.append(result)
        if result["status"] == "failed" and not manifest.global_config.continue_on_error:
            break

    summary = build_summary_payload(manifest, results=results, dry_run=False)
    summary = persist_summary(manifest, summary)
    print_run_summary(summary)
    return summary


def run_batch_watch(
    *,
    manifest_path: Path,
    save_root_override: Optional[Path] = None,
    only_names: Optional[Set[str]] = None,
    poll_seconds: float = 30.0,
    max_idle_polls: Optional[int] = None,
) -> Dict[str, Any]:
    if poll_seconds < 0:
        raise ManifestError("--poll-seconds must be >= 0.")
    if max_idle_polls is not None and max_idle_polls < 0:
        raise ManifestError("--max-idle-polls must be >= 0.")

    manifest = reload_manifest_for_watch(
        manifest_path,
        save_root_override=save_root_override,
        only_names=only_names,
    )
    reference_manifest = manifest
    processed_names: Set[str] = set()
    results: List[Dict[str, Any]] = []
    idle_polls = 0

    while True:
        manifest = reload_manifest_for_watch(
            manifest_path,
            save_root_override=save_root_override,
            only_names=only_names,
        )
        ensure_watch_compatible(reference=reference_manifest, candidate=manifest)
        pending_jobs = [job for job in manifest.jobs if job.name not in processed_names]

        if pending_jobs:
            idle_polls = 0
            pending_manifest = BatchManifest(
                manifest_path=manifest.manifest_path,
                global_config=manifest.global_config,
                jobs=pending_jobs,
            )
            for job in pending_jobs:
                result = execute_job(pending_manifest, job)
                results.append(result)
                processed_names.add(job.name)
                summary = build_summary_payload(
                    manifest,
                    results=results,
                    dry_run=False,
                    queue_mode=True,
                    poll_seconds=poll_seconds,
                    max_idle_polls=max_idle_polls,
                    idle_polls=idle_polls,
                )
                summary = persist_summary(manifest, summary)
                if result["status"] == "failed" and not manifest.global_config.continue_on_error:
                    print_run_summary(summary)
                    return summary
        else:
            idle_polls += 1
            summary = build_summary_payload(
                manifest,
                results=results,
                dry_run=False,
                queue_mode=True,
                poll_seconds=poll_seconds,
                max_idle_polls=max_idle_polls,
                idle_polls=idle_polls,
            )
            summary = persist_summary(manifest, summary)
            if max_idle_polls is not None and idle_polls >= max_idle_polls:
                print_run_summary(summary)
                return summary
            time.sleep(poll_seconds)




def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    try:
        only_names = parse_only_names(args.only)
        save_root_override = Path(args.save_root) if args.save_root else None
        if args.watch:
            if args.dry_run:
                raise ManifestError("--dry-run cannot be combined with --watch.")
            summary = run_batch_watch(
                manifest_path=Path(args.manifest),
                save_root_override=save_root_override,
                only_names=only_names,
                poll_seconds=args.poll_seconds,
                max_idle_polls=args.max_idle_polls,
            )
        else:
            manifest = load_manifest(
                Path(args.manifest),
                save_root_override=save_root_override,
                only_names=only_names,
            )
            summary = run_batch(manifest, dry_run=args.dry_run)
    except ManifestError as exc:
        print(f"Manifest error: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("Interrupted by user.", file=sys.stderr)
        return 130

    if args.dry_run:
        return 0
    failures = [result for result in summary["results"] if result["status"] == "failed"]
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
