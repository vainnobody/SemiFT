import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import yaml

from util.batch_runner import (
    DONE_MARKER_NAME,
    ManifestError,
    build_job_command,
    effective_config_path,
    gpu_is_idle,
    load_manifest,
    parse_target_gpu_tokens,
    run_batch,
    run_batch_watch,
)


class BatchTrainRunnerTest(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)
        repo_root = Path(__file__).resolve().parents[1]
        config = repo_root / "configs" / "pascal.yaml"
        labeled = repo_root / "splits" / "pascal" / "92" / "labeled.txt"
        unlabeled = repo_root / "splits" / "pascal" / "92" / "unlabeled.txt"
        script = repo_root / "supervised.py"

        manifest = {
            "global": {
                "nproc_per_node": 1,
                "save_root": str(self.tmp_path / "runs"),
                "port_base": 29600,
                "continue_on_error": True,
                "max_retries": 1,
            },
            "jobs": [
                {
                    "name": "job_a",
                    "script": str(script),
                    "config": str(config),
                    "config_overrides": {
                        "backbone": "dinov3_small",
                        "criterion": {"kwargs": {"ignore_index": 7}},
                    },
                    "labeled_id_path": str(labeled),
                    "unlabeled_id_path": str(unlabeled),
                    "save_subdir": "job_a",
                    "extra_args": ["--dummy-flag", "value"],
                },
                {
                    "name": "job_b",
                    "script": str(script),
                    "config": str(config),
                    "labeled_id_path": str(labeled),
                    "unlabeled_id_path": str(unlabeled),
                    "save_subdir": "job_b",
                },
            ],
        }

        self.manifest_path = self.tmp_path / "manifest.yaml"
        self.manifest_path.write_text(yaml.safe_dump(manifest), encoding="utf-8")

    def tearDown(self):
        self._tmpdir.cleanup()

    def test_load_manifest_assigns_ports_and_builds_command(self):
        manifest = load_manifest(self.manifest_path)
        self.assertEqual([job.name for job in manifest.jobs], ["job_a", "job_b"])
        self.assertEqual([job.port for job in manifest.jobs], [29600, 29601])
        self.assertTrue(str(effective_config_path(manifest.jobs[0])).endswith("_batch/generated_configs/manifest/job_a.yaml"))
        self.assertEqual(effective_config_path(manifest.jobs[1]), manifest.jobs[1].config)

        command = build_job_command(manifest, manifest.jobs[0])
        self.assertEqual(
            command[:3],
            ["torchrun", "--nproc_per_node=1", "--master_port=29600"],
        )
        self.assertEqual(command[-2:], ["--dummy-flag", "value"])
        self.assertEqual(command[5], str(effective_config_path(manifest.jobs[0])))
        self.assertTrue(str(manifest.jobs[0].save_path).endswith("runs/job_a"))

    def test_load_manifest_rejects_missing_required_fields(self):
        bad_manifest = self.tmp_path / "bad.yaml"
        bad_manifest.write_text(yaml.safe_dump({"global": {}, "jobs": [{}]}), encoding="utf-8")

        with self.assertRaises(ManifestError):
            load_manifest(bad_manifest)

    def test_parse_target_gpu_tokens_prefers_cuda_visible_devices(self):
        env = {"CUDA_VISIBLE_DEVICES": "3, 5"}
        self.assertEqual(parse_target_gpu_tokens(env, 2), ["3", "5"])
        self.assertEqual(parse_target_gpu_tokens({}, 2), ["0", "1"])

    def test_gpu_idle_requires_zero_mem_zero_util_and_no_process(self):
        self.assertTrue(
            gpu_is_idle(
                {
                    "has_compute_process": False,
                    "memory_used_mb": 0,
                    "utilization_gpu": 0,
                }
            )
        )
        self.assertFalse(
            gpu_is_idle(
                {
                    "has_compute_process": True,
                    "memory_used_mb": 0,
                    "utilization_gpu": 0,
                }
            )
        )
        self.assertFalse(
            gpu_is_idle(
                {
                    "has_compute_process": False,
                    "memory_used_mb": 1,
                    "utilization_gpu": 0,
                }
            )
        )

    def test_run_batch_waits_until_specified_gpus_are_idle(self):
        manifest_payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        manifest_payload["global"]["env"] = {"CUDA_VISIBLE_DEVICES": "4"}
        manifest_payload["jobs"] = [manifest_payload["jobs"][0]]
        self.manifest_path.write_text(yaml.safe_dump(manifest_payload), encoding="utf-8")
        manifest = load_manifest(self.manifest_path)

        gpu_snapshots = [
            [
                {
                    "index": 4,
                    "uuid": "GPU-4",
                    "memory_used_mb": 128,
                    "utilization_gpu": 75,
                    "has_compute_process": True,
                }
            ],
            [
                {
                    "index": 4,
                    "uuid": "GPU-4",
                    "memory_used_mb": 0,
                    "utilization_gpu": 0,
                    "has_compute_process": False,
                }
            ],
        ]
        sleeps = []
        launches = []

        def fake_resolve(_env, _nproc):
            return gpu_snapshots.pop(0)

        def fake_sleep(seconds):
            sleeps.append(seconds)

        def fake_run(command, *, cwd, env, log_path):
            launches.append((command, env.get("CUDA_VISIBLE_DEVICES"), Path(log_path)))
            return 0

        with mock.patch("util.batch_runner.resolve_target_gpus", side_effect=fake_resolve):
            with mock.patch("util.batch_runner.time.sleep", side_effect=fake_sleep):
                with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
                    summary = run_batch(
                        manifest,
                        wait_for_gpu=True,
                        gpu_poll_seconds=7,
                    )

        self.assertEqual(sleeps, [7])
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][1], "4")
        self.assertEqual(summary["results"][0]["status"], "succeeded")
        log_text = launches[0][2].read_text(encoding="utf-8")
        self.assertIn("GPU mode for job_a: fixed (4)", log_text)
        self.assertIn("waiting for GPUs before job_a", log_text)
        self.assertIn("GPUs ready for job_a", log_text)

    def test_run_batch_waits_for_dynamic_idle_gpu_pool_and_sets_cuda_visible_devices(self):
        manifest_payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        manifest_payload["global"]["nproc_per_node"] = 6
        manifest_payload["jobs"] = [manifest_payload["jobs"][0]]
        self.manifest_path.write_text(yaml.safe_dump(manifest_payload), encoding="utf-8")
        manifest = load_manifest(self.manifest_path)

        gpu_snapshots = [
            [
                {"index": 7, "uuid": "GPU-7", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 2, "uuid": "GPU-2", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 6, "uuid": "GPU-6", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 5, "uuid": "GPU-5", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 1, "uuid": "GPU-1", "memory_used_mb": 32, "utilization_gpu": 55, "has_compute_process": True},
                {"index": 3, "uuid": "GPU-3", "memory_used_mb": 64, "utilization_gpu": 70, "has_compute_process": True},
                {"index": 4, "uuid": "GPU-4", "memory_used_mb": 8, "utilization_gpu": 10, "has_compute_process": True},
                {"index": 0, "uuid": "GPU-0", "memory_used_mb": 0, "utilization_gpu": 90, "has_compute_process": True},
            ],
            [
                {"index": 7, "uuid": "GPU-7", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 2, "uuid": "GPU-2", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 6, "uuid": "GPU-6", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 5, "uuid": "GPU-5", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 1, "uuid": "GPU-1", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 3, "uuid": "GPU-3", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 4, "uuid": "GPU-4", "memory_used_mb": 16, "utilization_gpu": 5, "has_compute_process": True},
                {"index": 0, "uuid": "GPU-0", "memory_used_mb": 12, "utilization_gpu": 8, "has_compute_process": True},
            ],
        ]
        sleeps = []
        launches = []

        def fake_query():
            return gpu_snapshots.pop(0)

        def fake_sleep(seconds):
            sleeps.append(seconds)

        def fake_run(command, *, cwd, env, log_path):
            launches.append((command, env.get("CUDA_VISIBLE_DEVICES"), Path(log_path)))
            return 0

        with mock.patch("util.batch_runner.query_gpu_inventory", side_effect=fake_query):
            with mock.patch("util.batch_runner.time.sleep", side_effect=fake_sleep):
                with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
                    summary = run_batch(
                        manifest,
                        wait_for_gpu=True,
                        gpu_poll_seconds=5,
                    )

        self.assertEqual(sleeps, [5])
        self.assertEqual(len(launches), 1)
        self.assertEqual(launches[0][1], "1,2,3,5,6,7")
        self.assertEqual(summary["results"][0]["status"], "succeeded")
        log_text = launches[0][2].read_text(encoding="utf-8")
        self.assertIn("GPU mode for job_a: dynamic (need 6 idle GPUs)", log_text)
        self.assertIn("waiting for 6 idle GPUs before job_a: idle=4 (2, 5, 6, 7); sleep 5s", log_text)
        self.assertIn("selected GPUs for job_a: 1, 2, 3, 5, 6, 7", log_text)

    def test_run_batch_materializes_overridden_config_and_marks_done(self):
        manifest = load_manifest(self.manifest_path, only_names={"job_a"})
        calls = []

        def fake_run(command, *, cwd, env, log_path):
            calls.append((command, cwd, env, log_path))
            return 0

        with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
            first_summary = run_batch(manifest)
            result = first_summary["results"][0]
            self.assertEqual(result["status"], "succeeded")
            generated_config = Path(result["effective_config"])
            self.assertTrue(generated_config.exists())
            generated_payload = yaml.safe_load(generated_config.read_text(encoding="utf-8"))
            self.assertEqual(generated_payload["backbone"], "dinov3_small")
            self.assertEqual(generated_payload["criterion"]["kwargs"]["ignore_index"], 7)
            self.assertEqual(generated_payload["dataset"], "pascal")

            done_marker = manifest.jobs[0].save_path / DONE_MARKER_NAME
            self.assertTrue(done_marker.exists())
            done_payload = json.loads(done_marker.read_text(encoding="utf-8"))
            self.assertEqual(done_payload["job"], "job_a")
            self.assertEqual(done_payload["effective_config"], str(generated_config))
            self.assertEqual(done_payload["config_overrides"]["backbone"], "dinov3_small")
            self.assertEqual(len(calls), 1)

            second_summary = run_batch(manifest)
            self.assertEqual(second_summary["results"][0]["status"], "skipped")
            self.assertEqual(len(calls), 1)

    def test_run_batch_retries_resumed_jobs_and_continues_after_failures(self):
        manifest = load_manifest(self.manifest_path)
        checkpoint = manifest.jobs[0].save_path / "latest.pth"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.write_text("resume", encoding="utf-8")

        exit_codes = {
            "job_a": [1, 1],
            "job_b": [0],
        }
        call_order = []

        def fake_run(command, *, cwd, env, log_path):
            job_name = Path(log_path).parent.name
            call_order.append(job_name)
            return exit_codes[job_name].pop(0)

        with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
            summary = run_batch(manifest)

        statuses = {result["name"]: result for result in summary["results"]}
        self.assertEqual(statuses["job_a"]["status"], "failed")
        self.assertTrue(statuses["job_a"]["was_resume"])
        self.assertEqual(statuses["job_a"]["attempts"], 2)
        self.assertEqual(statuses["job_a"]["config_overrides"]["backbone"], "dinov3_small")
        self.assertEqual(statuses["job_b"]["status"], "succeeded")
        self.assertEqual(call_order, ["job_a", "job_a", "job_b"])

    def test_run_batch_watch_picks_up_jobs_appended_later(self):
        original_payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        manifest_payload = json.loads(json.dumps(original_payload))
        manifest_payload["jobs"] = [manifest_payload["jobs"][0]]
        self.manifest_path.write_text(
            yaml.safe_dump(manifest_payload), encoding="utf-8"
        )

        calls = []

        def fake_run(command, *, cwd, env, log_path):
            calls.append(Path(log_path).parent.name)
            return 0

        sleep_calls = {"count": 0}

        def fake_sleep(_seconds):
            sleep_calls["count"] += 1
            if sleep_calls["count"] == 1:
                updated = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
                updated["jobs"].append(original_payload["jobs"][1])
                self.manifest_path.write_text(
                    yaml.safe_dump(updated), encoding="utf-8"
                )

        with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
            with mock.patch("util.batch_runner.time.sleep", side_effect=fake_sleep):
                summary = run_batch_watch(
                    manifest_path=self.manifest_path,
                    poll_seconds=0,
                    max_idle_polls=2,
                )

        self.assertEqual(calls, ["job_a", "job_b"])
        self.assertEqual(
            [(result["name"], result["status"]) for result in summary["results"]],
            [("job_a", "succeeded"), ("job_b", "succeeded")],
        )
        self.assertTrue(summary["queue_mode"])
        self.assertEqual(summary["max_idle_polls"], 2)

    def test_run_batch_watch_allows_empty_queue_until_job_arrives(self):
        original_payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        manifest_payload = json.loads(json.dumps(original_payload))
        manifest_payload["jobs"] = []
        self.manifest_path.write_text(
            yaml.safe_dump(manifest_payload), encoding="utf-8"
        )

        calls = []

        def fake_run(command, *, cwd, env, log_path):
            calls.append(Path(log_path).parent.name)
            return 0

        sleep_calls = {"count": 0}

        def fake_sleep(_seconds):
            sleep_calls["count"] += 1
            if sleep_calls["count"] == 1:
                manifest_payload["jobs"] = [original_payload["jobs"][0]]
                self.manifest_path.write_text(
                    yaml.safe_dump(manifest_payload), encoding="utf-8"
                )

        with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
            with mock.patch("util.batch_runner.time.sleep", side_effect=fake_sleep):
                summary = run_batch_watch(
                    manifest_path=self.manifest_path,
                    poll_seconds=0,
                    max_idle_polls=2,
                )

        self.assertEqual(calls, ["job_a"])
        self.assertEqual(len(summary["results"]), 1)
        self.assertEqual(summary["results"][0]["name"], "job_a")

    def test_run_batch_watch_uses_dynamic_gpu_selection_for_new_jobs(self):
        manifest_payload = yaml.safe_load(self.manifest_path.read_text(encoding="utf-8"))
        manifest_payload["global"]["nproc_per_node"] = 2
        manifest_payload["jobs"] = [manifest_payload["jobs"][0]]
        self.manifest_path.write_text(
            yaml.safe_dump(manifest_payload), encoding="utf-8"
        )

        launches = []

        def fake_query():
            return [
                {"index": 5, "uuid": "GPU-5", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 3, "uuid": "GPU-3", "memory_used_mb": 0, "utilization_gpu": 0, "has_compute_process": False},
                {"index": 0, "uuid": "GPU-0", "memory_used_mb": 32, "utilization_gpu": 60, "has_compute_process": True},
            ]

        def fake_run(command, *, cwd, env, log_path):
            launches.append((Path(log_path).parent.name, env.get("CUDA_VISIBLE_DEVICES")))
            return 0

        with mock.patch("util.batch_runner.query_gpu_inventory", side_effect=fake_query):
            with mock.patch("util.batch_runner.run_subprocess", side_effect=fake_run):
                summary = run_batch_watch(
                    manifest_path=self.manifest_path,
                    poll_seconds=0,
                    max_idle_polls=1,
                    wait_for_gpu=True,
                    gpu_poll_seconds=0,
                )

        self.assertEqual(launches, [("job_a", "3,5")])
        self.assertEqual(summary["results"][0]["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
