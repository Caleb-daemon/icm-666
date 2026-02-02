#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run Tasks 1-3 in a single, consistent pipeline."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def run_cmd(cmd: list[str], cwd: str) -> None:
    """Run a command and fail fast on error."""
    print("\n>>>", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="ICM E pipeline: Task1 -> Task4")

    # Task selection
    parser.add_argument("--task", choices=["task1", "task2", "task3", "task4"], default=None,
                        help="Run a single task (legacy default runs Task1-3).")
    parser.add_argument("--skip_task1", action="store_true", help="Skip Task 1 (Sungrove retrofit).")
    parser.add_argument("--skip_task2", action="store_true", help="Skip Task 2 (Borealis optimization).")
    parser.add_argument("--skip_task3", action="store_true", help="Skip Task 3 (generalization).")
    parser.add_argument("--skip_task4", action="store_true", help="Skip Task 4 (Student Union).")
    parser.add_argument("--quick", action="store_true",
                        help="Quick smoke run (grid search for Task1, small NSGA-II for Task2).")

    # Task 1 options
    parser.add_argument("--task1_epw", default=os.path.join("data", "epw", "HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw"),
                        help="EPW file for Task 1.")
    parser.add_argument("--task1_synthetic", action="store_true", help="Use synthetic weather for Task 1.")
    parser.add_argument("--task1_no_plots", action="store_true", help="Disable Task 1 plotting.")

    # Task 2 options
    parser.add_argument("--task2_mode", choices=["nsga2", "single"], default="nsga2",
                        help="Task 2 mode: nsga2 (Pareto) or single (weighted objective).")
    parser.add_argument("--task2_epw", default=os.path.join("data", "epw", "NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw"),
                        help="EPW file for Task 2 (applies to both nsga2 and single).")
    parser.add_argument("--task2_synthetic", action="store_true", help="Use synthetic weather for Task 2.")

    # Task 3 options
    parser.add_argument("--task3_epw_train", nargs="+", default=[
        os.path.join("data", "epw", "HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw"),
        os.path.join("data", "epw", "NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw"),
    ], help="Training EPW files for Task 3.")
    parser.add_argument("--task3_epw_ext", default=os.path.join("data", "epw", "SAU_RI_Riyadh.AB.404380_TMYx.2009-2023.epw"),
                        help="External validation EPW for Task 3 (Riyadh).")
    parser.add_argument("--results_dir", default="results", help="Shared results directory.")
    parser.add_argument("--task3_outdir", default=os.path.join("results", "task3"),
                        help="Task 3 output directory.")

    # Task 4 options
    parser.add_argument("--task4_epw", default=os.path.join("data", "epw", "HKG_NT_Lau.Fau.Shan.450350_TMYx.2009-2023.epw"),
                        help="EPW file for Task 4.")
    parser.add_argument("--task4_outdir", default=os.path.join("results", "task4"),
                        help="Task 4 output directory.")
    parser.add_argument("--task4_pop", type=int, default=None, help="Task 4 NSGA-II population size.")
    parser.add_argument("--task4_gen", type=int, default=None, help="Task 4 NSGA-II generations.")
    parser.add_argument("--task4_no_plots", action="store_true", help="Disable Task 4 plotting.")

    args = parser.parse_args()

    repo_root = os.path.dirname(os.path.abspath(__file__))

    def should_run(task_name: str, skip_flag: bool) -> bool:
        if args.task is not None:
            return args.task == task_name
        if task_name == "task4":
            return False
        return not skip_flag

    # Task 1: Sungrove retrofit
    if should_run("task1", args.skip_task1):
        cmd = [
            sys.executable,
            os.path.join(repo_root, "src_task1_sungrove", "task1_sungrove.py"),
            "--epw", args.task1_epw,
        ]
        if args.task1_synthetic:
            cmd.append("--synthetic")
        if args.task1_no_plots:
            cmd.append("--no_plots")
        if args.quick:
            cmd += ["--optimizer", "grid", "--grid_n", "3"]
        run_cmd(cmd, cwd=repo_root)

        if not args.task1_no_plots:
            plot_cmd = [
                sys.executable,
                os.path.join(repo_root, "src_task1_sungrove", "plot_task1.py"),
            ]
            run_cmd(plot_cmd, cwd=repo_root)

    # Task 2: Borealis optimization
    if should_run("task2", args.skip_task2):
        if args.task2_mode == "nsga2":
            cmd = [
                sys.executable,
                os.path.join(repo_root, "src_task2_nsga2", "run_nsga2.py"),
            ]
            cmd += ["--epw", args.task2_epw]
            if args.task2_synthetic:
                cmd.append("--synthetic")
            if args.quick:
                cmd += ["--pop", "20", "--gen", "10"]
        else:
            cmd = [
                sys.executable,
                os.path.join(repo_root, "src_task2_nsga2", "task2_borealis.py"),
                "--epw", args.task2_epw,
            ]
            if args.task2_synthetic:
                cmd.append("--synthetic")
        run_cmd(cmd, cwd=repo_root)

        plot_cmd = [
            sys.executable,
            os.path.join(repo_root, "src_task2_nsga2", "plot_task2.py"),
        ]
        run_cmd(plot_cmd, cwd=repo_root)

    # Task 3: Generalization + Riyadh external validation
    if should_run("task3", args.skip_task3):
        cmd = [
            sys.executable,
            "-m",
            "src_task3_generalization.task3_generalization",
            "--epw_train",
            *args.task3_epw_train,
            "--epw_ext",
            args.task3_epw_ext,
            "--results_dir",
            args.results_dir,
            "--outdir",
            args.task3_outdir,
        ]
        run_cmd(cmd, cwd=repo_root)

    # Task 4: Student Union robust optimization
    if should_run("task4", args.skip_task4):
        cmd = [
            sys.executable,
            "-m",
            "src_task4_student_union.task4_student_union",
            "--epw",
            args.task4_epw,
            "--results_dir",
            args.task4_outdir,
        ]
        if args.quick:
            cmd += ["--pop", "20", "--gen", "10"]
        if args.task4_pop:
            cmd += ["--pop", str(args.task4_pop)]
        if args.task4_gen:
            cmd += ["--gen", str(args.task4_gen)]
        if args.task4_no_plots:
            cmd.append("--no_plots")
        run_cmd(cmd, cwd=repo_root)

    print("\nPipeline complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
