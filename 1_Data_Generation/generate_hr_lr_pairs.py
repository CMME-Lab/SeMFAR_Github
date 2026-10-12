"""Run the existing pipeline with LR acoustic maps downsampled from HR.

This wrapper preserves the original runner and simulator files and stores all
outputs in dedicated ``*_hr_derived_lr`` directories.
"""

import os
import subprocess
import sys

# Background batch runs must never initialize Tk.  Tk object cleanup from a
# worker thread can abort the interpreter with Tcl_AsyncDelete on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

# The original pipeline prints Unicode status symbols.  Windows redirected
# streams otherwise default to cp949 and can fail before simulation starts.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import pipeline_runner as original_pipeline
from core.simulate_hr_derived_lr import (
    HRDerivedLRUltrasoundSimulator,
)


class PairedPipelineConfig(original_pipeline.PipelineConfig):
    def __init__(self):
        super().__init__()
        project_root = os.path.dirname(os.path.abspath(__file__))
        work_root = os.path.abspath(
            os.environ.get("SEMFAR_WORK_ROOT", os.path.join(project_root, "local_outputs"))
        )
        self.simulation_results_path = os.path.join(
            work_root, "python_simulation_results_hr_derived_lr"
        )
        self.final_output_path = os.path.join(
            work_root, "final_training_data_hr_derived_lr"
        )
        # Pairing requires canonical HR to run before LR.
        self.scale_variants = [
            {"name": "sc1.0", "sc_w_x": 1.0, "sc_w_y": 1.0},
            {"name": "sc0.5", "sc_w_x": 0.5, "sc_w_y": 0.5},
        ]


class PairedPipelineRunner(original_pipeline.PipelineRunner):
    """Run shared HR/LR display processing after each complete raw pair."""

    def run_complete_pipeline(self):
        results = super().run_complete_pipeline()
        paired_results = self._run_shared_pair_postprocessing()
        if paired_results:
            results["shared_pair_postprocessing"] = paired_results
        return results

    def _run_shared_pair_postprocessing(self):
        project_root = os.path.dirname(os.path.abspath(__file__))
        simulation_root = self.config.simulation_results_path
        work_root = os.path.dirname(os.path.abspath(self.config.final_output_path))
        shared_root = os.path.join(
            work_root, "final_training_data_hr_derived_lr_shared_post_db40_gain"
        )
        final_root = os.path.join(
            work_root, "final_training_data_hr_derived_lr_shared_post_db40_gain_diff120"
        )
        shared_script = os.path.join(
            project_root, "postprocessing", "shared_pair_normalization.py"
        )
        scale_script = os.path.join(
            project_root, "postprocessing", "fixed_pair_brightness.py"
        )
        plot_script = os.path.join(
            project_root, "visualization", "plot_pair_difference.py"
        )

        completed = {}
        for sample_name in self.processed_sample_names:
            hr_simulation = os.path.join(
                simulation_root, "sc1.0", sample_name, "simulation_low.mat"
            )
            lr_simulation = os.path.join(
                simulation_root, "sc0.5", sample_name, "simulation_low.mat"
            )
            if not (os.path.exists(hr_simulation) and os.path.exists(lr_simulation)):
                print(
                    f"Skipping shared pair post-processing for incomplete pair: "
                    f"{sample_name}"
                )
                continue

            print("\n" + "=" * 72)
            print(f"SHARED HR/LR POST-PROCESSING: {sample_name}")
            print("40 dB shared transform, no scan-line interpolation, max difference 120")
            print("=" * 72)
            subprocess.run(
                [
                    sys.executable,
                    shared_script,
                    "--sample-name", sample_name,
                    "--simulation-root", simulation_root,
                    "--output-root", shared_root,
                ],
                check=True,
                cwd=project_root,
            )
            subprocess.run(
                [
                    sys.executable,
                    scale_script,
                    "--sample-name", sample_name,
                    "--source-root", shared_root,
                    "--output-root", final_root,
                    "--target-max-difference", "120",
                ],
                check=True,
                cwd=project_root,
            )

            hr_png = os.path.join(
                final_root, "sc1.0", "scans", f"scan_{sample_name}_HR.png"
            )
            lr_png = os.path.join(
                final_root, "sc0.5", "scans", f"scan_{sample_name}_LR.png"
            )
            difference_plot = os.path.join(
                final_root, f"scan_{sample_name}_HR_minus_LR_pixelwise.png"
            )
            subprocess.run(
                [
                    sys.executable,
                    plot_script,
                    hr_png,
                    lr_png,
                    difference_plot,
                    "--first-label", "diff120-HR",
                    "--second-label", "diff120-LR",
                ],
                check=True,
                cwd=project_root,
            )
            completed[sample_name] = {
                "hr_png": hr_png,
                "lr_png": lr_png,
                "difference_plot": difference_plot,
            }
        return completed


def main():
    original_pipeline.PipelineConfig = PairedPipelineConfig
    original_pipeline.PipelineRunner = PairedPipelineRunner
    original_pipeline.UltrasoundSimulator = HRDerivedLRUltrasoundSimulator
    return original_pipeline.main()


if __name__ == "__main__":
    main()
