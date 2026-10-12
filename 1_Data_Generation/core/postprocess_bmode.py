"""
Python Post-Processing Module
=============================

Post-processing module for Python k-Wave simulation results.
Converts simulation outputs to training-ready format for ML models.

This module follows the same workflow as post_processing_matlab_simulation_data.py
but is specifically designed for the Python simulation pipeline outputs.
"""

import os
import json
import pickle
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import cv2
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass

# Import post-processing functions from kwave
from kwave.reconstruction.beamform import envelope_detection
from kwave.reconstruction.tools import log_compression
from kwave.utils.conversion import db2neper
from kwave.utils.filters import gaussian_filter
from kwave.utils.signals import get_win

# Import centralized configuration
from config.pipeline_options import PostProcessingConfig


def _safe_normalized_log_compression(signal: np.ndarray, compression_factor: float) -> np.ndarray:
    """Match k-Wave normalized log compression without producing 0 / 0 NaNs.

    A completely silent scan line has a maximum envelope of zero.  k-Wave's
    normalized implementation divides that line by its maximum, so one silent
    line becomes NaN and later image-wide normalization can turn the entire
    B-mode image into NaN.  Preserve the usual result for non-silent lines and
    explicitly keep silent lines at zero.
    """
    signal = np.asarray(signal)
    maxima = np.max(signal, axis=-1, keepdims=True)
    valid = np.isfinite(maxima) & (maxima > 0)
    safe_maxima = np.where(valid, maxima, 1.0)
    compressed = safe_maxima * (
        np.log10(1.0 + compression_factor * signal / safe_maxima)
        / np.log10(1.0 + compression_factor)
    )
    compressed = np.where(valid, compressed, 0.0)

    return compressed

class PythonSimulationPostProcessor:
    """Post-processor for Python k-Wave simulation results"""
    
    def __init__(self, config: PostProcessingConfig):
        self.config = config
    
    def process_sample(self, sample_name: str, simulation_results_path: str, 
                      output_path: str, noise_level: str = 'low') -> Dict:
        """
        Process a single sample's simulation results
        
        Args:
            sample_name: Sample name (e.g., 'benign_1', 'malignant_5')
            simulation_results_path: Path to simulation results folder
            output_path: Path to save processed results
            noise_level: Noise level to process
            
        Returns:
            Dictionary with processed data paths
        """
        # Setup paths
        sample_folder = os.path.join(simulation_results_path, sample_name)
        simulation_file = os.path.join(sample_folder, f"simulation_{noise_level}.mat")
        phantom_file = os.path.join(sample_folder, f"phantom_{noise_level}.mat")
        
        if not os.path.exists(simulation_file):
            raise FileNotFoundError(f"Simulation file not found: {simulation_file}")
        if not os.path.exists(phantom_file):
            raise FileNotFoundError(f"Phantom file not found: {phantom_file}")
        
        print(f"Processing sample {sample_name} with noise level {noise_level}...")
        
        # Load simulation and phantom data
        sim_data = scipy.io.loadmat(simulation_file)
        phantom_data = scipy.io.loadmat(phantom_file)
        
        # Extract data
        scan_lines = sim_data['scan_lines']
        sound_speed_map = sim_data['sound_speed_map']
        density_map = sim_data['density_map']
        semantic_map = phantom_data['semantic_map']
        kgrid_dt = float(sim_data['kgrid_dt'])
        kgrid_Nt = int(sim_data['kgrid_Nt'])
        element_width = int(sim_data['element_width'])
        
        # Process the scan lines
        processed_results = self._process_scan_lines(
            scan_lines, sound_speed_map, semantic_map, 
            kgrid_dt, kgrid_Nt, element_width
        )
        
        
        simulation_metadata = self._extract_simulation_metadata(
            sim_data=sim_data,
            scan_lines=scan_lines,
            sound_speed_map=sound_speed_map,
            element_width=element_width,
        )

        processed_results["raw_simulation"] = {
            "scan_lines": scan_lines,
            "kgrid_dt": kgrid_dt,
            "kgrid_Nt": kgrid_Nt,
            "full_sound_speed_map": sound_speed_map,
            "full_density_map": density_map,
            "full_semantic_map": semantic_map,
        }

        processed_results["simulation_metadata"] = simulation_metadata

        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        scans_path = os.path.join(output_path, "scans")
        labels_path = os.path.join(output_path, "labels")
        os.makedirs(scans_path, exist_ok=True)
        os.makedirs(labels_path, exist_ok=True)
        
        # Save processed data
        result_paths = self._save_processed_data(
            processed_results, sample_name, scans_path, labels_path
        )
        
        # Optional visualization
        if self.config.show_preview:
            self._visualize_processed_data(processed_results, sample_name, noise_level)
        
        print(f"Successfully processed sample {sample_name}")
        return result_paths
    
    @staticmethod
    def _mat_scalar(sim_data: Dict, key: str, default=None, cast_type=None):
        if key not in sim_data:
            return default

        value = sim_data[key]

        if isinstance(value, np.ndarray):
            if value.size == 0:
                return default
            value = value.squeeze()
            if hasattr(value, "item"):
                try:
                    value = value.item()
                except Exception:
                    pass

        if cast_type is not None and value is not None:
            try:
                value = cast_type(value)
            except Exception:
                return default

        return value


    def _extract_simulation_metadata(
        self,
        sim_data: Dict,
        scan_lines: np.ndarray,
        sound_speed_map: np.ndarray,
        element_width: int,
    ) -> Dict:
        sc_w_x = self._mat_scalar(
            sim_data,
            "sc_w_x",
            getattr(self.config, "sc_w_x", None),
            float,
        )
        sc_w_y = self._mat_scalar(
            sim_data,
            "sc_w_y",
            getattr(self.config, "sc_w_y", None),
            float,
        )

        num_scan_lines = self._mat_scalar(
            sim_data,
            "num_scan_lines",
            int(scan_lines.shape[0]),
            int,
        )
        scan_element_width = self._mat_scalar(
            sim_data,
            "scan_element_width",
            int(element_width),
            int,
        )
        grid_x_size = self._mat_scalar(
            sim_data,
            "grid_x_size",
            getattr(self.config, "Nx", None),
            int,
        )
        grid_y_size = self._mat_scalar(
            sim_data,
            "grid_y_size",
            int(sound_speed_map.shape[1] - num_scan_lines * scan_element_width),
            int,
        )

        transducer_element_width = self._mat_scalar(
            sim_data,
            "transducer_element_width",
            None,
            int,
        )
        transducer_width = self._mat_scalar(
            sim_data,
            "transducer_width",
            None,
            int,
        )
        simulation_time_minutes = self._mat_scalar(
            sim_data,
            "simulation_time_minutes",
            None,
            float,
        )
        run_seed = self._mat_scalar(sim_data, "run_seed", None, int)
        sample_seed = self._mat_scalar(sim_data, "sample_seed", None, int)
        noise_level_seed = self._mat_scalar(
            sim_data, "noise_level_seed", None, int
        )
        fine_scatterer_correlation = self._mat_scalar(
            sim_data, "fine_scatterer_correlation", None, float
        )
        pml_x_size = self._mat_scalar(
            sim_data,
            "pml_x_size",
            getattr(self.config, "pml_x_size", None),
            int,
        )
        pml_y_size = self._mat_scalar(
            sim_data,
            "pml_y_size",
            getattr(self.config, "pml_y_size", None),
            int,
        )
        tone_burst_freq = self._mat_scalar(
            sim_data,
            "tone_burst_freq",
            getattr(self.config, "tone_burst_freq", None),
            float,
        )
        tone_burst_cycles = self._mat_scalar(
            sim_data,
            "tone_burst_cycles",
            getattr(self.config, "tone_burst_cycles", None),
            float,
        )
        if transducer_element_width is None and sc_w_y is not None:
            number_elements_before = 32
            element_width_before = 2
            number_elements = max(1, int(round(number_elements_before * float(sc_w_y))))
            transducer_element_width = int(
                (number_elements_before * element_width_before * self.config.dy_before)
                / (number_elements * self.config.dy)
            )

        if transducer_width is None and sc_w_y is not None and transducer_element_width is not None:
            number_elements_before = 32
            number_elements = max(1, int(round(number_elements_before * float(sc_w_y))))
            transducer_width = int(number_elements * transducer_element_width)

        return {
            "sc_w_x": sc_w_x,
            "sc_w_y": sc_w_y,
            "num_scan_lines": int(num_scan_lines),
            "scan_element_width": int(scan_element_width),
            "transducer_element_width": (
                int(transducer_element_width) if transducer_element_width is not None else None
            ),
            "grid_x_size": int(grid_x_size) if grid_x_size is not None else None,
            "grid_y_size": int(grid_y_size),
            "transducer_width": (
                int(transducer_width) if transducer_width is not None else None
            ),
            "pml_x_size": int(pml_x_size) if pml_x_size is not None else None,
            "pml_y_size": int(pml_y_size) if pml_y_size is not None else None,
            "tone_burst_freq": float(tone_burst_freq) if tone_burst_freq is not None else None,
            "tone_burst_cycles": float(tone_burst_cycles) if tone_burst_cycles is not None else None,
            "simulation_time_minutes": simulation_time_minutes,
            "run_seed": run_seed,
            "sample_seed": sample_seed,
            "noise_level_seed": noise_level_seed,
            "fine_scatterer_correlation": fine_scatterer_correlation,
        }
    def _process_scan_lines(self, scan_lines: np.ndarray, sound_speed_map: np.ndarray,
                           semantic_map: np.ndarray, kgrid_dt: float, kgrid_Nt: int,
                           element_width: int) -> Dict:
        """Process raw scan lines through the ultrasound imaging pipeline"""
        
        print(f"Processing scan lines with shape: {scan_lines.shape}")
        print(f"kgrid_Nt: {kgrid_Nt}, kgrid_dt: {kgrid_dt}")
        
        # Create time arrays
        t_array = np.arange(kgrid_Nt) * kgrid_dt
        
        # 1. Remove input signal (windowing) - Following demo script pattern
        tukey_win, _ = get_win(kgrid_Nt * 2, "Tukey", False, 0.05)
        ###transmit_len = tone_burst_freq = self.config.tone_burst_freq
        tone_burst_freq = self.config.tone_burst_freq ###1.5e6
        transmit_len = int(round(
            self.config.tone_burst_cycles
            / (tone_burst_freq * kgrid_dt)
        )) ###int(self.config.tone_burst_cycles * 20) ###4 * 20  # Estimate based on tone burst cycles * some factor
        
        # Smooth transmit suppression.
        # A hard 0 -> 1 transition leaves a bright horizontal stripe after
        # filtering and log compression. Use a short blank followed by a
        # raised-cosine ramp instead.
        blanking_len = max(
            1,
            int(np.ceil(1.25 * transmit_len))
        )
        ramp_len = max(
            4,
            int(np.ceil(0.15 * transmit_len))
        )

        # Ensure the gate always has exactly kgrid_Nt samples.
        blanking_len = min(blanking_len, kgrid_Nt - 2)
        ramp_len = min(ramp_len, kgrid_Nt - blanking_len - 1)

        scan_line_win = np.ones(
            (1, kgrid_Nt),
            dtype=np.float64
        )
        scan_line_win[:, :blanking_len] = 0.0

        if ramp_len > 0:
            ramp_phase = np.linspace(
                0.0,
                np.pi / 2.0,
                ramp_len,
                endpoint=True
            )
            scan_line_win[
                :,
                blanking_len:blanking_len + ramp_len
            ] = np.sin(ramp_phase) ** 2

        # Apply the smooth transmit gate.
        scan_lines_windowed = scan_lines * scan_line_win

        # 2. Time Gain Compensation (TGC)
        c0 = 1540  # Speed of sound
        t0 = (
            blanking_len + 0.5 * ramp_len
        ) * kgrid_dt / 2
        r = c0 * (
            np.arange(1, kgrid_Nt + 1) * kgrid_dt - t0
        ) / 2

        # Calculate TGC following demo pattern
        ###tone_burst_freq = self.config.tone_burst_freq ###1.5e6
        alpha_coeff = 0.75
        alpha_power = 1.5
        tgc_alpha_db_cm = alpha_coeff * (tone_burst_freq * 1e-6) ** alpha_power
        tgc_alpha_np_m = db2neper(tgc_alpha_db_cm) * 100
        tgc = np.exp(tgc_alpha_np_m * 2 * r)
        
        # Apply TGC
        scan_lines_tgc = scan_lines_windowed * tgc
        
        # 3. Frequency Filtering
        fs = 1 / kgrid_dt
        
        # Fundamental frequency filtering
        scan_lines_fund = gaussian_filter(
            scan_lines_tgc, fs, tone_burst_freq, self.config.fund_filter_bw
        )
        scan_lines_harm = gaussian_filter(
            scan_lines_tgc, fs, tone_burst_freq * 2, self.config.harm_filter_bw
        )
        
        # 4. Envelope Detection
        scan_lines_fund_env = envelope_detection(scan_lines_fund)
        scan_lines_harm_env = envelope_detection(scan_lines_harm)
        ###
        # 여기부터 추가
        debug_dir = os.path.join("debug_postprocessing")
        os.makedirs(debug_dir, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.imshow(
            scan_lines_fund_env.T,
            cmap="gray",
            aspect="auto",
            origin="upper",
        )
        plt.colorbar()
        plt.title("Fundamental Envelope")
        plt.tight_layout()
        plt.savefig(
            os.path.join(debug_dir, "fundamental_envelope.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        # 5. Log Compression
        scan_lines_fund_log = _safe_normalized_log_compression(
            scan_lines_fund_env,
            self.config.compression_ratio,
        )
        scan_lines_harm_log = _safe_normalized_log_compression(
            scan_lines_harm_env,
            self.config.compression_ratio,
        )

        # Suppress residual transmit/PML bands at the top and bottom.
        # Axis 1 is depth because scan_lines has shape [scan_line, time].
        depth_samples = scan_lines_fund_log.shape[1]
        top_n = 1
        bottom_n = max(2, int(round(0.030 * depth_samples)))

        depth_taper = np.ones(depth_samples, dtype=np.float32)
        if top_n > 1:
            top_phase = np.linspace(0.0, np.pi / 2.0, top_n)
            depth_taper[:top_n] = np.sin(top_phase) ** 2
        if bottom_n > 1:
            bottom_phase = np.linspace(np.pi / 2.0, 0.0, bottom_n)
            depth_taper[-bottom_n:] = np.sin(bottom_phase) ** 2

        scan_lines_fund_log *= depth_taper[np.newaxis, :]
        scan_lines_harm_log *= depth_taper[np.newaxis, :]
        

        ###
        plt.figure(figsize=(10, 6))
        plt.imshow(
            scan_lines_fund_log.T,
            cmap="gray",
            aspect="auto",
            origin="upper",
        )
        plt.colorbar()
        plt.title("Fundamental Log-compressed")
        plt.tight_layout()
        plt.savefig(
            os.path.join(debug_dir, "fundamental_log.png"),
            dpi=150,
            bbox_inches="tight",
        )
        plt.close()
        # 
        # 6. Create phantom labels using the SAME spatial processing as ultrasound images
        processed_phantom = self._process_phantom_labels_aligned(
            sound_speed_map, semantic_map, element_width, scan_lines_fund_log.shape
        )
        return {
            "b_mode": scan_lines_fund_log,
            "harmonic": scan_lines_harm_log,
            "phantom_labels": processed_phantom,
        }
    
    def _process_phantom_labels_aligned(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray,
                                      element_width: int, ultrasound_shape: Tuple[int, int]) -> Dict:
        """
        Process phantom data to create labels using the SAME spatial transformations as ultrasound images
        This ensures perfect pixel-level alignment between ultrasound and semantic data
        
        CRITICAL: Accounts for difference between phantom grid (Nx_tot, Ny_tot) and simulation grid (Nx, Ny)
        Updated to handle ROI-centered phantom positioning
        """
        
        print(f"Processing phantom labels with perfect alignment to ultrasound shape: {ultrasound_shape}")
        
        # Extract middle slice for 2D processing
        z_middle = sound_speed_map.shape[2] // 2
        sos_slice = sound_speed_map[:, :, z_middle]
        semantic_slice = semantic_map[:, :, z_middle]
        
        print(f"Original phantom slice shapes: SOS={sos_slice.shape}, Semantic={semantic_slice.shape}")
        
        # CRITICAL: Extract the region that was ACTUALLY SIMULATED
        # Updated to handle ROI-centered positioning instead of simple centering
        
        # Get actual simulation parameters
        num_scan_lines = ultrasound_shape[0]  # Actual scan lines from ultrasound data
        total_scan_width = num_scan_lines * element_width
        print(f"Simulation parameters: {num_scan_lines} scan lines, element_width={element_width}")
        print(f"total_scan_width: {total_scan_width}")
        
        # Calculate the scan coverage area using the actual simulated scan width
        Ny_tot = sound_speed_map.shape[1]  # Full phantom width
        # Use the number of scan lines and element width to compute the actual
        # coverage width in phantom coordinates (pixels)
        total_scan_width = int(total_scan_width)
        # Clamp to valid range and center within the phantom grid
        scan_coverage_width = max(1, min(Ny_tot, total_scan_width))
        scan_start_col = max(0, (Ny_tot - scan_coverage_width) // 2)
        scan_end_col = scan_start_col + scan_coverage_width
        
        print(f"Phantom grid width (Ny_tot): {Ny_tot}")
        print(f"Scan coverage area: cols {scan_start_col}:{scan_end_col} (width: {scan_coverage_width})")
        
        # For post-processing, we need to extract the region that was simulated
        # This corresponds to the scan coverage area
        start_y = scan_start_col
        end_y = scan_end_col
        
        print(f"Extracting simulated region: {start_y}:{end_y}")
        
        # Extract the exact same spatial region as the ultrasound simulation
        sos_region = sos_slice[:, start_y:end_y]
        semantic_region = semantic_slice[:, start_y:end_y]
        
        print(f"Extracted region shapes: SOS={sos_region.shape}, Semantic={semantic_region.shape}")
        
        # Verify that the extracted region width matches the ultrasound scan line coverage
        expected_width = scan_coverage_width
        actual_width = sos_region.shape[1]
        if actual_width != expected_width:
            print(f"WARNING: Extracted width {actual_width} != expected {expected_width}")
        
        # Check tumor coverage in extracted region
        tumor_mask_full = (semantic_slice == self.config.semantic_label).astype(np.float32)
        tumor_mask_extracted = (semantic_region == self.config.semantic_label).astype(np.float32)
        
        total_tumor_pixels = np.sum(tumor_mask_full)
        extracted_tumor_pixels = np.sum(tumor_mask_extracted)
        coverage_percentage = (extracted_tumor_pixels / total_tumor_pixels * 100) if total_tumor_pixels > 0 else 0
        
        print(f"Tumor coverage analysis:")
        print(f"  Total tumor pixels: {total_tumor_pixels}")
        print(f"  Extracted tumor pixels: {extracted_tumor_pixels}")
        print(f"  Coverage percentage: {coverage_percentage:.1f}%")
        
        if coverage_percentage < 80:
            print(f"  ⚠️  WARNING: Only {coverage_percentage:.1f}% of tumor is in extracted region!")
        else:
            print(f"  ✓ Good coverage: {coverage_percentage:.1f}% of tumor is in extracted region")
        
        # STEP 2: Create semantic masks from the extracted region
        # Create binary mask for tumor regions (semantic label 4 = tumor)
        tumor_mask = tumor_mask_extracted
        
        # Create grayscale label image
        gray_label = np.where(tumor_mask == 1, 255, 128).astype(np.uint8)
        
        # STEP 3: Apply the SAME spatial transformations as ultrasound images
        
        # NO transpose needed - both phantom and ultrasound are in the same orientation now
        print(f"Before resize: Tumor={tumor_mask.shape}, Gray={gray_label.shape}, SOS={sos_region.shape}")
        
        # Resize to target shape using the SAME method as ultrasound images
        target_shape = self.config.target_shape
        print(f"Resizing phantom labels to {target_shape} using SAME method as ultrasound images")
        
        tumor_mask_resized = cv2.resize(tumor_mask, target_shape, interpolation=cv2.INTER_NEAREST)
        gray_label_resized = cv2.resize(gray_label, target_shape, interpolation=cv2.INTER_NEAREST)
        sos_region_resized = cv2.resize(sos_region, target_shape, interpolation=cv2.INTER_CUBIC)
        
        # Ensure binary mask is properly thresholded after resize
        tumor_mask_resized = (tumor_mask_resized > 0.5).astype(np.float32)
        
        print(f"Final aligned shapes: Binary={tumor_mask_resized.shape}, Gray={gray_label_resized.shape}, SOS={sos_region_resized.shape}")
        
        # Verify alignment by checking that all outputs have the same shape as ultrasound
        expected_shape = target_shape
        assert tumor_mask_resized.shape == expected_shape, f"Tumor mask shape {tumor_mask_resized.shape} != expected {expected_shape}"
        assert gray_label_resized.shape == expected_shape, f"Gray label shape {gray_label_resized.shape} != expected {expected_shape}"
        assert sos_region_resized.shape == expected_shape, f"SOS region shape {sos_region_resized.shape} != expected {expected_shape}"
        
        print("✓ Perfect spatial alignment achieved between ultrasound and semantic data")
        
        return {
            'tumor_binary_mask': tumor_mask_resized,
            'tumor_gray_label': gray_label_resized,
            'sound_speed_region': sos_region_resized
        }
    
    def _process_phantom_labels(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray,
                               element_width: int) -> Dict:
        """
        DEPRECATED: Use _process_phantom_labels_aligned instead for perfect alignment
        This method is kept for backward compatibility but should not be used
        """
        raise NotImplementedError("Use _process_phantom_labels_aligned for perfect spatial alignment")
    
    def _save_processed_data(
        self,
        processed_results: Dict,
        sample_name: str,
        scans_path: str,
        labels_path: str,
    ) -> Dict:
        """Save processed data in ML-ready format"""

        b_mode = self._gamma_correction(processed_results["b_mode"])
        harmonic = self._gamma_correction(processed_results["harmonic"])

        target_shape = self.config.target_shape
        print(f"Resizing ultrasound images to {target_shape} using simple resize")

        b_mode_transposed = b_mode.T
        harmonic_transposed = harmonic.T

        print(f"B-mode shape before resize: {b_mode_transposed.shape}")
        print(f"Harmonic shape before resize: {harmonic_transposed.shape}")

        b_mode_resized = cv2.resize(
            b_mode_transposed,
            target_shape,
            interpolation=cv2.INTER_CUBIC,
        )
        harmonic_resized = cv2.resize(
            harmonic_transposed,
            target_shape,
            interpolation=cv2.INTER_CUBIC,
        )

        print(
            f"Final ultrasound image shapes: "
            f"B-mode={b_mode_resized.shape}, Harmonic={harmonic_resized.shape}"
        )

        # ------------------------------------------------------------------
        # Optional: apply spatially-varying background blur outside the tumor
        # This keeps the tumor region sharp while making the surrounding
        # speckle / "sprinkle" appearance smoother when requested.
        # ------------------------------------------------------------------
        # Background blur and darkening have been disabled globally.
        # To ensure there is no runtime background blur/darkening, bypass the
        # entire optional blur/composite code path regardless of config values.
        if False:
            tumor_mask = processed_results["phantom_labels"].get("tumor_binary_mask")
            if tumor_mask is not None:
                # Ensure mask is same size as the resized ultrasound images
                mask = tumor_mask
                if mask.shape != b_mode_resized.shape:
                    mask = cv2.resize(mask, (b_mode_resized.shape[1], b_mode_resized.shape[0]), interpolation=cv2.INTER_NEAREST)

                # Make sure mask is binary float in [0,1]
                mask = (mask > 0.5).astype(np.float32)

                # Determine kernel size from sigma if not explicitly set
                ksize_config = getattr(self.config, "background_blur_kernel", None)
                if ksize_config is None:
                    ksize = max(3, int(2 * round(bg_sigma * 3) + 1))
                else:
                    ksize = int(ksize_config) if int(ksize_config) % 2 == 1 else int(ksize_config) + 1

                # Apply Gaussian blur to the full images
                try:
                    b_blurred = cv2.GaussianBlur(b_mode_resized, (ksize, ksize), bg_sigma)
                    h_blurred = cv2.GaussianBlur(harmonic_resized, (ksize, ksize), bg_sigma)
                except Exception:
                    # Fallback: use scipy gaussian_filter if OpenCV fails
                    b_blurred = gaussian_filter(b_mode_resized, sigma=bg_sigma)
                    h_blurred = gaussian_filter(harmonic_resized, sigma=bg_sigma)

                # Optionally scale background brightness (make sprinkle darker if <1.0)
                try:
                    bg_scale = float(getattr(self.config, 'background_brightness_scale', 1.0))
                except Exception:
                    bg_scale = 1.0
                if bg_scale != 1.0:
                    b_blurred = b_blurred * float(bg_scale)
                    h_blurred = h_blurred * float(bg_scale)

                # Combine: keep tumor region from original, background from blurred
                inverse_mask = 1.0 - mask
                b_mode_resized = (mask * b_mode_resized) + (inverse_mask * b_blurred)
                harmonic_resized = (mask * harmonic_resized) + (inverse_mask * h_blurred)

                print(f"Applied background blur: sigma={bg_sigma}, ksize={ksize}")

        # ------------------------------------------------------------------

        seed_metadata = {
            "run_seed": processed_results["simulation_metadata"].get("run_seed"),
            "sample_seed": processed_results["simulation_metadata"].get("sample_seed"),
            "noise_level_seed": processed_results["simulation_metadata"].get(
                "noise_level_seed"
            ),
        }

        scan_data = {
            "noisy_us_scan_b_mode": b_mode_resized,
            "noisy_us_scan_harmonic": harmonic_resized,
            "scan_lines": processed_results["raw_simulation"]["scan_lines"],
            "kgrid_dt": processed_results["raw_simulation"]["kgrid_dt"],
            "kgrid_Nt": processed_results["raw_simulation"]["kgrid_Nt"],
            "full_sound_speed_map": processed_results["raw_simulation"]["full_sound_speed_map"],
            "full_density_map": processed_results["raw_simulation"]["full_density_map"],
            "full_semantic_map": processed_results["raw_simulation"]["full_semantic_map"],
            "simulation_metadata": processed_results["simulation_metadata"],
            "seed_metadata": seed_metadata,
        }
        # ============================================================
        # 3) 저장 후 pkl 안에 들어갈 항목
        # ============================================================
        # scan_*.pkl
        # - noisy_us_scan_b_mode
        # - noisy_us_scan_harmonic
        # - scan_lines
        # - kgrid_dt
        # - kgrid_Nt
        # - full_sound_speed_map
        # - full_density_map
        # - full_semantic_map
        # - simulation_metadata
        #
        # simulation_metadata 안:
        # - sc_w_x
        # - sc_w_y
        # - num_scan_lines
        # - scan_element_width
        # - transducer_element_width
        # - grid_y_size
        # - transducer_width
        # - simulation_time_minutes

        label_data = {
            "clean_phantom_gray": processed_results["phantom_labels"]["tumor_gray_label"],
            "clean_phantom_binary": processed_results["phantom_labels"]["tumor_binary_mask"],
            "clean_phantom_sound_speed": processed_results["phantom_labels"]["sound_speed_region"],
            "seed_metadata": seed_metadata,
        }

        suffix = getattr(self.config, "file_suffix", "")

        scan_file = os.path.join(
            scans_path,
            f"scan_{sample_name}{suffix}.pkl",
        )
        label_file = os.path.join(
            labels_path,
            f"label_{sample_name}{suffix}.pkl",
        )

        with open(scan_file, "wb") as f:
            pickle.dump(scan_data, f)

        with open(label_file, "wb") as f:
            pickle.dump(label_data, f)

        return {
            "scan_file": scan_file,
            "label_file": label_file,
            "processed_data": {**scan_data, **label_data},
        }
    
    def _gamma_correction(self, image: np.ndarray) -> np.ndarray:
        """Apply gamma correction to ultrasound images"""
        # Normalize to [0, 1]
        normalized = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        # Apply gamma correction
        corrected = np.power(normalized, self.config.gamma_correction)
        return corrected
    
    def _visualize_processed_data(self, processed_results: Dict, sample_name: str, noise_level: str):
        """Visualize processed data for quality check"""
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # B-mode image
        axes[0, 0].imshow(processed_results['b_mode'].T, cmap='gray', aspect='auto')
        axes[0, 0].set_title(f'B-mode - Sample {sample_name}')
        axes[0, 0].set_xlabel('Scan Line')
        axes[0, 0].set_ylabel('Depth')
        
        # Harmonic image
        axes[0, 1].imshow(processed_results['harmonic'].T, cmap='gray', aspect='auto')
        axes[0, 1].set_title(f'Harmonic - Sample {sample_name}')
        axes[0, 1].set_xlabel('Scan Line')
        axes[0, 1].set_ylabel('Depth')
        
        # Sound speed region
        axes[0, 2].imshow(processed_results['phantom_labels']['sound_speed_region'], cmap='jet')
        axes[0, 2].set_title('Sound Speed Region')
        
        # Binary mask
        axes[1, 0].imshow(processed_results['phantom_labels']['tumor_binary_mask'], cmap='jet')
        axes[1, 0].set_title('Tumor Binary Mask')
        
        # Gray label
        axes[1, 1].imshow(processed_results['phantom_labels']['tumor_gray_label'], cmap='gray')
        axes[1, 1].set_title('Tumor Gray Label')
        
        # Intensity histograms
        axes[1, 2].hist(processed_results['b_mode'].flatten(), bins=50, alpha=0.7, label='B-mode')
        axes[1, 2].hist(processed_results['harmonic'].flatten(), bins=50, alpha=0.7, label='Harmonic')
        axes[1, 2].set_title('Intensity Histograms')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.show()

def resize_with_aspect_ratio_and_padding(image: np.ndarray, target_shape: Tuple[int, int], 
                                       interpolation: int = cv2.INTER_CUBIC) -> np.ndarray:
    """
    Resize image while preserving aspect ratio and padding with zeros
    
    NOTE: This function is kept for reference but not used in the current pipeline.
    For ultrasound images, simple cv2.resize distortion is preferred over aspect ratio preservation.
    
    Args:
        image: Input image
        target_shape: Target (height, width)
        interpolation: OpenCV interpolation method
        
    Returns:
        Resized and padded image
    """
    target_h, target_w = target_shape
    h, w = image.shape[:2]
    
    # Calculate aspect ratios
    original_aspect = w / h
    target_aspect = target_w / target_h
    
    # Calculate new dimensions preserving aspect ratio
    if original_aspect > target_aspect:
        # Image is wider than target
        new_w = target_w
        new_h = int(target_w / original_aspect)
    else:
        # Image is taller than target
        new_h = target_h
        new_w = int(target_h * original_aspect)
    
    # Resize preserving aspect ratio
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
    
    # Create padded result
    if len(image.shape) == 3:
        padded = np.zeros((target_h, target_w, image.shape[2]), dtype=image.dtype)
    else:
        padded = np.zeros((target_h, target_w), dtype=image.dtype)
    
    # Calculate padding offsets (center the image)
    pad_h = (target_h - new_h) // 2
    pad_w = (target_w - new_w) // 2
    
    # Place resized image in center
    padded[pad_h:pad_h+new_h, pad_w:pad_w+new_w] = resized
    
    print(f"Aspect ratio resize: {w}x{h} -> {new_w}x{new_h} -> padded to {target_w}x{target_h}")
    return padded

def process_batch(simulation_results_path: str, output_path: str, 
                 sample_names: List[str], noise_levels: List[str] = ['low'],
                 config: Optional[PostProcessingConfig] = None) -> Dict:
    """
    Process a batch of simulation results
    
    Args:
        simulation_results_path: Path to simulation results folder
        output_path: Path to save processed results
        sample_names: List of sample names to process
        noise_levels: List of noise levels to process
        config: Post-processing configuration
        
    Returns:
        Dictionary with processing results
    """
    if config is None:
        config = PostProcessingConfig()
    
    processor = PythonSimulationPostProcessor(config)
    
    # Create dataset folders
    os.makedirs(output_path, exist_ok=True)
    
    results = {}
    processed_count = 0
    
    for sample_name in sample_names:
        for noise_level in noise_levels:
            try:
                sample_results = processor.process_sample(
                    sample_name=sample_name,
                    simulation_results_path=simulation_results_path,
                    output_path=output_path,
                    noise_level=noise_level
                )
                
                key = f"sample_{sample_name}_{noise_level}"
                results[key] = sample_results
                processed_count += 1
                
            except Exception as e:
                print(f"Error processing sample {sample_name} with noise {noise_level}: {e}")
                continue
    
    # Save processing metadata
    metadata = {
        'total_processed': processed_count,
        'config': config.__dict__,
        'sample_names': sample_names,
        'noise_levels': noise_levels,
        'output_path': output_path
    }
    
    metadata_file = os.path.join(output_path, 'processing_metadata.json')
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Batch processing completed! Processed {processed_count} samples.")
    print(f"Results saved to: {output_path}")
    
    return results

def main():
    """Main execution function for post-processing"""
    
    # Configuration
    config = PostProcessingConfig(
        target_shape=(256, 256),
        gamma_correction=0.5,
        show_preview=True  # Enable visualization to see results
    )
    
    # Paths (adjust to your setup)
    SIMULATION_RESULTS_PATH = r"D:\CMME\1_data_generation\data\BUSI\python_simulation_results"
    OUTPUT_PATH = r"D:\CMME\1_data_generation\data\BUSI\processed_python_results"
    
    # Processing settings - specifically for benign_10
    SAMPLE_NAMES = ['benign_10']  # Process only benign_10
    NOISE_LEVELS = ['low']  # ['low', 'medium', 'high']
    
    # Process batch
    results = process_batch(
        simulation_results_path=SIMULATION_RESULTS_PATH,
        output_path=OUTPUT_PATH,
        sample_names=SAMPLE_NAMES,
        noise_levels=NOISE_LEVELS,
        config=config
    )
    
    print("Post-processing pipeline completed!")
    print(f"Processed sample benign_10 results:")
    for key, result in results.items():
        print(f"  {key}: {result['scan_file']}")
        print(f"  {key}: {result['label_file']}")

if __name__ == "__main__":
    main() 
