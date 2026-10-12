"""
Python k-Wave Simulation Pipeline
=================================

This script replicates the MATLAB ultrasound simulation workflow using python-kwave.
It follows the same conceptual flow as working_pipeline_in_matlab.m but in Python.

Key features:
- GRF-based phantom generation
- Multiple noise levels
- k-Wave ultrasound simulation
- Progress tracking
- Modular and configurable design
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import scipy.io
from typing import Dict, Tuple, Optional, List
import cv2
import random ###
import zlib
# k-Wave imports
from kwave.data import Vector # type:ignore
from kwave.kgrid import kWaveGrid # type:ignore
from kwave.kmedium import kWaveMedium # type:ignore
from kwave.kspaceFirstOrder3D import kspaceFirstOrder3D # type:ignore
from kwave.ktransducer import NotATransducer, kWaveTransducerSimple # type:ignore
from kwave.options.simulation_execution_options import SimulationExecutionOptions # type:ignore
from kwave.options.simulation_options import SimulationOptions # type:ignore
from kwave.utils.signals import tone_burst # type:ignore 
from kwave.utils.dotdictionary import dotdict # type:ignore
from scipy.ndimage import convolve1d, zoom, gaussian_filter

# Import centralized configuration
from config.pipeline_options import SimulationConfig
import time

FINE_SCATTERER_CORRELATION = 1.0

###
def stable_seed(*parts: str) -> int:
    seed = 0
    for part in parts:
        seed = zlib.crc32(str(part).encode("utf-8"), seed)
    return seed % (2**32)


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
###
###
'''
class GaussianRandomFieldGenerator:
    """Generate Gaussian Random Fields for tissue modeling"""
    
    @staticmethod
    def create_gaussian_random_field(Nx: int, Ny: int, Nz: int, 
                                   sigma: float, kernel_size: int,
                                   coherence_level: str = 'very_high') -> np.ndarray:
        """
        Create a Gaussian Random Field for tissue heterogeneity modeling
        
        Args:
            Nx, Ny, Nz: Grid dimensions
            sigma: Standard deviation for Gaussian kernel
            kernel_size: Size of the Gaussian kernel
            coherence_level: Level of spatial coherence
            
        Returns:
            3D GRF array
        """
        # Create random noise
        noise = np.random.randn(Nx, Ny, Nz)
        
        # Create Gaussian kernel for filtering
        kernel_1d = GaussianRandomFieldGenerator._gaussian_kernel_1d(kernel_size, sigma)
        
        # Apply coherence based on level
        coherence_map = {
            'low': 0.3,
            'medium': 0.5, 
            'high': 0.7,
            'very_high': 0.9
        }
        coherence_factor = coherence_map.get(coherence_level, 0.9)
        
        # Apply spatial filtering
        
        # Filter along each dimension
        grf = noise
        grf = convolve1d(grf, kernel_1d, axis=0, mode='nearest')
        grf = convolve1d(grf, kernel_1d, axis=1, mode='nearest')
        if Nz > 1:
            grf = convolve1d(grf, kernel_1d, axis=2, mode='nearest')
        
        # Apply coherence factor
        grf = coherence_factor * grf + (1 - coherence_factor) * noise
        
        # Normalize to [0, 1]
        grf = (grf - np.min(grf)) / (np.max(grf) - np.min(grf))
        
        return grf
    
    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
        """Create 1D Gaussian kernel"""
        x = np.arange(size) - size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / np.sum(kernel)
'''
###
###
class GaussianRandomFieldGenerator:
    """Generate Gaussian Random Fields for tissue modeling"""
    
    @staticmethod
    def create_gaussian_random_field(Nx: int, Ny: int, Nz: int, 
                                   sigma: float, kernel_size: int,
                                   coherence_level: str = 'very_high') -> np.ndarray:
        """
        Create a Gaussian Random Field for tissue heterogeneity modeling
        
        Args:
            Nx, Ny, Nz: Grid dimensions
            sigma: Standard deviation for Gaussian kernel
            kernel_size: Size of the Gaussian kernel
            coherence_level: Level of spatial coherence
            
        Returns:
            3D GRF array
        """
        if coherence_level == 'very_high':
            return GaussianRandomFieldGenerator._create_patchy_gaussian_random_field(
                Nx, Ny, Nz, sigma, kernel_size
            )

        coherence_specs = {
            'low': {'sigma_scale': 0.75, 'passes': 1, 'smooth_weight': 0.35, 'contrast': 1.0},
            'medium': {'sigma_scale': 1.0, 'passes': 1, 'smooth_weight': 0.60, 'contrast': 1.15},
            'high': {'sigma_scale': 1.5, 'passes': 2, 'smooth_weight': 0.85, 'contrast': 1.45},
        }
        spec = coherence_specs.get(coherence_level, coherence_specs['high'])
        effective_sigma = max(float(sigma) * spec['sigma_scale'], 0.5)
        effective_kernel_size = GaussianRandomFieldGenerator._valid_kernel_size(
            kernel_size, effective_sigma
        )

        print(
            "GRF settings: "
            f"level={coherence_level}, sigma={effective_sigma:.2f}, "
            f"kernel={effective_kernel_size}, passes={spec['passes']}"
        )

        # Create random noise
        noise = np.random.randn(Nx, Ny, Nz)
        
        # Create Gaussian kernel for filtering
        kernel_1d = GaussianRandomFieldGenerator._gaussian_kernel_1d(
            effective_kernel_size, effective_sigma
        )
        
        # Apply repeated spatial filtering to form large coherent tissue patches.
        smooth_grf = noise.copy()
        for _ in range(spec['passes']):
            smooth_grf = convolve1d(smooth_grf, kernel_1d, axis=0, mode='reflect')
            smooth_grf = convolve1d(smooth_grf, kernel_1d, axis=1, mode='reflect')
            if Nz > 1:
                smooth_grf = convolve1d(smooth_grf, kernel_1d, axis=2, mode='reflect')
        
        # Keep a controllable amount of fine texture. For very_high coherence this is
        # almost entirely the smoothed field, giving patch-like fat/glandular regions.
        smooth_weight = spec['smooth_weight']
        grf = smooth_weight * smooth_grf + (1 - smooth_weight) * noise
        
        # Normalize to [0, 1]
        grf = GaussianRandomFieldGenerator._normalize_unit_interval(grf)
        
        # Increase separation around the classification threshold. This makes the
        # thresholded tissue labels look like coherent patches instead of speckles.
        if spec['contrast'] != 1.0:
            grf = 1 / (1 + np.exp(-spec['contrast'] * 8.0 * (grf - 0.5)))
            grf = GaussianRandomFieldGenerator._normalize_unit_interval(grf)
        
        return grf

    @staticmethod
    def _create_patchy_gaussian_random_field(
        Nx: int, Ny: int, Nz: int, sigma: float, kernel_size: int
    ) -> np.ndarray:
        """
        Create a coarse-control-point GRF for tissue-like fat/gland patches.

        Instead of smoothing voxel-scale noise, this samples a low-resolution
        Gaussian field and interpolates it to the simulation grid. The result is
        still a GRF, but its correlation length is large enough that thresholding
        produces coherent anatomical-looking regions.
        """
        patch_size = max(18, int(round(float(sigma) * 5.0)))
        coarse_nx = max(5, int(np.ceil(Nx / patch_size)) + 3)
        coarse_ny = max(5, int(np.ceil(Ny / patch_size)) + 3)
        coarse_nz = 1 if Nz <= 1 else max(3, min(6, int(np.ceil(Nz / patch_size)) + 2))

        print(
            "Patchy GRF settings: "
            f"patch_size={patch_size}, coarse_grid={coarse_nx}x{coarse_ny}x{coarse_nz}"
        )

        coarse_field = np.random.randn(coarse_nx, coarse_ny, coarse_nz)
        coarse_kernel = GaussianRandomFieldGenerator._gaussian_kernel_1d(7, 1.2)
        for _ in range(2):
            coarse_field = convolve1d(coarse_field, coarse_kernel, axis=0, mode='reflect')
            coarse_field = convolve1d(coarse_field, coarse_kernel, axis=1, mode='reflect')
            if coarse_nz > 1:
                coarse_field = convolve1d(coarse_field, coarse_kernel, axis=2, mode='reflect')

        zoom_factors = (Nx / coarse_nx, Ny / coarse_ny, Nz / coarse_nz)
        grf = zoom(coarse_field, zoom_factors, order=3, mode='reflect')
        if grf.shape[0] < Nx or grf.shape[1] < Ny or grf.shape[2] < Nz:
            pad_width = (
                (0, max(0, Nx - grf.shape[0])),
                (0, max(0, Ny - grf.shape[1])),
                (0, max(0, Nz - grf.shape[2])),
            )
            grf = np.pad(grf, pad_width, mode='edge')
        grf = grf[:Nx, :Ny, :Nz]

        # Smooth interpolation artifacts at the final resolution while preserving
        # large domains. Kernel size is auto-clamped to the requested sigma.
        final_sigma = max(float(sigma), 1.0)
        final_kernel_size = GaussianRandomFieldGenerator._valid_kernel_size(
            kernel_size, final_sigma
        )
        final_kernel = GaussianRandomFieldGenerator._gaussian_kernel_1d(
            final_kernel_size, final_sigma
        )
        grf = convolve1d(grf, final_kernel, axis=0, mode='reflect')
        grf = convolve1d(grf, final_kernel, axis=1, mode='reflect')
        if Nz > 1:
            # Use gentler elevation smoothing so each B-mode plane keeps the same
            # tissue layout without becoming identical slabs.
            z_kernel = GaussianRandomFieldGenerator._gaussian_kernel_1d(9, 1.5)
            grf = convolve1d(grf, z_kernel, axis=2, mode='reflect')

        # A very small amount of fine GRF texture keeps the tissue from looking
        # artificially flat while preserving patch boundaries.
        fine_texture = np.random.randn(Nx, Ny, Nz)
        fine_kernel = GaussianRandomFieldGenerator._gaussian_kernel_1d(9, 2.0)
        fine_texture = convolve1d(fine_texture, fine_kernel, axis=0, mode='reflect')
        fine_texture = convolve1d(fine_texture, fine_kernel, axis=1, mode='reflect')
        if Nz > 1:
            fine_texture = convolve1d(fine_texture, fine_kernel, axis=2, mode='reflect')

        grf = 0.94 * GaussianRandomFieldGenerator._normalize_unit_interval(grf)
        grf += 0.06 * GaussianRandomFieldGenerator._normalize_unit_interval(fine_texture)
        grf = GaussianRandomFieldGenerator._normalize_unit_interval(grf)

        # Stronger separation around 0.5 makes the semantic fat/gland split form
        # stable patches after thresholding.
        grf = 1 / (1 + np.exp(-14.0 * (grf - 0.5)))
        return GaussianRandomFieldGenerator._normalize_unit_interval(grf)

    @staticmethod
    def _valid_kernel_size(size: int, sigma: float) -> int:
        """Return an odd kernel large enough to represent the requested sigma."""
        min_size = int(np.ceil(6 * sigma)) + 1
        valid_size = max(int(size), min_size, 3)
        if valid_size % 2 == 0:
            valid_size += 1
        return valid_size

    @staticmethod
    def _normalize_unit_interval(array: np.ndarray) -> np.ndarray:
        """Normalize an array to [0, 1] with a constant-field fallback."""
        value_range = np.max(array) - np.min(array)
        if value_range < 1e-12:
            return np.full_like(array, 0.5, dtype=np.float64)
        return (array - np.min(array)) / value_range
    
    @staticmethod
    def _gaussian_kernel_1d(size: int, sigma: float) -> np.ndarray:
        """Create 1D Gaussian kernel"""
        size = GaussianRandomFieldGenerator._valid_kernel_size(size, sigma)
        x = np.arange(size) - size // 2
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return kernel / np.sum(kernel)
###
class TissuePropertiesManager:
    """Manage tissue acoustic properties and noise configurations"""
    
    def __init__(self):
        self.tissue_properties = {
            'background': {'sos': 1530, 'density': 1000}, #1495 ###1600
            'fatty': {'sos': 1515, 'density': 1000}, #1504 ###1600
            'glandular': {'sos': 1530, 'density': 1040}, #1508
            'tumor': {'sos': 1570, 'density': 995}  #1530 ###1480
        }
        
        self.noise_configs = {
            'low': {
                'background': 0.002, ###0.0001, 
                'fatty': 0.004, ###0.0003,
                'glandular': 0.006, ###0.0002,
                'tumor': 0.001 ###0.00005
            },
            'medium': {
                'background': 0.005,
                'fatty': 0.015,
                'glandular': 0.01,
                'tumor': 0.002
            },
            'high': {
                'background': 0.02,
                'fatty': 0.06,
                'glandular': 0.04,
                'tumor': 0.01
            }
        }
    
    def create_tissue_maps(self, tumor_mask_3d: np.ndarray, grf: np.ndarray, 
                          noise_level: str = 'low') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sound speed and density maps with tissue-specific noise
        
        Args:
            tumor_mask_3d: 3D binary mask for tumor regions
            grf: Gaussian random field for tissue classification
            noise_level: Level of noise to apply
            
        Returns:
            Tuple of (sound_speed_map, density_map, semantic_map)
        """
        
        # Use the CORRECT dimensions from the input arrays (Nx_tot, Ny_tot, Nz_tot)
        Nx_tot, Ny_tot, Nz_tot = tumor_mask_3d.shape
        print(f"Creating tissue maps for grid: {Nx_tot} x {Ny_tot} x {Nz_tot}")
        
        # Create tissue type masks based on GRF thresholding (like MATLAB)
        threshold = 0.5
        fatty_mask = (grf < threshold) & ~tumor_mask_3d
        glandular_mask = (grf >= threshold) & ~tumor_mask_3d
        
        # Background mask should be for regions where there's no tissue modeling
        # In MATLAB, background_mask = zeros, but ALL non-tumor regions get tissue properties
        # So we'll model ALL non-tumor regions as either fatty or glandular
        background_mask = np.zeros_like(tumor_mask_3d, dtype=bool)  # Keep minimal background
        
        print(f"Tissue mask statistics:")
        print(f"  Tumor pixels: {np.sum(tumor_mask_3d)} ({100*np.sum(tumor_mask_3d)/tumor_mask_3d.size:.2f}%)")
        print(f"  Fatty pixels: {np.sum(fatty_mask)} ({100*np.sum(fatty_mask)/tumor_mask_3d.size:.2f}%)")
        print(f"  Glandular pixels: {np.sum(glandular_mask)} ({100*np.sum(glandular_mask)/tumor_mask_3d.size:.2f}%)")
        print(f"  Background pixels: {np.sum(background_mask)} ({100*np.sum(background_mask)/tumor_mask_3d.size:.2f}%)")
        
        # Create semantic map for visualization
        semantic_map = np.zeros((Nx_tot, Ny_tot, Nz_tot))
        semantic_map[background_mask] = 1  # Background (should be minimal)
        semantic_map[fatty_mask] = 2       # Fatty tissue
        semantic_map[glandular_mask] = 3   # Glandular tissue
        semantic_map[tumor_mask_3d] = 4    # Tumor/cyst
        
        # Initialize property maps with background values (ensure float64 for noise multiplication)
        sound_speed_map = np.full((Nx_tot, Ny_tot, Nz_tot), self.tissue_properties['background']['sos'], dtype=np.float64)
        density_map = np.full((Nx_tot, Ny_tot, Nz_tot), self.tissue_properties['background']['density'], dtype=np.float64)
        
        print(f"Initialized maps with background properties:")
        print(f"  Sound speed: {self.tissue_properties['background']['sos']} m/s")
        print(f"  Density: {self.tissue_properties['background']['density']} kg/m³")
        
        # Apply tissue properties in order (like MATLAB)
        # Fatty tissue
        if np.any(fatty_mask):
            props = self.tissue_properties['fatty']
            sound_speed_map[fatty_mask] = float(props['sos'])
            density_map[fatty_mask] = float(props['density'])
            print(f"Applied fatty tissue properties: SoS={props['sos']}, density={props['density']}")
        
        # Glandular tissue
        if np.any(glandular_mask):
            props = self.tissue_properties['glandular']
            sound_speed_map[glandular_mask] = float(props['sos'])
            density_map[glandular_mask] = float(props['density'])
            print(f"Applied glandular tissue properties: SoS={props['sos']}, density={props['density']}")
        
        # Tumor tissue (applied last to override)
        if np.any(tumor_mask_3d):
            props = self.tissue_properties['tumor']
            sound_speed_map[tumor_mask_3d] = float(props['sos'])
            density_map[tumor_mask_3d] = float(props['density'])
            print(f"Applied tumor tissue properties: SoS={props['sos']}, density={props['density']}")
        
        # ------------------------------------------------------------------
        # HYBRID COHERENT PATCH SCATTERING
        #
        # Preserve the best-performing baseline behavior:
        #   - tissue-specific baseline noise levels
        #   - strongly correlated SoS/density fluctuations
        #
        # Add one key change:
        #   - patch strength is generated in the B-mode plane and repeated
        #     through elevation, so it is not averaged away in 3-D.
        # ------------------------------------------------------------------
        noise_std = self.noise_configs[noise_level]
        shape = (Nx_tot, Ny_tot, Nz_tot)
        print(f"Applying coherent patch noise (level: {noise_level}):")

        # Generate patch layout on the fixed 256x256 reference grid,
        # then resize it to the active simulation grid. This prevents the
        # pattern from changing merely because Nx and Ny changed.
        ref_nx = 256
        ref_ny = 256

        patch_raw_large = np.random.standard_normal(
            (ref_nx, ref_ny)
        )
        patch_raw_medium = np.random.standard_normal(
            (ref_nx, ref_ny)
        )
        patch_raw_small = np.random.standard_normal(
            (ref_nx, ref_ny)
        )

        patch_large_ref = gaussian_filter(
            patch_raw_large,
            sigma=(26.0, 26.0),
            mode='reflect'
        )
        patch_medium_ref = gaussian_filter(
            patch_raw_medium,
            sigma=(12.0, 12.0),
            mode='reflect'
        )
        patch_small_ref = gaussian_filter(
            patch_raw_small,
            sigma=(5.0, 5.0),
            mode='reflect'
        )

        patch_ref = (
            0.55 * patch_large_ref
            + 0.35 * patch_medium_ref
            + 0.10 * patch_small_ref
        )

        patch_2d = cv2.resize(
            patch_ref,
            (Ny_tot, Nx_tot),
            interpolation=cv2.INTER_LINEAR
        )
        patch_2d -= float(np.mean(patch_2d))
        patch_2d /= float(np.std(patch_2d)) + 1e-12

        # Distinct but softly bounded bright-patch regions.
        patch_strength_2d = 1.0 / (
            1.0 + np.exp(-5.5 * (patch_2d - 0.15))
        )
        patch_strength_2d = np.clip(
            patch_strength_2d ** 1.35, 0.0, 1.0
        )

        # Exact elevation coherence: the same patch layout is used for all z.
        patch_strength = np.repeat(
            patch_strength_2d[:, :, np.newaxis],
            Nz_tot,
            axis=2
        )

        # Preserve anatomical modulation already present in the original GRF.
        grf_centered = np.abs(grf - 0.5) * 2.0
        grf_modulation = 0.80 + 0.40 * np.clip(
            grf_centered, 0.0, 1.0
        )

        # Local noise ranges from 35% to 240% of the original tissue setting.
        local_std_gain = (
            0.08 + 5.20 * patch_strength
        ) * grf_modulation

        # Mostly shared noise preserves the strong baseline texture.
        # Independent components prevent perfect SoS-density identity.
        # Fine scatterers are also generated on a fixed XY reference grid.
        # Their exact B-mode appearance still depends on acoustic resolution,
        # but the underlying spatial realization remains aligned.
        ref_shape = (256, 256, Nz_tot)

        shared_ref = np.random.standard_normal(ref_shape)
        sos_ref = np.random.standard_normal(ref_shape)
        density_ref = np.random.standard_normal(ref_shape)

        def resize_reference_noise(
            reference_noise: np.ndarray
        ) -> np.ndarray:
            resized = np.empty(shape, dtype=np.float64)
            for z_idx in range(Nz_tot):
                resized[:, :, z_idx] = cv2.resize(
                    reference_noise[:, :, z_idx],
                    (Ny_tot, Nx_tot),
                    interpolation=cv2.INTER_AREA
                    if Nx_tot < 256 or Ny_tot < 256
                    else cv2.INTER_LINEAR
                )

            # Restore zero mean and unit variance after interpolation.
            resized -= np.mean(resized)
            resized /= np.std(resized) + 1e-12
            return resized

        shared_fine = resize_reference_noise(shared_ref)
        sos_independent = resize_reference_noise(sos_ref)
        density_independent = resize_reference_noise(density_ref)

        # Experimental variant: use exactly the same microscopic spatial
        # realization for sound-speed and density perturbations.  The tissue-
        # specific amplitudes remain different, but their fine-pattern
        # locations and shapes are perfectly correlated.
        correlation = FINE_SCATTERER_CORRELATION
        independent_weight = np.sqrt(
            1.0 - correlation ** 2
        )

        sos_fine = (
            correlation * shared_fine
            + independent_weight * sos_independent
        )
        density_fine = (
            correlation * shared_fine
            + independent_weight * density_independent
        )

        sos_fine = np.clip(sos_fine, -3.0, 3.0)
        density_fine = np.clip(density_fine, -3.0, 3.0)

        for tissue_type, mask in [
            ('fatty', fatty_mask),
            ('glandular', glandular_mask),
        ]:
            if not np.any(mask):
                continue

            base_std = float(noise_std[tissue_type])
            sos_std_map = base_std * local_std_gain
            density_std_map = (
                1.65 * base_std * local_std_gain
            )

            sound_speed_map[mask] *= (
                1.0
                + sos_std_map[mask]
                * sos_fine[mask]
            )
            density_map[mask] *= (
                1.0
                + density_std_map[mask]
                * density_fine[mask]
            )

            print(
                f"  {tissue_type}: base_std={base_std:.5f}, "
                f"local gain={np.min(local_std_gain[mask]):.2f}-"
                f"{np.max(local_std_gain[mask]):.2f}"
            )

        # Keep cyst/tumor comparatively uniform.
        if np.any(tumor_mask_3d):
            tumor_std = float(noise_std['tumor'])
            tumor_sos_noise = np.random.standard_normal(shape)
            tumor_density_noise = np.random.standard_normal(shape)

            # Make the tumor/cyst genuinely anechoic:
            # no internal microscopic SoS or density fluctuations.
            # The boundary contrast is preserved by the baseline tissue values.
            sound_speed_map[tumor_mask_3d] *= 1.0
            density_map[tumor_mask_3d] *= 1.0

        background_regions = ~(
            fatty_mask | glandular_mask | tumor_mask_3d
        )
        if np.any(background_regions):
            background_std = float(noise_std['background'])
            background_noise = np.random.standard_normal(shape)

            sound_speed_map[background_regions] *= (
                1.0
                + background_std
                * background_noise[background_regions]
            )
            density_map[background_regions] *= (
                1.0
                + background_std
                * background_noise[background_regions]
            )

        # Diagnostic image: this must show visible patches before k-Wave runs.
        debug_folder = "debug_visualizations"
        os.makedirs(debug_folder, exist_ok=True)
        patch_path = os.path.join(
            debug_folder,
            f"debug_patch_strength_{noise_level}.png"
        )

        plt.figure(figsize=(6, 6))
        plt.imshow(
            patch_strength_2d,
            cmap='gray',
            vmin=0.0,
            vmax=1.0,
            aspect='auto'
        )
        plt.colorbar(label='Patch strength')
        plt.title(
            f'Coherent scattering patches - {noise_level}'
        )
        plt.tight_layout()
        plt.savefig(
            patch_path,
            dpi=150,
            bbox_inches='tight'
        )
        plt.close()

        print(
            f"DEBUG: Patch-strength map saved to {patch_path}"
        )

        # Print final statistics
        print(f"Final property map statistics:")
        print(f"  Sound speed range: {np.min(sound_speed_map):.1f} - {np.max(sound_speed_map):.1f} m/s")
        print(f"  Density range: {np.min(density_map):.1f} - {np.max(density_map):.1f} kg/m³")
        
        # Save debug visualization of speed of sound map
        self._save_debug_sos_visualization(sound_speed_map, semantic_map, noise_level)
        
        return sound_speed_map, density_map, semantic_map
    
    def _save_debug_sos_visualization(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray, noise_level: str):
        """Save simple debug visualization of speed of sound map"""
        
        # Extract middle slice
        z_middle = sound_speed_map.shape[2] // 2
        sos_slice = sound_speed_map[:, :, z_middle]
        semantic_slice = semantic_map[:, :, z_middle]
        
        # Create simple visualization
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Speed of sound map
        im1 = axes[0].imshow(sos_slice, cmap='jet', aspect='auto')
        axes[0].set_title(f'Speed of Sound Map - {noise_level}\nRange: {np.min(sos_slice):.0f} - {np.max(sos_slice):.0f} m/s')
        axes[0].set_xlabel('Width (pixels)')
        axes[0].set_ylabel('Depth (pixels)')
        cbar1 = plt.colorbar(im1, ax=axes[0])
        cbar1.set_label('Speed [m/s]')
        
        # Semantic map
        im2 = axes[1].imshow(semantic_slice, cmap='viridis', vmin=1, vmax=4, aspect='auto')
        axes[1].set_title(f'Tissue Types - {noise_level}')
        axes[1].set_xlabel('Width (pixels)')
        axes[1].set_ylabel('Depth (pixels)')
        cbar2 = plt.colorbar(im2, ax=axes[1])
        cbar2.set_ticks([1, 2, 3, 4])
        cbar2.set_ticklabels(['Background', 'Fatty', 'Glandular', 'Tumor'])
        
        plt.tight_layout()
        
        # Save to debug folder
        debug_folder = "debug_visualizations"
        os.makedirs(debug_folder, exist_ok=True)
        save_path = os.path.join(debug_folder, f'debug_sos_map_{noise_level}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"DEBUG: Speed of sound map saved to {save_path}")

class PhantomResizer:
    """Utility class for resizing phantoms while preserving aspect ratio and ensuring ROI coverage"""
    
    @staticmethod
    def resize_preserve_aspect_ratio(input_image: np.ndarray, target_height: int, 
                                   target_width: int, resize_method: str = 'nearest') -> np.ndarray:
        """
        Resize an image while preserving aspect ratio (MATLAB-compatible implementation)
        
        This function resizes an input image to fit within target dimensions while
        preserving the original aspect ratio. If necessary, it pads the result to
        exactly match the target dimensions.
        
        Args:
            input_image: Original image to resize
            target_height: Target height (rows) - Nx_tot
            target_width: Target width (columns) - Ny_tot  
            resize_method: Interpolation method ('nearest', 'bilinear', 'bicubic')
            
        Returns:
            Resized and padded image matching target dimensions
        """
        # Get original dimensions
        rows, cols = input_image.shape[:2]
        original_aspect_ratio = cols / rows
        target_aspect_ratio = target_width / target_height
        
        print(f"Original phantom dimensions: {rows} x {cols} (aspect ratio: {original_aspect_ratio:.3f})")
        print(f"Target grid dimensions: {target_height} x {target_width} (aspect ratio: {target_aspect_ratio:.3f})")
        
        # Calculate new dimensions that preserve aspect ratio
        if original_aspect_ratio > target_aspect_ratio:
            # Original is wider than target
            new_width = target_width
            new_height = round(new_width / original_aspect_ratio)
            if new_height > target_height:
                new_height = target_height
                new_width = round(new_height * original_aspect_ratio)
        else:
            # Original is taller than target
            new_height = target_height
            new_width = round(new_height * original_aspect_ratio)
            if new_width > target_width:
                new_width = target_width
                new_height = round(new_width / original_aspect_ratio)
        
        print(f"Resized dimensions (aspect preserved): {new_height} x {new_width}")
        
        # Resize preserving aspect ratio
        if resize_method == 'nearest':
            interpolation = cv2.INTER_NEAREST
        elif resize_method == 'bilinear':
            interpolation = cv2.INTER_LINEAR
        elif resize_method == 'bicubic':
            interpolation = cv2.INTER_CUBIC
        else:
            interpolation = cv2.INTER_NEAREST
        
        resized_aspect_preserved = cv2.resize(input_image, (new_width, new_height), interpolation=interpolation)
        
        # Create padded version to match target dimensions (centered)
        #         # % Create padded version to match target dimensions
        # resized_image = zeros(target_height, target_width);
        # start_row = round((target_height - new_height) / 2) + 1;
        # start_col = round((target_width - new_width) / 2) + 1;
        # resized_image(start_row:start_row+new_height-1, start_col:start_col+new_width-1) = resized_aspect_preserved;

        resized_image = np.zeros((target_height, target_width), dtype=input_image.dtype)
        start_row = round((target_height - new_height) / 2) 
        start_col = round((target_width - new_width) / 2)
        
        print(f"Phantom positioned at rows {start_row}:{start_row + new_height}, cols {start_col}:{start_col + new_width}")
        
        resized_image[start_row:start_row + new_height, start_col:start_col + new_width] = resized_aspect_preserved
        
        return resized_image
    
    @staticmethod
    def position_phantom_roi_aware(input_image: np.ndarray, target_height: int, target_width: int, 
                                 scan_coverage_info: dict, tumor_threshold: int = 100, 
                                 resize_method: str = 'nearest') -> np.ndarray:
        """
        Position phantom ensuring tumor ROI is well within scan coverage area
        
        Args:
            input_image: Original phantom image
            target_height: Target grid height (Nx_tot)
            target_width: Target grid width (Ny_tot)
            scan_coverage_info: Dictionary with scan coverage parameters
            tumor_threshold: Pixel value threshold for tumor detection
            resize_method: Interpolation method
            
        Returns:
            Positioned phantom with tumor guaranteed within scan coverage
        """
        print(f"\n🎯 SIMPLE PHANTOM POSITIONING")
        print(f"=" * 40)
        print(f"Original phantom: {input_image.shape[0]} x {input_image.shape[1]}")
        print(f"Target grid: {target_height} x {target_width}")
        print(f"Scan coverage width: {scan_coverage_info['scan_coverage_width']}")
        
        # Step 1: Resize phantom to fit exactly within scan coverage width
        # cv2.resize takes (width, height) = (cols, rows)
        mask_2d_resized = cv2.resize(input_image, (scan_coverage_info['scan_coverage_width'], target_height), interpolation=cv2.INTER_NEAREST)
        print(f"Resized phantom to: {mask_2d_resized.shape[0]} x {mask_2d_resized.shape[1]} (rows x cols)")
        
        # Step 2: Create full canvas and center the phantom in Y direction (columns)
        canvas = np.zeros((target_height, target_width), dtype=input_image.dtype)
        
        # Center in Y direction (columns) - place phantom in middle of grid width
        center_y = target_width // 2  # Center column position
        y1 = center_y - scan_coverage_info['scan_coverage_width'] // 2  # Left edge
        y2 = y1 + scan_coverage_info['scan_coverage_width']  # Right edge
        
        print(f"Canvas: {canvas.shape[0]} x {canvas.shape[1]} (rows x cols)")
        print(f"Placing phantom at columns {y1}:{y2} (center at {center_y})")
        
        # Step 3: Place resized phantom in center of canvas
        canvas[:, y1:y2] = mask_2d_resized
        mask_2d_resized = canvas
        
        # Verify phantom positioning
        tumor_mask_check = (mask_2d_resized == tumor_threshold)
        if np.any(tumor_mask_check):
            tumor_rows, tumor_cols = np.where(tumor_mask_check)
            tumor_min_col, tumor_max_col = tumor_cols.min(), tumor_cols.max()
            tumor_center_col = (tumor_min_col + tumor_max_col) / 2
            
            print(f"✅ POSITIONING VERIFICATION:")
            print(f"  Final phantom shape: {mask_2d_resized.shape[0]} x {mask_2d_resized.shape[1]}")
            print(f"  Tumor spans columns: {tumor_min_col}:{tumor_max_col}")
            print(f"  Tumor center column: {tumor_center_col:.1f}")
            print(f"  Scan coverage: {scan_coverage_info['scan_start_col']}:{scan_coverage_info['scan_end_col']}")
            print(f"  Tumor within scan: {'✅ YES' if tumor_min_col >= scan_coverage_info['scan_start_col'] and tumor_max_col <= scan_coverage_info['scan_end_col'] else '(Error) NO'}")
            
            # Calculate margins to scan edges
            left_margin = tumor_min_col - scan_coverage_info['scan_start_col']
            right_margin = scan_coverage_info['scan_end_col'] - tumor_max_col
            print(f"  Margins to scan edges: left={left_margin:.1f}, right={right_margin:.1f}")
        else:
            print(f"⚠️ No tumor found after positioning")
        
        return mask_2d_resized

class UltrasoundSimulator:
    """Main ultrasound simulation class using python-kwave"""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.tissue_manager = TissuePropertiesManager()
        self.grf_generator = GaussianRandomFieldGenerator()
        self.last_transducer_debug_info = {}

        # Calculate grid parameters using Vector objects
        self.pml_size_points = Vector([config.pml_x_size, config.pml_y_size, config.pml_z_size])
        self.grid_size_points = Vector([config.Nx, config.Ny, config.Nz])
        
        # Calculate grid spacing
        grid_spacing_x = config.x_size / config.Nx
        ###
        grid_spacing_y = config.y_size / config.Ny
        grid_spacing_z = config.z_size / config.Nz
        ###
        self.grid_spacing_meters = Vector([grid_spacing_x, grid_spacing_y, grid_spacing_z]) ###grid_spacing_x, grid_spacing_x])
        
        # Setup k-Wave grid using correct API
        self.kgrid = kWaveGrid(self.grid_size_points, self.grid_spacing_meters)
        t_end = (config.Nx * grid_spacing_x) * 2.2 / config.c0 # CFL Condition is satissfied always ;  delta t and delta x
        self.kgrid.makeTime(config.c0, t_end=t_end)
        
        
        # Setup input signal
        self.input_signal = tone_burst(
            1 / self.kgrid.dt, 
            config.tone_burst_freq, 
            config.tone_burst_cycles
        )
        self.input_signal = (config.source_strength / (config.c0 * config.rho0)) * self.input_signal
        
        # Setup medium
        self.medium = kWaveMedium(
            sound_speed=None,  # will be set for each scan line
            alpha_coeff=config.alpha_coeff,
            alpha_power=config.alpha_power,
            BonA=config.BonA
        )
    
    def setup_transducer(self) -> NotATransducer:
        """Setup the ultrasound transducer"""
        config = self.config
        
        transducer_params = dotdict()
        ###
        number_elements_before = 32
        element_width_before = 2
        ###
        # Create simple transducer - smaller for reduced domain
        
        transducer_params.number_elements =  int(round(number_elements_before * config.sc_w_y)) ###4 ###8 ###32  # Reduced from 32
        transducer_params.element_width = int((number_elements_before * element_width_before * config.dy_before) / (transducer_params.number_elements * config.dy)) ###2 ### config.element_width
        transducer_params.element_length = 24  # Reduced from 24
        transducer_params.element_spacing = 0
        transducer_params.radius = float("inf")
        
        # Position transducer
        
        
        transducer_width = (transducer_params.number_elements * transducer_params.element_width + 
                          (transducer_params.number_elements - 1) * transducer_params.element_spacing)
        
        transducer_params.position = np.round([
            1, 
            self.grid_size_points.y / 2 - transducer_width / 2, 
            self.grid_size_points.z / 2 - transducer_params.element_length / 2
        ])
        
        ###
        '''
        transducer_width = (
            transducer_params.number_elements
            * transducer_params.element_width
            + (transducer_params.number_elements - 1)
            * transducer_params.element_spacing
        )

        transducer_params.position = np.round([
            1,
            1 + (
                self.grid_size_points.y
                - transducer_width
            ) / 2,
            1 + (
                self.grid_size_points.z
                - transducer_params.element_length
            ) / 2
        ]).astype(int)
        '''
        if transducer_width > self.grid_size_points.y:
            raise ValueError(
                f"Transducer width {transducer_width} exceeds "
                f"grid y-size {self.grid_size_points.y}"
            )
        
        print("Grid y size:", self.grid_size_points.y)
        print("Transducer width:", transducer_width)
        print("Transducer position:", transducer_params.position)

        self.last_transducer_debug_info = {
            "sc_w_x": float(config.sc_w_x),
            "sc_w_y": float(config.sc_w_y),
            "num_scan_lines": int(config.number_scan_lines),
            "scan_element_width": int(config.element_width),
            "transducer_element_width": int(transducer_params.element_width),
            "grid_x_size": int(self.grid_size_points.x),
            "grid_y_size": int(self.grid_size_points.y),
            "transducer_width": int(transducer_width),
        }

        ###
        simple_transducer = kWaveTransducerSimple(self.kgrid, **transducer_params)
        
        # Create NotATransducer for beamforming
        not_transducer_params = dotdict()
        not_transducer_params.sound_speed = config.c0
        not_transducer_params.focus_distance = 20e-3  # Reduced from 20e-3
        not_transducer_params.elevation_focus_distance = 19e-3  # Reduced from 19e-3
        not_transducer_params.steering_angle = 0
        not_transducer_params.transmit_apodization = "Hanning"
        not_transducer_params.receive_apodization = "Rectangular"
        not_transducer_params.active_elements = np.ones((simple_transducer.number_elements, 1))
        not_transducer_params.input_signal = self.input_signal
        
        return NotATransducer(simple_transducer, self.kgrid, **not_transducer_params)
    
    def load_phantom_mask(self, mask_path: str) -> np.ndarray:
        """Load and prepare phantom mask for simulation with ROI-aware positioning ensuring tumor is well within scan coverage"""
        config = self.config
        
        # Load 2D mask
        mask_2d = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_2d is None:
            raise FileNotFoundError(f"Could not load mask from {mask_path}")
        
        # Calculate total grid size like MATLAB code:
        # Nx_tot = Nx (simulation grid size in depth direction)
        # Ny_tot = Ny + number_scan_lines * element_width (includes scan line coverage)
        # Nz_tot = Nz (elevation direction)
        Nx_tot = config.Nx  # Depth direction
        Ny_tot = config.Ny + config.number_scan_lines * config.element_width  # Width + scan coverage
        Nz_tot = config.Nz  # Elevation direction
        
        print(f"Calculated grid dimensions:")
        print(f"  Nx_tot (depth): {Nx_tot}")
        print(f"  Ny_tot (width + scan coverage): {Ny_tot}")
        print(f"  Nz_tot (elevation): {Nz_tot}")
        print(f"  Scan coverage: {config.number_scan_lines} scan lines × {config.element_width} width = {config.number_scan_lines * config.element_width}")
        
        # Calculate scan coverage information
        scan_coverage_width = config.number_scan_lines * config.element_width
        scan_start_col = config.Ny  // 2 
        scan_end_col = Ny_tot - config.Ny // 2
        
        scan_coverage_info = {
            'scan_start_col': scan_start_col,
            'scan_end_col': scan_end_col,
            'scan_coverage_width': scan_coverage_width,
            'total_grid_width': Ny_tot
        }
        
        print(f"Scan coverage info:")
        print(f"  Total grid width: {Ny_tot}")
        print(f"  Scan coverage: cols {scan_start_col}:{scan_end_col} (width: {scan_end_col - scan_start_col})")
        print(f"  Scan lines coverage: {scan_coverage_width}")
        
        # 🎯 Use simple positioning to ensure tumor is well within scan coverage
        resizer = PhantomResizer()
        mask_2d_resized = resizer.position_phantom_roi_aware(
            input_image=mask_2d,
            target_height=Nx_tot,
            target_width=Ny_tot,
            scan_coverage_info=scan_coverage_info,
            tumor_threshold=100,  # Assumed tumor pixel value
            resize_method='nearest'
        )
        
        # Create 3D mask (tumor regions have value 100 or 255, depending on source)
        # Be more flexible with tumor detection values
        tumor_threshold = 100 # Assumed that the tumor is 100 in the mask
        tumor_mask_3d = np.repeat((mask_2d_resized == tumor_threshold)[:, :, np.newaxis], Nz_tot, axis=2)
        
        print(f"Tumor mask created:")
        print(f"  Unique values in 2D mask: {np.unique(mask_2d_resized)}")
        print(f"  Tumor threshold: {tumor_threshold}")
        print(f"  Tumor pixels found: {np.sum(tumor_mask_3d)}")
        print(f"  Tumor percentage: {100 * np.sum(tumor_mask_3d) / tumor_mask_3d.size:.2f}%")
        
        # Visualize the resized phantom to verify ROI placement
        self._save_phantom_resize_visualization(mask_2d, mask_2d_resized, config, scan_coverage_info)
        
        return tumor_mask_3d, (Nx_tot, Ny_tot, Nz_tot)
    
    def _save_phantom_resize_visualization(self, original_mask: np.ndarray, 
                                         resized_mask: np.ndarray, config: SimulationConfig, scan_coverage_info: dict):
        """Save visualization of phantom resizing to verify ROI placement with ROI-aware positioning"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('ROI-Aware Phantom Positioning & Scan Coverage Analysis', fontsize=16)
        
        # Original mask
        axes[0, 0].imshow(original_mask, cmap='gray')
        axes[0, 0].set_title(f'Original Mask\n{original_mask.shape[0]}×{original_mask.shape[1]}')
        axes[0, 0].axis('off')
        
        # Add tumor detection to original
        tumor_mask_orig = (original_mask == 100)
        if np.any(tumor_mask_orig):
            tumor_rows, tumor_cols = np.where(tumor_mask_orig)
            tumor_min_row, tumor_max_row = tumor_rows.min(), tumor_rows.max()
            tumor_min_col, tumor_max_col = tumor_cols.min(), tumor_cols.max()
            rect_orig = Rectangle((tumor_min_col, tumor_min_row), 
                                tumor_max_col - tumor_min_col, tumor_max_row - tumor_min_row,
                                linewidth=2, edgecolor='red', facecolor='none', linestyle='-')
            axes[0, 0].add_patch(rect_orig)
            axes[0, 0].text(tumor_min_col, tumor_min_row-5, 'Original Tumor', 
                           color='red', fontweight='bold', fontsize=10)
        
        # Resized mask with scan coverage overlay
        axes[0, 1].imshow(resized_mask, cmap='gray', alpha=0.8)
        axes[0, 1].set_title(f'ROI-Aware Positioned Mask\n{resized_mask.shape[0]}×{resized_mask.shape[1]}')
        axes[0, 1].axis('off')
        
        # Get scan coverage info
        scan_start_col = scan_coverage_info['scan_start_col']
        scan_end_col = scan_coverage_info['scan_end_col']
        
        # Draw scan line coverage rectangle
        rect = Rectangle((scan_start_col, 0), scan_end_col - scan_start_col, resized_mask.shape[0], 
                        linewidth=3, edgecolor='blue', facecolor='none', linestyle='-', alpha=0.8)
        axes[0, 1].add_patch(rect)
        axes[0, 1].text(scan_start_col + 5, 15, 'Scan Coverage', 
                       color='blue', fontweight='bold', fontsize=12, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Add margins visualization
        margin_percentage = 0.05  # 5% margin on each side for simple positioning
        scan_coverage_width = scan_end_col - scan_start_col
        effective_scan_start = scan_start_col + scan_coverage_width * margin_percentage
        effective_scan_end = scan_end_col - scan_coverage_width * margin_percentage
        
        rect_effective = Rectangle((effective_scan_start, 0), 
                                 effective_scan_end - effective_scan_start, resized_mask.shape[0], 
                                 linewidth=2, edgecolor='green', facecolor='none', linestyle='--', alpha=0.8)
        axes[0, 1].add_patch(rect_effective)
        axes[0, 1].text(effective_scan_start + 5, 35, 'Effective Scan Area\n(with margins)', 
                       color='green', fontweight='bold', fontsize=10,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Highlight final tumor position
        tumor_mask_final = (resized_mask == 100)
        if np.any(tumor_mask_final):
            tumor_rows, tumor_cols = np.where(tumor_mask_final)
            tumor_min_row, tumor_max_row = tumor_rows.min(), tumor_rows.max()
            tumor_min_col, tumor_max_col = tumor_cols.min(), tumor_cols.max()
            tumor_center_col = (tumor_min_col + tumor_max_col) / 2
            
            rect_tumor = Rectangle((tumor_min_col, tumor_min_row), 
                                 tumor_max_col - tumor_min_col, tumor_max_row - tumor_min_row,
                                 linewidth=3, edgecolor='red', facecolor='none', linestyle='-')
            axes[0, 1].add_patch(rect_tumor)
            axes[0, 1].text(tumor_min_col, tumor_min_row-10, 'Positioned Tumor', 
                           color='red', fontweight='bold', fontsize=12,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # ROI analysis - focused view
        ax = axes[1, 0]
        if np.any(tumor_mask_final):
            # Create focused view around tumor
            tumor_center_row = (tumor_min_row + tumor_max_row) // 2
            tumor_center_col = int(tumor_center_col)
            
            # Define region around tumor
            view_size = 100  # pixels around tumor center
            view_start_row = max(0, tumor_center_row - view_size)
            view_end_row = min(resized_mask.shape[0], tumor_center_row + view_size)
            view_start_col = max(0, tumor_center_col - view_size)
            view_end_col = min(resized_mask.shape[1], tumor_center_col + view_size)
            
            focused_view = resized_mask[view_start_row:view_end_row, view_start_col:view_end_col]
            ax.imshow(focused_view, cmap='Reds')
            ax.set_title(f'Focused Tumor View\nCenter: ({tumor_center_row}, {tumor_center_col:.0f})')
            
            # Add scan coverage lines in focused view
            if scan_start_col >= view_start_col and scan_start_col <= view_end_col:
                ax.axvline(x=scan_start_col - view_start_col, color='blue', linewidth=2, linestyle='-', alpha=0.8, label='Scan Start')
            if scan_end_col >= view_start_col and scan_end_col <= view_end_col:
                ax.axvline(x=scan_end_col - view_start_col, color='blue', linewidth=2, linestyle='-', alpha=0.8, label='Scan End')
            if effective_scan_start >= view_start_col and effective_scan_start <= view_end_col:
                ax.axvline(x=effective_scan_start - view_start_col, color='green', linewidth=2, linestyle='--', alpha=0.8, label='Effective Start')
            if effective_scan_end >= view_start_col and effective_scan_end <= view_end_col:
                ax.axvline(x=effective_scan_end - view_start_col, color='green', linewidth=2, linestyle='--', alpha=0.8, label='Effective End')
            
            ax.legend(fontsize=8)
        else:
            ax.text(0.5, 0.5, 'No tumor found\nin resized image', 
                   ha='center', va='center', transform=ax.transAxes, fontsize=12)
            ax.set_title('Focused Tumor View')
        ax.set_xlabel('Width (pixels)')
        ax.set_ylabel('Depth (pixels)')
        
        # Detailed analysis
        ax = axes[1, 1]
        ax.axis('off')
        
        # Calculate comprehensive statistics
        analysis_text = []
        analysis_text.append("🎯 ROI-AWARE POSITIONING ANALYSIS")
        analysis_text.append("=" * 35)
        
        # Original tumor stats
        if np.any(tumor_mask_orig):
            orig_tumor_pixels = np.sum(tumor_mask_orig)
            analysis_text.append(f"Original tumor pixels: {orig_tumor_pixels}")
            analysis_text.append(f"Original tumor area: {orig_tumor_pixels / (original_mask.shape[0] * original_mask.shape[1]) * 100:.1f}%")
        
        # Final tumor stats
        if np.any(tumor_mask_final):
            final_tumor_pixels = np.sum(tumor_mask_final)
            analysis_text.append(f"Final tumor pixels: {final_tumor_pixels}")
            analysis_text.append(f"Final tumor area: {final_tumor_pixels / (resized_mask.shape[0] * resized_mask.shape[1]) * 100:.1f}%")
            
            # Positioning analysis
            analysis_text.append("")
            analysis_text.append("📍 POSITIONING RESULTS:")
            analysis_text.append(f"Tumor cols: {tumor_min_col}:{tumor_max_col}")
            analysis_text.append(f"Tumor center: {tumor_center_col:.1f}")
            analysis_text.append(f"Scan coverage: {scan_start_col}:{scan_end_col}")
            analysis_text.append(f"Effective scan: {effective_scan_start:.1f}:{effective_scan_end:.1f}")
            
            # Check if tumor is within effective scan area
            tumor_in_effective = (tumor_min_col >= effective_scan_start and tumor_max_col <= effective_scan_end)
            tumor_in_scan = (tumor_min_col >= scan_start_col and tumor_max_col <= scan_end_col)
            
            analysis_text.append("")
            analysis_text.append("✅ COVERAGE VERIFICATION:")
            analysis_text.append(f"Tumor in scan area: {'✅ YES' if tumor_in_scan else '(Error) NO'}")
            analysis_text.append(f"Tumor in effective area: {'✅ YES' if tumor_in_effective else '(Error) NO'}")
            
            # Calculate margins
            left_margin = tumor_min_col - scan_start_col
            right_margin = scan_end_col - tumor_max_col
            effective_left_margin = tumor_min_col - effective_scan_start
            effective_right_margin = effective_scan_end - tumor_max_col
            
            analysis_text.append("")
            analysis_text.append("📏 MARGINS:")
            analysis_text.append(f"To scan edges: L={left_margin:.1f}, R={right_margin:.1f}")
            analysis_text.append(f"To effective edges: L={effective_left_margin:.1f}, R={effective_right_margin:.1f}")
            
            # Success assessment
            analysis_text.append("")
            if tumor_in_effective and effective_left_margin > 10 and effective_right_margin > 10:
                analysis_text.append("🎉 SUCCESS: Tumor well positioned!")
                analysis_text.append("   ✓ Tumor within effective scan area")
                analysis_text.append("   ✓ Adequate margins on both sides")
            elif tumor_in_scan:
                analysis_text.append("⚠️  PARTIAL: Tumor in scan area but")
                analysis_text.append("   close to edges")
            else:
                analysis_text.append("(Error) ISSUE: Tumor extends outside scan area")
        else:
            analysis_text.append("(Error) ERROR: No tumor found in final image")
        
        # Display analysis
        analysis_str = '\n'.join(analysis_text)
        ax.text(0.05, 0.95, analysis_str, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # Save the visualization
        save_path = 'phantom_roi_aware_positioning_analysis.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"ROI-aware positioning analysis saved to: {save_path}")
        plt.close()
        
        # Print summary to console
        if np.any(tumor_mask_final):
            print(f"\n🎯 ROI-AWARE POSITIONING SUMMARY:")
            print(f"  Tumor position: cols {tumor_min_col}:{tumor_max_col} (center: {tumor_center_col:.1f})")
            print(f"  Scan coverage: cols {scan_start_col}:{scan_end_col}")
            print(f"  Effective scan: cols {effective_scan_start:.1f}:{effective_scan_end:.1f}")
            print(f"  Tumor in effective area: {'✅ YES' if tumor_in_effective else '(Error) NO'}")
            print(f"  Margins: left={effective_left_margin:.1f}, right={effective_right_margin:.1f}")
            
            if tumor_in_effective and effective_left_margin > 10 and effective_right_margin > 10:
                print(f"  🎉 SUCCESS: Tumor optimally positioned with adequate margins!")
            elif tumor_in_scan:
                print(f"  ⚠️  WARNING: Tumor in scan area but close to edges")
            else:
                print(f"  (Error) ERROR: Tumor positioning failed - extends outside scan area")
        else:
            print(f"  (Error) ERROR: No tumor found after positioning")
    
    def run_simulation(self, sample_name: str, input_data_path: str, output_path: str,
                      noise_levels: List[str] = ['low'], run_simulation: bool = True) -> Dict:
        """
        Run the complete simulation pipeline for a sample
        
        Args:
            sample_name: Sample name (e.g., 'benign_1', 'malignant_5')
            input_data_path: Path to processed phantom data folder  
            output_path: Path to save simulation results
            noise_levels: List of noise levels to simulate
            run_simulation: Whether to run simulation or just load results
            
        Returns:
            Dictionary with simulation results
        """
        config = self.config

        # Deterministic seed per sample.
        # Same sample name -> same GRF and tissue texture on every run.
        base_seed = self.config.base_seed if self.config.base_seed is not None else 0

        sample_seed = stable_seed(
            str(base_seed),
            "sample",
            sample_name,
        )
        set_global_seed(sample_seed)
        print(f"Deterministic sample seed: {sample_seed}")
        
        # Setup paths
        ###sample_folder = os.path.join(input_data_path, sample_name)
        ###mask_path = os.path.join(sample_folder, "mask.png")
        
        ###if not os.path.exists(mask_path):
            ###raise FileNotFoundError(f"Mask not found for sample {sample_name}: {mask_path}")
        # Setup paths
        sample_folder = os.path.join(input_data_path, sample_name)

        folder_mask_path = os.path.join(sample_folder, "mask.png")
        flat_mask_path = os.path.join(input_data_path, f"{sample_name}_mask.png")

        if os.path.exists(folder_mask_path):
            mask_path = folder_mask_path
        elif os.path.exists(flat_mask_path):
            mask_path = flat_mask_path
        else:
            raise FileNotFoundError(
                f"Mask not found for sample {sample_name}: "
                f"tried '{folder_mask_path}' and '{flat_mask_path}'"
            )
        # Create output directory
        results_folder = os.path.join(output_path, sample_name)
        os.makedirs(results_folder, exist_ok=True)
        
        print(f"Processing sample {sample_name}...")
        
        # Load and prepare phantom
        tumor_mask_3d, grid_dims = self.load_phantom_mask(mask_path)
        Nx_tot, Ny_tot, Nz_tot = grid_dims
        
        # Generate the anatomical GRF on a fixed reference grid.
        # Same sample seed + same reference grid => same physical pattern
        # for sc_w_x/sc_w_y = 1.0, 0.4, etc.
        ref_nx = int(config.Nx_before)
        ref_ny = int(config.Ny_before)

        print(
            f"Generating reference GRF at "
            f"{ref_nx}x{ref_ny}x{Nz_tot}..."
        )
        grf_ref = self.grf_generator.create_gaussian_random_field(
            ref_nx, ref_ny, Nz_tot,
            config.grf_sigma,
            config.grf_kernel_size,
            config.coherence_level
        )

        grf = zoom(
            grf_ref,
            (
                Nx_tot / ref_nx,
                Ny_tot / ref_ny,
                1.0
            ),
            order=1,
            mode='nearest',
            prefilter=False
        )

        # zoom can differ by one sample because of rounding.
        grf = grf[:Nx_tot, :Ny_tot, :Nz_tot]
        if grf.shape != (Nx_tot, Ny_tot, Nz_tot):
            fixed = np.empty(
                (Nx_tot, Ny_tot, Nz_tot),
                dtype=grf.dtype
            )
            for z_idx in range(Nz_tot):
                fixed[:, :, z_idx] = cv2.resize(
                    grf_ref[:, :, z_idx],
                    (Ny_tot, Nx_tot),
                    interpolation=cv2.INTER_LINEAR
                )
            grf = fixed

        grf -= np.min(grf)
        grf /= np.max(grf) + 1e-12
        ###
        print("GRF shape:", grf.shape)
        print("Tumor mask shape:", tumor_mask_3d.shape)

        if grf.shape != tumor_mask_3d.shape:
            raise ValueError(
                f"Shape mismatch: "
                f"grf={grf.shape}, "
                f"tumor_mask={tumor_mask_3d.shape}"
            )
        ###
        results = {}
        
        sample_output_path = os.path.join(output_path, sample_name)
        os.makedirs(sample_output_path, exist_ok=True)

        # Process each noise level
        for noise_level in noise_levels:
            level_seed = stable_seed(
                str(base_seed),
                "sample",
                sample_name,
                "noise",
                noise_level,
            )
            set_global_seed(level_seed)
            print(f"Deterministic noise-level seed: {level_seed}")

            print(f"Processing noise level: {noise_level}")
            
            # Create tissue property maps
            sound_speed_map, density_map, semantic_map = self.tissue_manager.create_tissue_maps(
                tumor_mask_3d, grf, noise_level
            )
            
            # Save phantom data
            phantom_file = os.path.join(results_folder, f"phantom_{noise_level}.mat")
            scipy.io.savemat(phantom_file, {
                'sound_speed_map': sound_speed_map,
                'density_map': density_map,
                'semantic_map': semantic_map,
                'grf': grf,
                'tumor_mask_3d': tumor_mask_3d,
                'sample_name': sample_name,  # Include sample name for reference
                'run_seed': int(base_seed),
                'sample_seed': int(sample_seed),
                'noise_level_seed': int(level_seed),
                'fine_scatterer_correlation': float(FINE_SCATTERER_CORRELATION),
            })
            
            # Visualize phantom
            self._visualize_phantom(sound_speed_map, semantic_map, sample_name, noise_level, results_folder)
            
            if run_simulation:
                # Run k-Wave simulation
                print("Running k-Wave simulation...")
                simulation_start_time = time.perf_counter()

                scan_lines = self._run_kwave_simulation(
                    sound_speed_map=sound_speed_map,
                    density_map=density_map,
                )

                simulation_time_minutes = (time.perf_counter() - simulation_start_time) / 60.0

                print(
                    f"Simulation runtime for {sample_name} [{noise_level}] "
                    f"({self.config.sc_w_x}, {self.config.sc_w_y}): "
                    f"{simulation_time_minutes:.2f} minutes"
                )
                
                # Save simulation results
                simulation_file = os.path.join(results_folder, f"simulation_{noise_level}.mat")
                scipy.io.savemat(simulation_file, {
                    'scan_lines': scan_lines,
                    'sound_speed_map': sound_speed_map,
                    'density_map': density_map,
                    'kgrid_dt': self.kgrid.dt,
                    'kgrid_Nt': self.kgrid.Nt,
                    'element_width': config.element_width,
                    'sample_name': sample_name,  # Include sample name for reference
                    "run_seed": int(base_seed),
                    "sample_seed": int(sample_seed),
                    "noise_level_seed": int(level_seed),
                    "fine_scatterer_correlation": float(FINE_SCATTERER_CORRELATION),
                    "sc_w_x": float(self.config.sc_w_x),
                    "sc_w_y": float(self.config.sc_w_y),
                    "num_scan_lines": int(self.last_transducer_debug_info.get("num_scan_lines", self.config.number_scan_lines)),
                    "scan_element_width": int(self.last_transducer_debug_info.get("scan_element_width", self.config.element_width)),
                    "transducer_element_width": int(self.last_transducer_debug_info.get("transducer_element_width", 0)),
                    "grid_x_size": int(self.last_transducer_debug_info.get("grid_x_size", self.config.Nx)),
                    "grid_y_size": int(self.last_transducer_debug_info.get("grid_y_size", self.config.Ny)),
                    "transducer_width": int(self.last_transducer_debug_info.get("transducer_width", 0)),
                    "simulation_time_minutes": float(simulation_time_minutes),
                    "pml_x_size": int(self.config.pml_x_size),
                    "pml_y_size": int(self.config.pml_y_size),
                    "tone_burst_freq": float(self.config.tone_burst_freq),
                    "tone_burst_cycles": float(self.config.tone_burst_cycles),
                })
                
                results[noise_level] = {
                    'scan_lines': scan_lines,
                    'phantom_file': phantom_file,
                    'simulation_file': simulation_file
                }
            else:
                results[noise_level] = {
                    'phantom_file': phantom_file
                }
        
        print(f"Completed processing sample {sample_name}")
        return results
    
    def _run_kwave_simulation(self, sound_speed_map: np.ndarray, 
                             density_map: np.ndarray) -> np.ndarray:
        """Run the k-Wave simulation for all scan lines"""
        config = self.config
        
        # Setup transducer
        transducer = self.setup_transducer()
        
        # Initialize scan lines storage
        scan_lines = np.zeros((config.number_scan_lines, self.kgrid.Nt))
        
        # Run simulation for each scan line
        medium_position = 0
        
        for scan_line_index in range(config.number_scan_lines):
            print(f"Computing scan line {scan_line_index + 1}/{config.number_scan_lines}")
            
            # Extract medium slice
            end_pos = medium_position + self.grid_size_points.y
            self.medium.sound_speed = sound_speed_map[:, medium_position:end_pos, :]
            self.medium.density = density_map[:, medium_position:end_pos, :]
            
            # Set GPU usage based on config
            use_gpu = config.data_cast == 'gpuArray-single'
            
            # Setup simulation options for this scan line - use 'single' for data_cast always
            input_filename = f"scan_line_{scan_line_index}.h5"
            scan_simulation_options = SimulationOptions(
                pml_inside=False,
                pml_size=self.pml_size_points,
                data_cast='single',  # Always use 'single' - GPU is controlled via execution_options
                data_recast=True,
                save_to_disk=True,  # Required for CPU simulations
                input_filename=input_filename,
                save_to_disk_exit=False
            )
            
            # Run k-Wave simulation
            try:
                sensor_data = kspaceFirstOrder3D(
                    medium=self.medium,
                    kgrid=self.kgrid,
                    source=transducer,
                    sensor=transducer,
                    simulation_options=scan_simulation_options,
                    execution_options=SimulationExecutionOptions(is_gpu_simulation=use_gpu)
                )
                
                # Extract scan line
                scan_lines[scan_line_index, :] = transducer.scan_line(
                    transducer.combine_sensor_data(sensor_data["p"].T)
                )
                
                # Clean up temporary file
                if os.path.exists(input_filename):
                    os.remove(input_filename)
                
            except Exception as e:
                print(f"Error in scan line {scan_line_index}: {e}")
                # Fill with zeros if simulation fails
                scan_lines[scan_line_index, :] = np.zeros(self.kgrid.Nt)
                
                # Clean up temporary file on error
                if os.path.exists(input_filename):
                    os.remove(input_filename)
            
            # Update position
            medium_position += config.element_width
        
        return scan_lines
    
    def _visualize_phantom(self, sound_speed_map: np.ndarray, semantic_map: np.ndarray,
                          sample_name: str, noise_level: str, output_folder: str):
        """Visualize phantom properties"""
        
        # Extract middle slice
        z_middle = sound_speed_map.shape[2] // 2
        sos_slice = sound_speed_map[:, :, z_middle]
        semantic_slice = semantic_map[:, :, z_middle]
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Sound speed map
        im1 = axes[0].imshow(sos_slice, aspect='auto', cmap='jet')
        axes[0].set_title(f'Sound Speed - Sample {sample_name} - {noise_level}')
        axes[0].set_xlabel('Width [pixels]')
        axes[0].set_ylabel('Depth [pixels]')
        plt.colorbar(im1, ax=axes[0], label='Speed [m/s]')
        
        # Semantic map
        im2 = axes[1].imshow(semantic_slice, aspect='auto', cmap='viridis')
        axes[1].set_title(f'Tissue Types - Sample {sample_name}')
        axes[1].set_xlabel('Width [pixels]')
        axes[1].set_ylabel('Depth [pixels]')
        cbar = plt.colorbar(im2, ax=axes[1])
        cbar.set_ticks([1, 2, 3, 4])
        cbar.set_ticklabels(['Background', 'Fatty', 'Glandular', 'Tumor'])
        
        plt.tight_layout()
        
        # Save figure
        save_path = os.path.join(output_folder, f'phantom_visualization_{noise_level}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

def main():
    """Main execution function"""
    
    # Configuration
    config = SimulationConfig()
    
    # Paths (adjust these to your setup)
    INPUT_DATA_PATH = r"D:\CMME\1_data_generation\data\BUSI\processed_data_v2"
    OUTPUT_PATH = r"D:\CMME\1_data_generation\data\BUSI\python_simulation_results"
    
    # Simulation settings
    SAMPLE_NAMES = ['benign_1', 'malignant_5']  # List of sample names to process
    NOISE_LEVELS = ['low']  # ['low', 'medium', 'high']
    RUN_SIMULATION = True   # Set to False to skip simulation
    
    # Create simulator
    simulator = UltrasoundSimulator(config)
    
    # Process each sample
    all_results = {}
    
    for sample_name in SAMPLE_NAMES:
        try:
            results = simulator.run_simulation(
                sample_name=sample_name,
                input_data_path=INPUT_DATA_PATH,
                output_path=OUTPUT_PATH,
                noise_levels=NOISE_LEVELS,
                run_simulation=RUN_SIMULATION
            )
            all_results[sample_name] = results
            
        except Exception as e:
            print(f"Error processing sample {sample_name}: {e}")
            continue
    
    print("Simulation pipeline completed!")
    print(f"Processed {len(all_results)} samples successfully")

if __name__ == "__main__":
    main() 
