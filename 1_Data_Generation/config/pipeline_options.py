"""
Base Options and Default Configuration for Ultrasound Simulation Pipeline
=========================================================================

This module provides default configuration options and common settings
that can be easily used across the pipeline components.
"""

import os
import argparse
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
###
import math
class BaseOptions:
    """Base configuration class with default options for the pipeline"""
    
    def __init__(self):
        # =================================================================
        # 📁 DEFAULT DATA PATHS
        # =================================================================
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        work_root = os.path.abspath(
            os.environ.get("SEMFAR_WORK_ROOT", os.path.join(project_root, "local_outputs"))
        )
        self.busi_dataset_path = os.path.abspath(
            os.environ.get(
                "BUSI_DATASET_PATH",
                os.path.join(project_root, "local_data", "Dataset_BUSI_with_GT"),
            )
        )
        self.processed_phantoms_path = os.path.join(work_root, "processed_phantoms")
        self.simulation_results_path = os.path.join(work_root, "python_simulation_results")
        self.final_output_path = os.path.join(work_root, "final_training_data")
        
        # =================================================================
        # 🎯 DEFAULT SAMPLE PROCESSING OPTIONS
        # =================================================================
        self.default_sample_range = (1, 100)  # (start_id, end_id)
        self.max_samples = 100
        self.noise_levels = ['low']  # ['low', 'medium', 'high']
        self.target_phantom_size = (256, 256)
        
        # =================================================================
        # 🚀 DEFAULT PIPELINE STAGES
        # =================================================================
        self.run_data_preparation = False  # Skip since already done
        self.run_simulation = True
        self.run_post_processing = True
        self.auto_continue = True  # Don't wait for user input
        
        # =================================================================
        # 🖥️ TECHNICAL DEFAULT SETTINGS
        # =================================================================
        self.use_gpu = True  # Enable GPU by default for faster processing
        self.data_cast = 'gpuArray-single'  # GPU acceleration
        self.cpu_data_cast = 'single'  # CPU fallback
        
        # =================================================================
        # 👁️ VISUALIZATION DEFAULT OPTIONS
        # =================================================================
        self.show_phantoms = False  # Disabled for batch processing
        self.show_simulation_previews = False
        self.show_post_processing_previews = False
        self.save_visualizations = True
        
        # =================================================================
        # 📊 WANDB DEFAULT CONFIGURATION
        # =================================================================
        self.enable_wandb = True  # Enable experiment tracking
        self.wandb_project = "super_resolution_demo"
        self.wandb_entity = None  # Set to your wandb username/team
        self.wandb_dir = None  # Directory for wandb files (None uses default)
        self.wandb_log_frequency = 1  # Log every N samples
        
        # =================================================================
        # 🔧 SIMULATION DEFAULT PARAMETERS
        # =================================================================
        # Grid parameters
        self.sc = 1.0  # scaling factor
        self.sc_w_x = 1
        self.sc_w_y = 1
        self.Nx_before = 256
        self.Nx = int(self.Nx_before * self.sc_w_x)
        self.Ny_before = 256
        self.Ny = int(self.Ny_before * self.sc_w_y) ###32 ###64 ###256 ### Ny_before = 256
        self.Nz = 48   # Reduced from 128 for faster processing
        # self.dx = 0.000152  # Grid spacing in x [m]
        # self.dy = 0.000152  # Grid spacing in y [m] 
        # self.dz = 0.000152  # Grid spacing in z [m]
        ###
        self.scale_variants = [
            {"name": "sc1.0", "sc_w_x": 1.0, "sc_w_y": 1.0},
            {"name": "sc0.5", "sc_w_x": 0.5, "sc_w_y": 0.5},
        ]
        self.base_seed = None
        # Physical parameters
        self.x_size = 40e-3  # [m]
        self.y_size = 40e-3  ###   # Will be calculated from x_size
        self.z_size = None   # Will be calculated from x_size
        

        # Step size
        self.dx_before = self.x_size / self.Nx_before
        self.dx = self.x_size / self.Nx
        self.dy_before = self.y_size / self.Ny_before
        self.dy = self.y_size / self.Ny  ### = dy_before / sc_w_y
        self.dz = 40e-3/256 ###self.dx
        
        # PML parameters
        self.pml_x_size_before = int(math.ceil(10)) ###
        self.pml_x_size = max(8, int(math.ceil(self.pml_x_size_before * self.dx_before / self.dx))) ###1.25 ###10  # Reduced for faster processing
        self.pml_y_size_before = int(math.ceil(10))
        self.pml_y_size = max(8, int(math.ceil(self.pml_y_size_before * self.dy_before / self.dy))) ###1.25 ###3 ###10
        self.pml_z_size = 8
        
        # Medium properties
        self.c0 = 1540  # Sound speed [m/s]
        self.rho0 = 1000  # Density [kg/m^3]
        self.alpha_coeff = 0.75  # Absorption coefficient
        self.alpha_power = 1.5   # Absorption power law exponent
        self.BonA = 6.0  # Nonlinearity parameter
        
        # Source parameters 
        self.source_strength = 1e6
        self.tone_burst_freq = 1e6 * math.sqrt(self.sc_w_x) ###1.5e6  # Frequency [Hz] - Reduced for better penetration
        self.tone_burst_cycles = 4 * math.sqrt(self.sc_w_x) ### 4
        # ==============================scan lines and element_width before & after ===================================
        self.number_scan_lines_before = 384 ###384
        self.element_width_before = 2
        ###element_width*dy = scan_line_step and this should be 1/self.sc_w_y  times = 8 times, therefore self.element_width = self.element_width_before
        self.element_width = int(self.element_width_before) #2 # Element width in pixels
        self.number_scan_lines =  int(self.number_scan_lines_before * self.element_width_before * self.dy_before / (self.element_width*self.dy)) ###96 ###48 ###96  ###
        
        # ================================================================================================
        # GRF parameters
        self.grf_sigma = 4.0  # Reduced for faster processing
        self.grf_kernel_size = 21  # Reduced for faster processing
        self.coherence_level = 'high'
        # Generation-time blur: applied to the GRF (lateral axis) before tissue mapping
        # Set to 0.0 to disable. Units are in pixels (scan-line units along Ny).
        self.generation_blur_sigma = 0.0
        
        # =================================================================
        # 📊 POST-PROCESSING DEFAULT PARAMETERS
        # =================================================================
        self.target_shape = (256, 256)  # Target shape for final outputs
        self.show_preview = False  # Disabled for batch processing
        
        # =================================================================
        # 🎨 COMMONLY USED SAMPLE SETS
        # =================================================================
        self.quick_test_samples = ["benign_1", "benign_10", "malignant_1", "malignant_10"]
        self.debug_samples = ["benign_10"]  # Single sample for debugging
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments"""
        
        # Update common arguments if they exist
        arg_mappings = {
            'gpu': 'use_gpu',
            'visualize': 'show_phantoms',
            'max_samples': 'max_samples',
            'noise_levels': 'noise_levels',
            'noise_level': 'noise_levels',  # Single noise level
            'wandb_project': 'wandb_project',
            'wandb_entity': 'wandb_entity',
            'wandb_dir': 'wandb_dir',
            'no_wandb': 'enable_wandb',  # Inverse mapping
            'auto_continue': 'auto_continue'
        }
        
        for arg_name, config_attr in arg_mappings.items():
            if hasattr(args, arg_name) and getattr(args, arg_name) is not None:
                value = getattr(args, arg_name)
                
                # Handle special cases
                if arg_name == 'no_wandb':
                    setattr(self, config_attr, not value)  # Inverse
                elif arg_name == 'noise_level':
                    setattr(self, config_attr, [value])  # Convert single to list
                elif arg_name == 'visualize':
                    # Enable all visualization options
                    self.show_phantoms = value
                    self.show_simulation_previews = value
                    self.show_post_processing_previews = value
                else:
                    setattr(self, config_attr, value)
        
        # Update data cast based on GPU setting
        if hasattr(self, 'use_gpu'):
            self.data_cast = 'gpuArray-single' if self.use_gpu else 'single'
        
        print(f"Configuration updated from arguments:")
        print(f"  GPU enabled: {self.use_gpu}")
        print(f"  Data cast: {self.data_cast}")
        print(f"  Visualizations: {self.show_phantoms}")
        print(f"  WandB enabled: {self.enable_wandb}")
    
    def get_gpu_config(self) -> Dict[str, Any]:
        """Get GPU-optimized configuration"""
        return {
            'use_gpu': True,
            'data_cast': 'gpuArray-single',
            'show_visualizations': False,
            'enable_wandb': True
        }
    
    def get_cpu_config(self) -> Dict[str, Any]:
        """Get CPU-optimized configuration"""
        return {
            'use_gpu': False,
            'data_cast': 'single',
            'show_visualizations': False,
            'enable_wandb': False
        }
    
    def get_debug_config(self) -> Dict[str, Any]:
        """Get debug configuration with visualizations enabled"""
        return {
            'use_gpu': False,
            'data_cast': 'single',
            'show_phantoms': True,
            'show_simulation_previews': True,
            'show_post_processing_previews': True,
            'enable_wandb': False,
            'max_samples': 1
        }
    
    def get_batch_config(self, samples: int = 100) -> Dict[str, Any]:
        """Get batch processing configuration"""
        return {
            'use_gpu': True,
            'data_cast': 'gpuArray-single',
            'show_visualizations': False,
            'enable_wandb': True,
            'auto_continue': True,
            'max_samples': samples,
            'noise_levels': ['low']
        }
    
    def get_multi_noise_config(self, samples: int = 20) -> Dict[str, Any]:
        """Get configuration for multiple noise levels"""
        return {
            'use_gpu': True,
            'data_cast': 'gpuArray-single',
            'show_visualizations': False,
            'enable_wandb': True,
            'auto_continue': True,
            'max_samples': samples,
            'noise_levels': ['low', 'medium', 'high']
        }

@dataclass  
class SimulationConfig(BaseOptions):
    """Simulation-specific configuration inheriting from BaseOptions"""
    
    def __init__(self, **kwargs):
        # Initialize base options
        super().__init__()
        
        # Override with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown config parameter '{key}' = {value}")
        
        # Calculate derived parameters
        self._calculate_derived_parameters()
    
    def _calculate_derived_parameters(self):
        """Calculate derived parameters from basic settings"""
        # Calculate grid spacing
        self.Nx = int(self.Nx_before * self.sc_w_x)
        self.Ny = int(self.Ny_before * self.sc_w_y)

        self.dx = self.x_size / self.Nx
        self.dy = self.y_size / self.Ny
        self.z_size = self.Nz * self.dz

        self.pml_x_size = max(
            8,
            int(math.ceil(self.pml_x_size_before * self.dx_before / self.dx)),
        )
        self.pml_y_size = max(
            8,
            int(math.ceil(self.pml_y_size_before * self.dy_before / self.dy)),
        )

        self.tone_burst_freq = 1e6 * math.sqrt(self.sc_w_x)
        self.tone_burst_cycles = 4 * math.sqrt(self.sc_w_x)

        self.element_width = int(self.element_width_before)
        self.number_scan_lines = int(
            self.number_scan_lines_before
            * self.element_width_before
            * self.dy_before
            / (self.element_width * self.dy)
        )

        print("Calculated derived parameters:")
        print(
            f"  Scale: sc_w_x={self.sc_w_x}, sc_w_y={self.sc_w_y}"
        )
        print(
            f"  Grid: Nx={self.Nx}, Ny={self.Ny}, Nz={self.Nz}"
        )
        print(
            f"  Grid spacing: dx={self.dx*1000:.3f}mm, "
            f"dy={self.dy*1000:.3f}mm, dz={self.dz*1000:.3f}mm"
        )
        print(
            f"  Physical size: {self.x_size*1000:.1f}×"
            f"{self.y_size*1000:.1f}×{self.z_size*1000:.1f} mm"
        )
    
    def update_for_gpu(self):
        """Update configuration for GPU processing"""
        self.use_gpu = True
        self.data_cast = 'gpuArray-single'
        print("✓ Configuration updated for GPU processing")
    
    def update_for_cpu(self):
        """Update configuration for CPU processing"""
        self.use_gpu = False
        self.data_cast = 'single'
        print("✓ Configuration updated for CPU processing")
    
    def update_for_debug(self):
        """Update configuration for debug mode"""
        self.show_phantoms = True
        self.show_simulation_previews = True
        self.show_post_processing_previews = True
        self.enable_wandb = False
        self.use_gpu = False
        self.data_cast = 'single'
        print("✓ Configuration updated for debug mode")

class PostProcessingConfig(BaseOptions):
    """Post-processing specific configuration inheriting from BaseOptions"""
    
    def __init__(self, **kwargs):
        super().__init__()
        
        self.target_shape = (256, 256)
        self.show_preview = False
        self.file_suffix = ""
        self.gamma_correction = 0.5
        # Background blur applied to regions outside the tumor mask (0 = disabled)
        self.background_blur_sigma = 0.0
        # If you prefer to control kernel size directly, set background_blur_kernel to an odd integer.
        # When None the kernel size is derived from sigma.
        self.background_blur_kernel = None
        # Multiply the blurred background by this factor before compositing (1.0 = unchanged, <1.0 darker)
        self.background_brightness_scale = 1.0
        self.fund_filter_bw = 20 ###40 ###100
        self.harm_filter_bw = 8 ###15 ###30
        self.compression_ratio = 3
        self.semantic_label = 4
        
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown post-processing config parameter '{key}' = {value}")
        
        self._calculate_derived_parameters()

    def _calculate_derived_parameters(self):
        self.Nx = int(self.Nx_before * self.sc_w_x)
        self.Ny = int(self.Ny_before * self.sc_w_y)

        self.dx = self.x_size / self.Nx
        self.dy = self.y_size / self.Ny
        self.z_size = self.Nz * self.dz

        self.pml_x_size = max(
            8,
            int(math.ceil(self.pml_x_size_before * self.dx_before / self.dx)),
        )
        self.pml_y_size = max(
            8,
            int(math.ceil(self.pml_y_size_before * self.dy_before / self.dy)),
        )

        self.tone_burst_freq = 1e6 * math.sqrt(self.sc_w_x)
        self.tone_burst_cycles = 4 * math.sqrt(self.sc_w_x)

        self.element_width = int(self.element_width_before)
        self.number_scan_lines = int(
            self.number_scan_lines_before
            * self.element_width_before
            * self.dy_before
            / (self.element_width * self.dy)
        )

class DataPreparationConfig(BaseOptions):
    """Data preparation specific configuration inheriting from BaseOptions"""
    
    def __init__(self, **kwargs):
        # Initialize base options
        super().__init__()
        
        # Data preparation specific defaults
        self.preserve_aspect_ratio = True
        self.center_tumors = True
        
        # Override with any provided keyword arguments
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                print(f"Warning: Unknown data preparation config parameter '{key}' = {value}")

class QuickCommands:
    """Pre-configured command templates for common use cases"""
    
    @staticmethod
    def get_phantom_analysis_cmd(sample_name: str) -> List[str]:
        """Get command for phantom analysis"""
        return [
            'python', 'run_python_pipeline.py',
            '--sample-name', sample_name,
            '--stages', 'phantom-analysis'
        ]
    
    @staticmethod
    def get_single_sample_cmd(sample_name: str, gpu: bool = True) -> List[str]:
        """Get command for processing a single sample"""
        cmd = [
            'python', 'run_python_pipeline.py',
            '--sample-name', sample_name,
            '--auto-continue'
        ]
        if gpu:
            cmd.append('--gpu')
        return cmd
    
    @staticmethod
    def get_sample_range_cmd(start_id: int, end_id: int, gpu: bool = True) -> List[str]:
        """Get command for processing a sample range"""
        cmd = [
            'python', 'run_python_pipeline.py',
            '--sample-range', str(start_id), str(end_id),
            '--auto-continue'
        ]
        if gpu:
            cmd.append('--gpu')
        return cmd
    
    @staticmethod
    def get_first_n_samples_cmd(n: int = 100, gpu: bool = True) -> List[str]:
        """Get command for processing first N samples"""
        cmd = [
            'python', 'run_python_pipeline.py',
            '--end-sample-id', str(n),
            '--auto-continue'
        ]
        if gpu:
            cmd.append('--gpu')
        return cmd
    
    @staticmethod
    def get_multi_noise_cmd(end_id: int = 20, noise_levels: List[str] = None) -> List[str]:
        """Get command for multi-noise level processing"""
        if noise_levels is None:
            noise_levels = ['low', 'medium', 'high']
        
        cmd = [
            'python', 'run_python_pipeline.py',
            '--end-sample-id', str(end_id),
            '--noise-levels'] + noise_levels + [
            '--gpu',
            '--auto-continue'
        ]
        return cmd
    
    @staticmethod
    def get_debug_cmd(sample_name: str = "benign_10") -> List[str]:
        """Get command for debugging with visualizations"""
        return [
            'python', 'run_python_pipeline.py',
            '--sample-name', sample_name,
            '--visualize',
            '--stages', 'sim,post'
        ]

class DefaultPaths:
    """Centralized path management"""
    
    def __init__(self, base_data_dir: str = r"D:\CMME\1_data_generation\data\BUSI"):
        self.base_data_dir = base_data_dir
        self.busi_original = r"D:\CMME\2_denoising_ultrasound_experiments\data_busi\Dataset_BUSI_with_GT"
        
        # Derived paths
        self.processed_phantoms = os.path.join(base_data_dir, "processed_data_python_pipeline")
        self.simulation_results = os.path.join(base_data_dir, "python_simulation_results")
        self.final_training_data = os.path.join(base_data_dir, "final_training_data")
        
        # Output paths
        self.output_scans = os.path.join(self.final_training_data, "scans")
        self.output_labels = os.path.join(self.final_training_data, "labels")
    
    def ensure_all_paths_exist(self):
        """Create all necessary directories if they don't exist"""
        paths_to_create = [
            self.processed_phantoms,
            self.simulation_results, 
            self.final_training_data,
            self.output_scans,
            self.output_labels
        ]
        
        for path in paths_to_create:
            os.makedirs(path, exist_ok=True)
    
    def get_paths_dict(self) -> Dict[str, str]:
        """Get all paths as a dictionary"""
        return {
            'busi_original': self.busi_original,
            'processed_phantoms': self.processed_phantoms,
            'simulation_results': self.simulation_results,
            'final_training_data': self.final_training_data,
            'output_scans': self.output_scans,
            'output_labels': self.output_labels
        }

# =================================================================
# 🚀 QUICK ACCESS INSTANCES
# =================================================================

# Default configuration instance
default_options = BaseOptions()

# Default paths instance  
default_paths = DefaultPaths()

# Quick commands instance
quick_commands = QuickCommands()

# =================================================================
# 📋 PRESET CONFIGURATIONS
# =================================================================

PRESET_CONFIGS = {
    'quick_test': {
        'sample_names': ["benign_1", "benign_10", "malignant_1"],
        'use_gpu': True,
        'show_visualizations': False,
        'enable_wandb': False,
        'noise_levels': ['low']
    },
    
    'debug_single': {
        'sample_names': ["benign_10"],
        'use_gpu': False,
        'show_visualizations': True,
        'enable_wandb': False,
        'noise_levels': ['low']
    },
    
    'batch_production': {
        'end_sample_id': 100,
        'use_gpu': True,
        'show_visualizations': False,
        'enable_wandb': True,
        'auto_continue': True,
        'noise_levels': ['low']
    },
    
    'multi_noise_small': {
        'end_sample_id': 20,
        'use_gpu': True,
        'show_visualizations': False,
        'enable_wandb': True,
        'auto_continue': True,
        'noise_levels': ['low', 'medium', 'high']
    }
}

def get_preset_config(preset_name: str) -> Optional[Dict[str, Any]]:
    """Get a preset configuration by name"""
    return PRESET_CONFIGS.get(preset_name)

def list_available_presets() -> List[str]:
    """List all available preset configurations"""
    return list(PRESET_CONFIGS.keys())

def print_preset_info():
    """Print information about all available presets"""
    print("🎯 Available Preset Configurations:")
    print("=" * 50)
    
    for name, config in PRESET_CONFIGS.items():
        print(f"\n📋 {name.upper()}:")
        for key, value in config.items():
            print(f"  • {key}: {value}")
    
    print("\n" + "=" * 50)
    print("Usage: python -c \"from helpers.base_options import get_preset_config; print(get_preset_config('quick_test'))\"")

def create_config_from_args(args: argparse.Namespace, config_type: str = 'simulation') -> BaseOptions:
    """Create a configuration object from command line arguments"""
    
    if config_type == 'simulation':
        config = SimulationConfig()
    elif config_type == 'post_processing':
        config = PostProcessingConfig()
    elif config_type == 'data_preparation':
        config = DataPreparationConfig()
    else:
        config = BaseOptions()
    
    # Update from arguments
    config.update_from_args(args)
    
    return config
