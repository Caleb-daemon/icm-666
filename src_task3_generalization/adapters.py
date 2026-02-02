#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapter layer for calling functions from src directory."""

import os
import sys
from typing import Optional, Dict, Any, Callable

# Add the parent src directory to path for importing physical modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)

# Import modules with fallback for missing functions
class Adapter:
    """Adapter class for calling functions from src directory."""
    
    def __init__(self):
        """Initialize the adapter."""
        self.imports = {}
        self._try_imports()
    
    def _try_import(self, module_name: str, alias: Optional[str] = None) -> bool:
        """Try to import a module."""
        try:
            module = __import__(module_name)
            self.imports[alias or module_name] = module
            return True
        except ImportError:
            return False
    
    def _try_imports(self):
        """Try to import all required modules."""
        # Core modules
        self._try_import('io_epw', 'io_epw')
        self._try_import('solar_geometry', 'solar')
        self._try_import('irradiance_perez', 'perez')
        self._try_import('shading_geometry', 'shading')
        self._try_import('fenestration', 'fenestration')
        self._try_import('thermal_loads', 'thermal')
        self._try_import('visual_glare', 'glare')
        self._try_import('metrics', 'metrics')
        self._try_import('economics', 'economics')
        self._try_import('optimize', 'optimize')
        self._try_import('run_analysis', 'run_analysis')
    
    def has_function(self, module: str, func_name: str) -> bool:
        """Check if a function exists in a module."""
        if module not in self.imports:
            return False
        return hasattr(self.imports[module], func_name)
    
    def get_function(self, module: str, func_name: str, default: Optional[Callable] = None) -> Optional[Callable]:
        """Get a function from a module."""
        if self.has_function(module, func_name):
            return getattr(self.imports[module], func_name)
        return default
    
    def read_epw(self, epw_path: str) -> Optional[Any]:
        """Read EPW file."""
        if self.has_function('io_epw', 'read_epw'):
            return self.get_function('io_epw', 'read_epw')(epw_path)
        return None
    
    def calculate_solar_position(self, times, latitude, longitude, altitude=0.0) -> Optional[Any]:
        """Calculate solar position."""
        if self.has_function('solar', 'calculate_solar_position'):
            return self.get_function('solar', 'calculate_solar_position')(
                times=times, latitude=latitude, longitude=longitude, altitude=altitude
            )
        return None
    
    def tilted_irradiance(self, surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, 
                         dni, dhi, ghi, albedo) -> Optional[Any]:
        """Calculate tilted irradiance."""
        if self.has_function('perez', 'tilted_irradiance'):
            return self.get_function('perez', 'tilted_irradiance')(
                surface_tilt=surface_tilt, surface_azimuth=surface_azimuth,
                solar_zenith=solar_zenith, solar_azimuth=solar_azimuth,
                dni=dni, dhi=dhi, ghi=ghi, albedo=albedo
            )
        return None
    
    def shading_factor_vectorized(self, solar_zenith_deg, solar_azimuth_deg, facade, overhang_depth_m) -> Optional[Any]:
        """Calculate shading factor vectorized."""
        if self.has_function('shading', 'shading_factor_vectorized'):
            return self.get_function('shading', 'shading_factor_vectorized')(
                solar_zenith_deg=solar_zenith_deg, solar_azimuth_deg=solar_azimuth_deg,
                facade=facade, overhang_depth_m=overhang_depth_m
            )
        return None
    
    def simulate_2R1C(self, temp_outdoor, q_solar_a, q_solar_m, q_internal, h_in, h_out, 
                     c_mass, r_mass, r_wall, area, **kwargs) -> Optional[Any]:
        """Simulate 2R1C model."""
        if self.has_function('thermal', 'simulate_2R1C'):
            return self.get_function('thermal', 'simulate_2R1C')(
                temp_outdoor=temp_outdoor, q_solar_a=q_solar_a, q_solar_m=q_solar_m,
                q_internal=q_internal, h_in=h_in, h_out=h_out,
                c_mass=c_mass, r_mass=r_mass, r_wall=r_wall, area=area, **kwargs
            )
        return None
    
    def ASE_1000(self, beam_poa_shaded, F_sunlit, threshold) -> Optional[Any]:
        """Calculate ASE 1000."""
        if self.has_function('glare', 'ASE_1000'):
            return self.get_function('glare', 'ASE_1000')(
                beam_poa_shaded=beam_poa_shaded, F_sunlit=F_sunlit, threshold=threshold
            )
        return None
    
    def cooling_load_calculation(self, temp_air, temp_setpoint, area, h_in) -> Optional[Any]:
        """Calculate cooling load."""
        if self.has_function('thermal', 'cooling_load_calculation'):
            return self.get_function('thermal', 'cooling_load_calculation')(
                temp_air=temp_air, temp_setpoint=temp_setpoint, area=area, h_in=h_in
            )
        return None
    
    def heating_load_calculation(self, temp_air, temp_setpoint, area, h_in) -> Optional[Any]:
        """Calculate heating load."""
        if self.has_function('thermal', 'heating_load_calculation'):
            return self.get_function('thermal', 'heating_load_calculation')(
                temp_air=temp_air, temp_setpoint=temp_setpoint, area=area, h_in=h_in
            )
        return None
    
    def calculate_discomfort_penalty(self, ase_hours) -> Optional[Any]:
        """Calculate discomfort penalty."""
        if self.has_function('metrics', 'calculate_discomfort_penalty'):
            return self.get_function('metrics', 'calculate_discomfort_penalty')(ase_hours)
        return None
    
    def run_analysis(self, epw_path: str, params: Dict[str, Any]) -> Optional[Any]:
        """Run analysis using run_analysis module if available."""
        if self.has_function('run_analysis', 'run_analysis'):
            return self.get_function('run_analysis', 'run_analysis')(epw_path, params)
        return None

# Create a global adapter instance
adapter = Adapter()

# Wrapper functions
def read_epw(epw_path: str) -> Optional[Any]:
    """Wrapper for io_epw.read_epw."""
    return adapter.read_epw(epw_path)

def calculate_solar_position(times, latitude, longitude, altitude=0.0) -> Optional[Any]:
    """Wrapper for solar_geometry.calculate_solar_position."""
    return adapter.calculate_solar_position(times, latitude, longitude, altitude)

def tilted_irradiance(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, 
                     dni, dhi, ghi, albedo) -> Optional[Any]:
    """Wrapper for irradiance_perez.tilted_irradiance."""
    return adapter.tilted_irradiance(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth, dni, dhi, ghi, albedo
    )

def shading_factor_vectorized(solar_zenith_deg, solar_azimuth_deg, facade, overhang_depth_m) -> Optional[Any]:
    """Wrapper for shading_geometry.shading_factor_vectorized."""
    return adapter.shading_factor_vectorized(
        solar_zenith_deg, solar_azimuth_deg, facade, overhang_depth_m
    )

def simulate_2R1C(temp_outdoor, q_solar_a, q_solar_m, q_internal, h_in, h_out, 
                 c_mass, r_mass, r_wall, area, **kwargs) -> Optional[Any]:
    """Wrapper for thermal_loads.simulate_2R1C."""
    return adapter.simulate_2R1C(
        temp_outdoor, q_solar_a, q_solar_m, q_internal, h_in, h_out, 
        c_mass, r_mass, r_wall, area, **kwargs
    )

def ASE_1000(beam_poa_shaded, F_sunlit, threshold) -> Optional[Any]:
    """Wrapper for visual_glare.ASE_1000."""
    return adapter.ASE_1000(beam_poa_shaded, F_sunlit, threshold)

def cooling_load_calculation(temp_air, temp_setpoint, area, h_in) -> Optional[Any]:
    """Wrapper for thermal_loads.cooling_load_calculation."""
    return adapter.cooling_load_calculation(temp_air, temp_setpoint, area, h_in)

def heating_load_calculation(temp_air, temp_setpoint, area, h_in) -> Optional[Any]:
    """Wrapper for thermal_loads.heating_load_calculation."""
    return adapter.heating_load_calculation(temp_air, temp_setpoint, area, h_in)

def calculate_discomfort_penalty(ase_hours) -> Optional[Any]:
    """Wrapper for metrics.calculate_discomfort_penalty."""
    return adapter.calculate_discomfort_penalty(ase_hours)

def run_analysis(epw_path: str, params: Dict[str, Any]) -> Optional[Any]:
    """Wrapper for run_analysis.run_analysis."""
    return adapter.run_analysis(epw_path, params)
