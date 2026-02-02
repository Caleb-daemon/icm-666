#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Borealis University optimization problem definition for NSGA-II.

This module defines the optimization problem for Borealis University
using NSGA-II algorithm with pymoo library.
"""

import sys
import os
import numpy as np
from pymoo.core.problem import ElementwiseProblem

# Add the parent src directory to path for importing physical modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)

# Import physical calculation modules
try:
    from thermal_loads import simulate_2R2C, cooling_load_calculation, heating_load_calculation
    from metrics import calculate_discomfort_penalty
    
    # Import other necessary modules
    import pandas as pd
    from io_epw import read_epw
    from solar_geometry import calculate_solar_position, incidence_cos_window
    from irradiance_perez import tilted_irradiance
    from shading_geometry import shading_factor_vectorized, compute_shading_factors
    from visual_glare import ASE_1000
except ImportError:
    # Fallback for IDE linter
    simulate_2R2C = None
    cooling_load_calculation = None
    heating_load_calculation = None
    calculate_discomfort_penalty = None
    pd = None
    read_epw = None
    calculate_solar_position = None
    incidence_cos_window = None
    tilted_irradiance = None
    shading_factor_vectorized = None
    compute_shading_factors = None
    ASE_1000 = None
    # Re-raise the error at runtime
    raise


class BorealisOptimization(ElementwiseProblem):
    """
    Borealis University optimization problem for NSGA-II.
    
    This class defines the multi-objective optimization problem for Borealis University,
    considering energy consumption, lifecycle cost, and discomfort metric.
    """
    
    def __init__(self, epw_path: str, use_synthetic: bool = False):
        """
        Initialize the optimization problem.
        """
        # Define decision variables
        # x[0]: WWR (0.2-0.8)
        # x[1]: dN (0.0-1.0)
        # x[2]: dE (0.0-1.0)
        # x[3]: dS (0.0-1.0)
        # x[4]: dW (0.0-1.0)
        # x[5]: C_m (0.0-5e8) J/K
        super().__init__(
            n_var=6,
            n_obj=3,
            n_constr=0,
            xl=np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0]),
            xu=np.array([0.8, 1.0, 1.0, 1.0, 1.0, 5e8])
        )

        self.epw_path = epw_path
        self.use_synthetic = use_synthetic

        # Initialize building configuration
        self._initialize_building_config()
        
        # Load weather data
        self._load_weather_data()
        
        # Calculate solar position
        self._calculate_solar_position()
        
        # Calculate building geometry
        self._calculate_building_geometry()
    
    def _initialize_building_config(self):
        """
        Initialize building configuration parameters.
        """
        # Window properties - using triple glazing for better insulation
        self.u_window_W_m2K = 1.2  # W/m²·K (triple glazing)
        self.shgc = 0.5  # Dimensionless (slightly lower for triple glazing)
        
        # Building properties
        self.albedo = 0.2  # Dimensionless
        self.cooling_setpoint_C = 24.0  # °C
        self.heating_setpoint_C = 20.0  # °C
        self.internal_gains_W = 15000.0  # W (assuming ~10-15 W/m² for university building)
        
        # Thermal properties
        self.h_in = 3.0  # W/m²·K
        self.h_out = 25.0  # W/m²·K
        self.c_mass = 5.0e8  # J/K
        self.r_mass = 0.1  # m²·K/W
        self.r_wall = 0.3  # m²·K/W
        
        # Glare and shading properties
        self.ase_beam_threshold_W_m2 = 50.0  # W/m² (about 5000 Lux, strong glare threshold)
    
    def _load_weather_data(self):
        """
        Load weather data for Oslo.
        """
        if self.use_synthetic:
            # Use synthetic weather data for Oslo (colder climate)
            times = pd.date_range('2023-01-01', '2023-12-31 23:00', freq='h')
            df = pd.DataFrame(index=times)
            df['datetime'] = times
            # Oslo has lower solar irradiance
            df['DNI'] = 300.0 * np.sin(np.radians(df.index.hour * 15))
            df['DHI'] = 150.0
            df['GHI'] = df['DNI'] + df['DHI']
            # Colder climate: winter temperatures below 0C
            df['Temp'] = 10.0 + 15.0 * np.sin(np.radians(df.index.dayofyear * 360 / 365)) - 5.0
            # Clip negative values
            numeric_cols = ['DNI', 'DHI', 'GHI', 'Temp']
            for col in numeric_cols:
                df[col] = df[col].clip(lower=0)

            self.weather = df
            self.meta = {'latitude': 60.0, 'longitude': 10.75, 'altitude': 10.0}
            return

        # Read from EPW (preferred for paper chain)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(script_dir)
        epw_path = self.epw_path
        if not os.path.isabs(epw_path):
            epw_path = os.path.join(project_root, epw_path)

        epw_df, meta = read_epw(epw_path)
        if epw_df is None:
            raise FileNotFoundError(f"Failed to read EPW: {epw_path}")

        self.weather = epw_df
        self.meta = meta

    def _calculate_solar_position(self):
        """
        Calculate solar position for each time step.
        """
        self.solar_position = calculate_solar_position(
            times=self.weather.index,
            latitude=self.meta['latitude'],
            longitude=self.meta['longitude'],
            altitude=self.meta.get('altitude', 0.0)
        )
    
    def _calculate_building_geometry(self):
        """
        Calculate building geometry parameters.
        """
        # Building dimensions: 2 stories
        self.height = 7.0  # 2 stories * 3.5m
        self.L_long = 60.0  # N/S length
        self.L_short = 24.0  # E/W length
        
        # Calculate facade areas
        self.win_areas = {
            "N": self.L_long * self.height,  # 420.0
            "E": self.L_short * self.height,  # 168.0
            "S": self.L_long * self.height,  # 420.0
            "W": self.L_short * self.height   # 168.0
        }
        
        # Calculate total envelope area (Wall + Roof)
        self.total_envelope_area = 2 * (self.L_long + self.L_short) * self.height + (self.L_long * self.L_short)
    
    def _simulate_building(self, wwr, overhang_depths_m, c_mass):
        """
        Simulate building performance for given parameters.
        
        Args:
            wwr: Window-to-wall ratio.
            overhang_depths_m: Overhang depths for each facade.
            c_mass: Thermal mass capacitance (J/K).
            
        Returns:
            Dictionary with simulation results.
        """
        # Get weather data
        dni = self.weather['DNI'].values
        dhi = self.weather['DHI'].values
        ghi = self.weather['GHI'].values
        temp_air = self.weather['Temp'].values
        
        # Get solar position
        zen = self.solar_position['zenith'].values
        az = self.solar_position['azimuth'].values
        
        # Calculate solar altitude angle (alpha)
        alpha = 90.0 - zen
        alpha = np.maximum(0.0, alpha)  # Ensure non-negative
        
        # Facade configurations
        FACADE_AZIMUTH_DEG = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
        
        # Initialize variables
        Qsolar_total_W = 0.0
        ASE_hours_total = 0
        Q_solar_hourly_sum = np.zeros(len(self.weather))
        Q_solar_a = np.zeros(len(self.weather))
        Q_solar_m = np.zeros(len(self.weather))
        
        # Building height for penetration depth calculation
        H = self.height  # Building height
        D_mass = 0.5  # Thermal mass position (m)
        
        # Calculate occupancy schedule (simplified)
        # Assume occupancy from 8:00 to 18:00 on weekdays
        occupancy = np.zeros(len(self.weather))
        for i, dt in enumerate(self.weather.index):
            hour = dt.hour
            day_of_week = dt.dayofweek
            if 8 <= hour <= 18 and day_of_week < 5:
                occupancy[i] = 1.0
        
        # Calculate heating season (simplified)
        # Assume heating season from October to March
        heating_season = np.zeros(len(self.weather))
        for i, dt in enumerate(self.weather.index):
            month = dt.month
            if month in [10, 11, 12, 1, 2, 3]:
                heating_season[i] = 1.0
        
        # Calculate for each facade
        for f in ["N", "E", "S", "W"]:
            surf_tilt = 90.0
            surf_az = FACADE_AZIMUTH_DEG[f]
            area = self.win_areas[f] * wwr
            d_overhang = float(overhang_depths_m.get(f, 0.0))
            
            # Calculate shading factor
            S = shading_factor_vectorized(
                solar_zenith_deg=zen,
                solar_azimuth_deg=az,
                facade=f,
                overhang_depth_m=d_overhang
            )
            
            # Calculate beam irradiance on tilted surface
            cos_theta = incidence_cos_window(az, zen, surf_az, surf_tilt)
            beam_poa = dni * cos_theta
            
            # Calculate diffuse and reflected irradiance
            _, diffuse_tilted, reflected_tilted = tilted_irradiance(
                surface_tilt=surf_tilt,
                surface_azimuth=surf_az,
                solar_zenith=zen,
                solar_azimuth=az,
                dni=dni,
                dhi=dhi,
                ghi=ghi,
                albedo=self.albedo
            )
            
            # Apply occupancy-driven control logic
            S_b = np.zeros(len(self.weather))
            for i in range(len(self.weather)):
                if occupancy[i] == 0 and heating_season[i] == 1:
                    # Non-occupancy period during heating season: fully open shading
                    S_b[i] = 1.0
                else:
                    # Occupancy period or non-heating season: use calculated shading factor
                    S_b[i] = S[i]
                    # If direct irradiance > 1000 lux (ASE trigger), reduce S_b to prevent glare
                    if occupancy[i] == 1 and beam_poa[i] > 100.0:  # ~1000 lux
                        # Check if it's extremely cold season
                        month = self.weather.index[i].month
                        if month in [12, 1, 2]:
                            # Allow more direct sunlight in extremely cold season to reduce heating energy
                            S_b[i] = max(0.5, S_b[i] * 0.75)  # Less reduction in cold season
                        else:
                            S_b[i] = max(0.3, S_b[i] * 0.5)  # Normal reduction
            
            S_d = 0.5  # Default SVF value
            S_r = 1.0  # Default for reflected component
            
            beam_poa_shaded = S_b * beam_poa
            diffuse_tilted_shaded = S_d * diffuse_tilted
            reflected_tilted_shaded = S_r * reflected_tilted
            
            # Calculate total irradiance
            total_irradiance = beam_poa_shaded + diffuse_tilted_shaded + reflected_tilted_shaded
            
            # Calculate angle-dependent SHGC
            shgc_modifier = np.maximum(0.0, 1.0 - np.power(1.0 - cos_theta, 3.5))
            dynamic_shgc = self.shgc * shgc_modifier
            
            # Calculate solar heat gain
            Qsolar_f = dynamic_shgc * area * total_irradiance
            Qsolar_total_W += np.sum(Qsolar_f)
            Q_solar_hourly_sum += Qsolar_f
            
            # Calculate ASE
            _, F_sunlit = compute_shading_factors(
                solar_zenith=zen,
                solar_azimuth=az,
                window_azimuth=surf_az,
                overhang_depth=d_overhang
            )
            beam_poa_shaded = S_b * beam_poa
            ase_hours = ASE_1000(beam_poa_shaded, F_sunlit, threshold=self.ase_beam_threshold_W_m2)
            ASE_hours_total += ase_hours
        
        # Calculate solar penetration depth and split solar gains
        for i in range(len(self.weather)):
            if alpha[i] > 0:
                L_pen = H / np.tan(np.radians(alpha[i]))
            else:
                L_pen = 0.0
            
            # Apply solar distribution logic using continuous function
            # η_mass(t) = clip(L_pen(t) / (2 * D_mass), 0, 1)
            eta_mass = np.clip(L_pen / (2 * D_mass), 0, 1)
            # Allocate solar gains based on eta_mass
            Q_solar_m[i] = eta_mass * Q_solar_hourly_sum[i]
            Q_solar_a[i] = (1 - eta_mass) * Q_solar_hourly_sum[i]
        
        # Run 2R2C model
        temp_air_rc, temp_mass, q_mass = simulate_2R2C(
            temp_outdoor=temp_air,
            q_solar_a=Q_solar_a,
            q_solar_m=Q_solar_m,
            q_internal=np.full(len(self.weather), self.internal_gains_W),
            h_in=self.h_in,
            h_out=self.h_out,
            c_air=1.0e8,  # Air thermal capacitance
            c_mass=c_mass,
            r_env=0.1,  # Envelope thermal resistance
            r_am=0.05,  # Air-mass thermal resistance
            area=self.total_envelope_area
        )
        
        # Calculate cooling load
        cooling_load_hourly_W = cooling_load_calculation(
            temp_air_rc,
            self.cooling_setpoint_C,
            area=self.total_envelope_area,
            h_in=self.h_in
        )
        
        # Calculate heating load
        heating_load_hourly_W = heating_load_calculation(
            temp_air_rc,
            self.heating_setpoint_C,
            area=self.total_envelope_area,
            h_in=self.h_in
        )
        
        # Convert to kWh
        cooling_cop = 3.0
        heating_cop = 4.0
        annual_cooling_kwh = np.sum(cooling_load_hourly_W) / 1000.0 / cooling_cop
        annual_heating_kwh = np.sum(heating_load_hourly_W) / 1000.0 / heating_cop
        
        # Calculate discomfort penalty
        discomfort_penalty = calculate_discomfort_penalty(ASE_hours_total)
        
        # Calculate costs
        # Simple cost model with glass pricing tiers
        installation_factor = 0.20  # Installation cost as 20% of material cost
        maintenance_rate = 0.02  # Annual maintenance cost as 2% of CAPEX
        
        # Glass pricing
        price_glass_high = 450.0  # High performance glass ($/m²)
        wall_cost_per_m2 = 150.0  # Wall cost ($/m²)
        
        # Calculate window and wall areas
        window_area = sum(self.win_areas.values()) * wwr
        wall_area = sum(self.win_areas.values()) * (1 - wwr)
        overhang_length = sum(self.win_areas.values()) * 0.5
        
        # Calculate overhang cost with tiered pricing
        overhang_cost = 0.0
        for depth in overhang_depths_m.values():
            if depth < 0.2:
                # Simple installation for shallow overhangs
                overhang_cost += depth * overhang_length * 100.0  # Lower unit price
            else:
                overhang_cost += depth * overhang_length * 150.0  # Standard unit price
        
        # Calculate costs
        window_cost = window_area * price_glass_high
        wall_cost = wall_area * wall_cost_per_m2
        material_cost = window_cost + overhang_cost + wall_cost
        installation_cost = material_cost * installation_factor
        total_capex = material_cost + installation_cost
        
        # Apply government subsidy for thermal mass active materials
        if c_mass > 1e8:  # Considered as thermal mass active material
            total_capex *= 0.85  # 15% CAPEX reduction
        
        # Energy costs with differential pricing for heating peak loads
        electricity_cost_per_kwh = 0.15
        
        # Calculate heating cost with peak load differential pricing
        # Identify peak heating hours (coldest months, highest demand)
        peak_heating_hours = 0
        for i in range(len(self.weather)):
            month = self.weather.index[i].month
            if month in [12, 1, 2] and heating_load_hourly_W[i] > 0:
                peak_heating_hours += 1
        
        # Differential pricing for peak vs off-peak heating
        peak_heating_kwh = annual_heating_kwh * (peak_heating_hours / len(self.weather))
        off_peak_heating_kwh = annual_heating_kwh - peak_heating_kwh
        
        heating_peak_cost_per_kwh = 0.27  # Higher rate for peak heating
        heating_off_peak_cost_per_kwh = 0.20  # Lower rate for off-peak heating
        
        annual_heating_cost = (peak_heating_kwh * heating_peak_cost_per_kwh) + (off_peak_heating_kwh * heating_off_peak_cost_per_kwh)
        annual_cooling_cost = annual_cooling_kwh * electricity_cost_per_kwh
        annual_energy_cost = annual_cooling_cost + annual_heating_cost
        
        # Calculate peak delay hours
        # Group data by day and calculate daily peak delays
        import pandas as pd
        
        # Create a DataFrame with hourly data
        hourly_df = pd.DataFrame({
            'date': self.weather.index.date,
            'hour': self.weather.index.hour,
            'T_outdoor': temp_air,
            'T_air': temp_air_rc
        })
        
        # Calculate daily peak delay
        peak_delay_hours = 0.0
        day_count = 0
        
        for date in hourly_df['date'].unique():
            day_data = hourly_df[hourly_df['date'] == date]
            if len(day_data) > 0:
                # Find peak hours
                peak_outdoor_idx = day_data['T_outdoor'].idxmax()
                peak_air_idx = day_data['T_air'].idxmax()
                
                if peak_outdoor_idx is not None and peak_air_idx is not None:
                    peak_outdoor_hour = day_data.loc[peak_outdoor_idx, 'hour']
                    peak_air_hour = day_data.loc[peak_air_idx, 'hour']
                    
                    # Calculate delay (wrap around if necessary)
                    delay = (peak_air_hour - peak_outdoor_hour) % 24
                    peak_delay_hours += delay
                    day_count += 1
        
        # Calculate average delay
        if day_count > 0:
            peak_delay_hours = peak_delay_hours / day_count
        else:
            peak_delay_hours = 0.0
        
        # Lifecycle cost (20 years)
        annual_maintenance_cost = total_capex * maintenance_rate
        total_lifecycle_cost = total_capex + 20 * (annual_energy_cost + annual_maintenance_cost)
        
        return {
            'annual_heating_kwh': annual_heating_kwh,
            'annual_cooling_kwh': annual_cooling_kwh,
            'total_energy_kwh': annual_heating_kwh + annual_cooling_kwh,
            'ase_hours': ASE_hours_total,
            'total_capex': total_capex,
            'total_lifecycle_cost': total_lifecycle_cost,
            'discomfort_penalty': discomfort_penalty,
            'temp_mass': temp_mass,
            'q_mass': q_mass,
            'peak_delay_hours': peak_delay_hours
        }
    
    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluate the objectives for given decision variables.
        
        Args:
            x: Decision variables [wwr, dN, dE, dS, dW, c_mass].
            out: Dictionary to store objectives.
        """
        wwr = x[0]
        overhang_depths_m = {
            "N": x[1],
            "E": x[2],
            "S": x[3],
            "W": x[4]
        }
        c_mass = x[5]
        
        # Run simulation
        results = self._simulate_building(wwr, overhang_depths_m, c_mass)
        
        # Objectives to minimize
        f1 = results['total_energy_kwh']  # Annual energy consumption (kWh)
        f2 = results['total_lifecycle_cost']  # Lifecycle cost ($)
        f3 = results['ase_hours']  # Discomfort metric (ASE hours)
        
        out['F'] = np.array([f1, f2, f3])
