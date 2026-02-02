#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task 2 runner for the Borealis University building retrofit (Oslo).

This module implements the main runner for Task 2 of the 2026 ICM Contest (Problem E).
It loads EPW data for Oslo, defines the Borealis University building,
and runs the simulation and optimization workflow focused on heating.
"""

import os
import sys

# Add the parent src directory to path for importing physical modules
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(os.path.dirname(script_dir), 'src')
sys.path.append(src_dir)

import argparse
import pandas as pd
import numpy as np

# Import modules
try:
    from io_epw import read_epw
    from solar_geometry import calculate_solar_position, incidence_cos_window
    from irradiance_perez import tilted_irradiance
    from shading_geometry import compute_shading_factors, shading_factor_vectorized
    from thermal_loads import simulate_2R1C, cooling_load_calculation, heating_load_calculation
    from visual_glare import ASE_1000
    from metrics import calculate_energy_savings, calculate_discomfort_penalty
    from economics import ROI_20
    from optimize import OptimizationConfig, run_optimization
except ImportError:
    # Fallback for IDE linter
    read_epw = None
    calculate_solar_position = None
    incidence_cos_window = None
    tilted_irradiance = None
    compute_shading_factors = None
    shading_factor_vectorized = None
    simulate_2R1C = None
    cooling_load_calculation = None
    heating_load_calculation = None
    ASE_1000 = None
    calculate_energy_savings = None
    calculate_discomfort_penalty = None
    ROI_20 = None
    OptimizationConfig = None
    run_optimization = None
    # Re-raise the error at runtime
    raise


class BuildingConfig:
    """
    Configuration class for building parameters.
    
    Attributes:
        u_window_W_m2K: Window thermal transmittance (W/m²·K).
        shgc: Solar Heat Gain Coefficient (dimensionless).
        albedo: Ground albedo (dimensionless).
        cooling_setpoint_C: Cooling setpoint temperature (°C).
        heating_setpoint_C: Heating setpoint temperature (°C).
        internal_gains_W: Internal heat gains (W).
        h_in: Indoor convective heat transfer coefficient (W/m²·K).
        h_out: Outdoor convective heat transfer coefficient (W/m²·K).
        c_mass: Thermal mass capacitance (J/K).
        r_mass: Thermal resistance between air and mass (m²·K/W).
        r_wall: Wall thermal resistance (m²·K/W).
        window_height: Window height (m).
        window_width: Window width (m).
        ase_beam_threshold_W_m2: Threshold for ASE calculation (W/m²).
        svf_min: Minimum Sky View Factor (dimensionless).
        svf_slope_per_m: SVF slope per meter of overhang (dimensionless/m).
        svf_retrofit: SVF for retrofit (dimensionless).
    """
    def __init__(self):
        # Window properties - using triple glazing for better insulation
        self.u_window_W_m2K = 1.2  # W/m²·K (triple glazing)
        self.shgc = 0.5  # Dimensionless (slightly lower for triple glazing)
        
        # Building properties
        self.albedo = 0.2  # Dimensionless
        self.cooling_setpoint_C = 24.0  # °C
        self.heating_setpoint_C = 20.0  # °C
        self.internal_gains_W = 15000.0  # W (假设约 1000-1500m² 建筑，10-15 W/m²)
        
        # Thermal properties
        self.h_in = 3.0  # W/m²·K
        self.h_out = 25.0  # W/m²·K
        self.c_mass = 5.0e8  # J/K
        self.r_mass = 0.1  # m²·K/W
        self.r_wall = 0.3  # m²·K/W
        
        # Window geometry
        self.window_height = 2.1  # m
        self.window_width = 1.8  # m
        
        # Glare and shading properties
        self.ase_beam_threshold_W_m2 = 50.0  # W/m² (约 5000 Lux，强眩光阈值)
        self.svf_min = 0.1  # Dimensionless
        self.svf_slope_per_m = 0.4  # Dimensionless/m
        self.svf_retrofit = 0.5  # Dimensionless


class Building:
    """
    Building class with geometry and properties.
    
    Attributes:
        config: Building configuration.
        geometry: Building geometry object.
    """
    def __init__(self, config: BuildingConfig):
        self.config = config
        self.geometry = BuildingGeometry()


class BuildingGeometry:
    """
    Building geometry class for shading calculations.
    """
    def shading_factor_vectorized(self, solar_zenith_deg, solar_azimuth_deg, facade, overhang_depth_m):
        """
        Calculate shading factor for a facade.
        
        Args:
            solar_zenith_deg: Solar zenith angle (degrees).
            solar_azimuth_deg: Solar azimuth angle (degrees).
            facade: Facade direction ('N', 'E', 'S', 'W').
            overhang_depth_m: Overhang depth (meters).
            
        Returns:
            Shading factor (0 = fully shaded, 1 = fully sunlit).
        """
        return shading_factor_vectorized(
            solar_zenith_deg, solar_azimuth_deg, facade, overhang_depth_m
        )


class SimulationEngine:
    """
    Simulation engine for building performance simulation.
    
    Attributes:
        epw_path: Path to EPW file.
        use_synthetic: Whether to use synthetic weather data.
        building_cfg: Building configuration.
        weather: Weather data.
        meta: Weather metadata.
        solar_position: Solar position data.
        dni_extra: Extraterrestrial DNI.
        airmass_rel: Relative air mass.
    """
    def __init__(self, epw_path: str, use_synthetic: bool, building_cfg: BuildingConfig):
        self.epw_path = epw_path
        self.use_synthetic = use_synthetic
        self.building_cfg = building_cfg
        self.building = Building(building_cfg)
        
        # Load weather data
        self.weather, self.meta = self._load_weather()
        
        # Calculate solar position
        self.solar_position = self._calculate_solar_position()
        
        # Calculate additional solar parameters
        self.dni_extra = self._calculate_dni_extra()
        self.airmass_rel = self._calculate_airmass()
    
    def _load_weather(self):
        """
        Load weather data from EPW file.
        
        Returns:
            Weather data and metadata.
        """
        if self.use_synthetic:
            # Generate synthetic weather data for Oslo (colder climate)
            times = pd.date_range('2023-01-01', '2023-12-31 23:00', freq='h')
            df = pd.DataFrame(index=times)
            df['datetime'] = times
            # Oslo has lower solar irradiance
            df['DNI'] = 300.0 * np.sin(np.radians(df.index.hour * 15))
            df['DHI'] = 150.0
            df['GHI'] = df['DNI'] + df['DHI']
            # Colder climate: winter temperatures below 0°C
            df['Temp'] = 10.0 + 15.0 * np.sin(np.radians(df.index.dayofyear * 360 / 365)) - 5.0
            # 只对数值列进行裁剪，避免对 datetime 列操作
            numeric_cols = ['DNI', 'DHI', 'GHI', 'Temp']
            for col in numeric_cols:
                df[col] = df[col].clip(lower=0)
            meta = {'latitude': 60.0, 'longitude': 10.75, 'altitude': 10.0}
            return df, meta
        else:
            # Load from EPW file
            # 使用相对于脚本的路径，确保无论从哪个目录运行都能找到文件
            import os
            script_dir = os.path.dirname(os.path.abspath(__file__))
            epw_path = os.path.join(script_dir, self.epw_path)
            return read_epw(epw_path)
    
    def _calculate_solar_position(self):
        """
        Calculate solar position.
        
        Returns:
            Solar position data.
        """
        return calculate_solar_position(
            times=self.weather.index,
            latitude=self.meta['latitude'],
            longitude=self.meta['longitude'],
            altitude=self.meta.get('altitude', 0.0)
        )
    
    def _calculate_dni_extra(self):
        """
        Calculate extraterrestrial DNI.
        
        Returns:
            Extraterrestrial DNI.
        """
        # Fix function name: 'extraradiation' is now 'get_extra_radiation'
        from pvlib.irradiance import get_extra_radiation
        return get_extra_radiation(self.weather.index)
    
    def _calculate_airmass(self):
        """
        Calculate relative air mass.
        
        Returns:
            Relative air mass.
        """
        # Fix import: 'relative_airmass' is now in 'pvlib.atmosphere' or may have been renamed
        try:
            from pvlib.atmosphere import relative_airmass
        except ImportError:
            # Fallback implementation if function is not available
            def relative_airmass(zenith):
                """Simple relative air mass calculation."""
                zenith_rad = np.radians(zenith)
                return 1.0 / np.cos(zenith_rad)
        return relative_airmass(self.solar_position['zenith'])
    
    def set_baseline(self, baseline_wwr=0.4):
        """
        Set baseline building parameters.
        
        Args:
            baseline_wwr: Baseline window-to-wall ratio.
            
        Returns:
            Baseline simulation result.
        """
        result = self.simulate(
            wwr=baseline_wwr,
            overhang_depths_m={"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0},
            label="baseline"
        )
        # 保存基准能耗
        self.baseline_heating_kwh = result['annual_heating_kwh']
        self.baseline_cooling_kwh = result['annual_cooling_kwh']
        return result
    
    def simulate(self, wwr, overhang_depths_m, label="simulation", detailed=False, override_shgc=None, override_u_val=None):
        """
        Run building performance simulation.
        
        Args:
            wwr: Window-to-wall ratio.
            overhang_depths_m: Overhang depths for each facade (meters).
            label: Simulation label.
            detailed: If True, return detailed hourly data.
            override_shgc: Optional override for SHGC value.
            override_u_val: Optional override for window U-value.
            
        Returns:
            Simulation result dictionary. If detailed=True, also returns hourly data.
        """
        # 使用覆盖参数或默认参数
        shgc = override_shgc if override_shgc is not None else self.building_cfg.shgc
        u_window = override_u_val if override_u_val is not None else self.building_cfg.u_window_W_m2K
        # Get weather data
        dni = self.weather['DNI'].values
        dhi = self.weather['DHI'].values
        ghi = self.weather['GHI'].values
        temp_air = self.weather['Temp'].values
        
        # Get solar position
        zen = self.solar_position['zenith'].values
        az = self.solar_position['azimuth'].values
        
        # Facade configurations
        FACADE_AZIMUTH_DEG = {"N": 0.0, "E": 90.0, "S": 180.0, "W": 270.0}
        
        # 使用真实的建筑尺寸计算墙面面积
        height = 7.0  # 2 stories * 3.5m
        L_long = 60.0 # N/S
        L_short = 24.0 # E/W
        
        # 用真实的总立面面积（Gross Wall Area）替换硬编码的 100.0
        win_areas = {
            "N": L_long * height,  # 420.0
            "E": L_short * height, # 168.0
            "S": L_long * height,  # 420.0
            "W": L_short * height  # 168.0
        }
        
        # Initialize variables
        Qsolar_total_W = 0.0
        ASE_hours_total = 0
        Q_solar_hourly_sum = np.zeros(len(self.weather)) # 新增：用于存储逐时总得热
        
        # Initialize hourly data storage if detailed
        if detailed:
            hourly_data = []
        
        # Calculate for each facade
        for f in ["N", "E", "S", "W"]:
            surf_tilt = 90.0
            surf_az = FACADE_AZIMUTH_DEG[f]
            area = win_areas[f] * wwr
            d_overhang = float(overhang_depths_m.get(f, 0.0))
            
            # Calculate shading factor
            S = self.building.geometry.shading_factor_vectorized(
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
                albedo=self.building_cfg.albedo
            )
            
            # 遮阳系数应用：
            # S_b (直射遮阳系数) = S
            # S_d (散射遮阳系数) = 常数 SVF (天空视角系数)
            # S_r (反射遮阳系数) = 常数
            S_b = S
            S_d = 0.5  # 默认 SVF 值
            S_r = 1.0  # 反射分量默认不遮阳
            
            # 应用遮阳系数
            # S_b 是 Direct solar blocking factor (0 = fully shaded, 1 = fully sunlit)
            # 所以不需要 (1.0 - S_b)，直接使用 S_b
            beam_poa_shaded = S_b * beam_poa
            diffuse_tilted_shaded = S_d * diffuse_tilted
            reflected_tilted_shaded = S_r * reflected_tilted
            
            # Calculate total irradiance with proper shading coefficients
            total_irradiance = beam_poa_shaded + diffuse_tilted_shaded + reflected_tilted_shaded
            
            # 实现角度相关的 SHGC
            # 使用简单的经验公式：当入射角增大(cos_theta减小)时，SHGC衰减
            # 修正系数 modifier = 1.0 - (1.0 - cos_theta)^3.5
            shgc_modifier = np.maximum(0.0, 1.0 - np.power(1.0 - cos_theta, 3.5))
            dynamic_shgc = shgc * shgc_modifier
            
            # Calculate solar heat gain
            Qsolar_f = dynamic_shgc * area * total_irradiance
            Qsolar_total_W += np.sum(Qsolar_f)
            Q_solar_hourly_sum += Qsolar_f       # 新增：累加逐时曲线
            
            # Calculate ASE
            _, F_sunlit = compute_shading_factors(
                solar_zenith=zen,
                solar_azimuth=az,
                window_azimuth=surf_az,
                overhang_depth=d_overhang
            )
            # 使用遮阳后的直射辐射计算 ASE，这样添加遮阳板才能真正减少眩光小时数
            # S_b 是 Direct solar blocking factor (0 = fully shaded, 1 = fully sunlit)
            beam_poa_shaded = S_b * beam_poa
            ase_hours = ASE_1000(beam_poa_shaded, F_sunlit, threshold=self.building_cfg.ase_beam_threshold_W_m2)
            ASE_hours_total += ase_hours
            
            # Store hourly data if detailed
            if detailed:
                for i in range(len(self.weather)):
                    hourly_data.append({
                        'time': self.weather.index[i],
                        'T_out': temp_air[i],
                        'T_in': temp_air[i] + 2.0,  # Simplified indoor temperature
                        'Q_solar': Qsolar_f[i],
                        'Cooling_Load': max(0, Qsolar_f[i]),
                        'S_b': S_b[i],
                        'facade': f
                    })
        
        # Calculate annual cooling and heating energy using RC model
        # 估算外围护结构总面积 (Gross Wall + Roof) 用于 RC 参数
        total_envelope_area = 2 * (L_long + L_short) * height + (L_long * L_short) # Wall + Roof
        
        # 运行 RC 模型
        temp_air_rc, temp_mass = simulate_2R1C(
            temp_outdoor=temp_air, # 来自 weather['Temp']
            q_solar_a=Q_solar_hourly_sum,  # 对于 task2，全部太阳得热都给空气
            q_solar_m=np.zeros(len(self.weather)),  # 对于 task2，热质量不直接吸收太阳得热
            q_internal=np.full(len(self.weather), self.building_cfg.internal_gains_W),
            h_in=self.building_cfg.h_in,
            h_out=self.building_cfg.h_out,
            c_mass=self.building_cfg.c_mass,
            r_mass=self.building_cfg.r_mass,
            r_wall=self.building_cfg.r_wall,
            area=total_envelope_area # 传入估算的总面积
        )
        
        # 计算冷负荷 (Watts)
        cooling_load_hourly_W = cooling_load_calculation(
            temp_air_rc, 
            self.building_cfg.cooling_setpoint_C,
            area=total_envelope_area,
            h_in=self.building_cfg.h_in
        )
        
        # 计算热负荷 (Watts)
        heating_load_hourly_W = heating_load_calculation(
            temp_air_rc, 
            self.building_cfg.heating_setpoint_C,
            area=total_envelope_area,
            h_in=self.building_cfg.h_in
        )
        
        # 转换成年制冷量和制热量 (kWh)
        # 假设 COP = 3.0 (制冷) 和 COP = 4.0 (供暖，热泵)
        cooling_cop = 3.0
        heating_cop = 4.0
        annual_cooling_kwh = np.sum(cooling_load_hourly_W) / 1000.0 / cooling_cop
        annual_heating_kwh = np.sum(heating_load_hourly_W) / 1000.0 / heating_cop
        
        # 实现 SDI 接口（Interface for Task 2）
        # 定义供暖需求 (简单的设定点判断)
        heating_load_mask = temp_air_rc < self.building_cfg.heating_setpoint_C
        
        # 计算 SDI (Shading Demand Index)
        # 逻辑：
        # 1. 如果当前有冷负荷 (cooling_load > 0) -> SDI = +1 * Q_solar (需要遮阳)
        # 2. 如果当前有供暖需求 (temp < 20) -> SDI = -1 * Q_solar (需要阳光)
        # 3. 否则 -> SDI = 0
        
        current_sdi = np.zeros_like(Q_solar_hourly_sum)
        # 制冷工况：SDI 正值
        current_sdi[cooling_load_hourly_W > 0] = Q_solar_hourly_sum[cooling_load_hourly_W > 0]
        # 供暖工况：SDI 负值
        current_sdi[heating_load_mask] = -1.0 * Q_solar_hourly_sum[heating_load_mask]
        
        # 生成 Mask Target (仅当 SDI > 0 时需要遮阳)
        mask_target = (current_sdi > 0).astype(int)
        
        # 修改 hourly_data 的存储逻辑 (如果是 detailed 模式)
        if detailed:
            # 重构 hourly_data，使用汇总信息
            hourly_data = []
            for i in range(len(self.weather)):
                hourly_data.append({
                    'time': self.weather.index[i],
                    'T_out': temp_air[i],
                    'T_in': temp_air_rc[i], # RC 模型算出的真实 T_in
                    'Q_solar_total': Q_solar_hourly_sum[i],
                    'Cooling_Load_W': cooling_load_hourly_W[i],
                    'Heating_Load_W': heating_load_hourly_W[i],
                    'SDI': current_sdi[i],
                    'Mask_Target': mask_target[i]
                })
        
        # Calculate discomfort penalty
        discomfort_penalty = calculate_discomfort_penalty(ASE_hours_total)
        
        # Calculate cost
        # Simple cost model with glass pricing tiers
        installation_factor = 0.20  # 安装费占材料费的 20%
        maintenance_rate = 0.02  # 年维护费占总 CAPEX 的 2%
        
        # 玻璃分级定价
        price_glass_std = 300.0  # 普通玻璃 ($/m²)
        price_glass_high = 450.0  # 高性能玻璃 ($/m²)
        
        # 根据 SHGC 判断玻璃类型
        if shgc < 0.5:  # 高性能玻璃
            window_price = price_glass_high
        else:  # 普通玻璃
            window_price = price_glass_std
        
        wall_cost_per_m2 = 150.0  # 实墙造价
        window_area = sum(win_areas.values()) * wwr
        wall_area = sum(win_areas.values()) * (1 - wwr)  # 实墙面积
        overhang_length = sum(win_areas.values()) * 0.5  # Approximate
        overhang_cost = sum(overhang_depths_m.values()) * overhang_length * 150.0  # 遮阳板单价调整为 $150/m
        window_cost = window_area * window_price
        wall_cost = wall_area * wall_cost_per_m2
        
        # 计算材料费和安装费
        material_cost = window_cost + overhang_cost + wall_cost
        installation_cost = material_cost * installation_factor
        total_capex = material_cost + installation_cost
        
        # Calculate energy cost
        # 奥斯陆的供暖成本权重设为电费的 1.5 倍
        # 假设电费为 0.15 USD/kWh，供暖费为 0.225 USD/kWh
        electricity_cost_per_kwh = 0.15
        heating_cost_per_kwh = 0.225  # 1.5 倍电费
        
        annual_cooling_cost = annual_cooling_kwh * electricity_cost_per_kwh
        annual_heating_cost = annual_heating_kwh * heating_cost_per_kwh
        annual_energy_cost = annual_cooling_cost + annual_heating_cost
        
        # Calculate ROI
        # 基线成本也包含实墙成本
        baseline_wwr = 0.4
        baseline_window_area = sum(win_areas.values()) * baseline_wwr
        baseline_wall_area = sum(win_areas.values()) * (1 - baseline_wwr)
        baseline_window_cost = baseline_window_area * 1000.0
        baseline_wall_cost = baseline_wall_area * wall_cost_per_m2
        baseline_material_cost = baseline_window_cost + baseline_wall_cost  # 基线没有遮阳板
        baseline_installation_cost = baseline_material_cost * installation_factor
        baseline_capex = baseline_material_cost + baseline_installation_cost
        
        # 使用保存的基准能耗计算节能
        baseline_heating_kwh = getattr(self, 'baseline_heating_kwh', annual_heating_kwh)
        baseline_cooling_kwh = getattr(self, 'baseline_cooling_kwh', annual_cooling_kwh)
        
        # 计算年度节能
        baseline_cooling_cost = baseline_cooling_kwh * electricity_cost_per_kwh
        baseline_heating_cost = baseline_heating_kwh * heating_cost_per_kwh
        baseline_energy_cost = baseline_cooling_cost + baseline_heating_cost
        annual_energy_savings = baseline_energy_cost - annual_energy_cost
        
        # 计算年维护费和净年度收益
        annual_maintenance_cost = total_capex * maintenance_rate
        net_annual_savings = annual_energy_savings - annual_maintenance_cost
        
        # 计算 ROI
        # 对于既有建筑改造，投资为全额改造成本，不能减去旧墙的“残值”
        investment = total_capex
        
        # 确保 ROI 计算的合理性
        if investment <= 0:
            # 如果投资为零，ROI 为正（节省了成本）
            roi = 1.0  # 表示非常好的投资回报
        else:
            # 正常计算 ROI
            roi = ROI_20(investment, net_annual_savings)
        
        # Calculate score for Borealis (heating-dominated)
        # 目标函数：最小化总能耗费用 + 惩罚项
        total_cost = annual_energy_cost + discomfort_penalty
        # 为了与优化器兼容，返回负值作为分数（优化器寻找最大值）
        score = -total_cost
        
        # Create result dictionary
        result = {
            'label': label,
            'wwr': wwr,
            'overhang_depths_m': overhang_depths_m,
            'annual_cooling_kwh': annual_cooling_kwh,
            'annual_heating_kwh': annual_heating_kwh,
            'total_energy_cost': annual_energy_cost,
            'ase_hours': ASE_hours_total,
            'discomfort_penalty': discomfort_penalty,
            'cost_usd': total_capex,
            'material_cost': material_cost,
            'installation_cost': installation_cost,
            'annual_maintenance_cost': annual_maintenance_cost,
            'net_annual_savings': net_annual_savings,
            'roi': roi,
            'score': score
        }
        
        # Add hourly data to result if detailed
        if detailed:
            result['hourly_data'] = hourly_data
        
        # Return result
        return result
    
    def export_hourly_data(self, wwr, overhang_depths_m, output_path):
        """
        Export hourly simulation data to CSV file.
        
        Args:
            wwr: Window-to-wall ratio.
            overhang_depths_m: Overhang depths for each facade (meters).
            output_path: Output CSV file path.
        """
        # Run detailed simulation
        result = self.simulate(wwr, overhang_depths_m, label="detailed", detailed=True)
        
        # Create DataFrame from hourly data
        df = pd.DataFrame(result['hourly_data'])
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Hourly data exported to: {output_path}")


def main():
    """
    Main function for Task 2 runner.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ICM E Task 2: Borealis retrofit (Oslo)")
    parser.add_argument("--epw", type=str, default=os.path.join("data", "epw", "NOR_OS_Oslo.Blindern.014920_TMYx.2009-2023.epw"),
                        help="EPW file path.")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic weather data instead of EPW.")
    parser.add_argument("--optimizer", type=str, default="de", choices=["de", "grid"],
                        help="Optimization method.")
    parser.add_argument("--wwr_min", type=float, default=0.2,
                        help="Minimum window-to-wall ratio.")
    parser.add_argument("--wwr_max", type=float, default=0.6,
                        help="Maximum window-to-wall ratio.")
    parser.add_argument("--d_min", type=float, default=0.0,
                        help="Minimum overhang depth (m).")
    parser.add_argument("--d_max", type=float, default=0.8,
                        help="Maximum overhang depth (m) - limited for historical building.")
    parser.add_argument("--baseline_wwr", type=float, default=0.40,
                        help="Baseline window-to-wall ratio.")
    parser.add_argument("--w1_energy", type=float, default=1.0,
                        help="Weight for energy objective.")
    parser.add_argument("--w2_cost", type=float, default=1.0,
                        help="Weight for cost objective.")
    parser.add_argument("--w3_discomfort", type=float, default=0.01,
                        help="Weight for discomfort objective.")
    args = parser.parse_args()
    
    # Create simulation engine
    engine = SimulationEngine(
        epw_path=None if args.synthetic else args.epw,
        use_synthetic=args.synthetic,
        building_cfg=BuildingConfig(),
    )
    
    # Output directories
    results_dir = os.path.join("results", "task2")
    tables_dir = os.path.join(results_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)

    # Run baseline simulation
    baseline = engine.set_baseline(baseline_wwr=args.baseline_wwr)
    
    # Print baseline results
    print("\n================= BASELINE =================")
    print(f"Label: {baseline['label']}")
    print(f"  WWR={baseline['wwr']:.3f} | dN={baseline['overhang_depths_m']['N']:.2f} dE={baseline['overhang_depths_m']['E']:.2f} dS={baseline['overhang_depths_m']['S']:.2f} dW={baseline['overhang_depths_m']['W']:.2f}")
    print(f"  Annual Heating (kWh)    = {int(baseline['annual_heating_kwh']):,}")
    print(f"  Annual Cooling (kWh)    = {int(baseline['annual_cooling_kwh']):,}")
    print(f"  Total Energy Cost (USD) = {int(baseline['total_energy_cost']):,}")
    print(f"  ASE proxy (hours/year)  = {int(baseline['ase_hours']):,}")
    print(f"  Discomfort penalty      = {baseline['discomfort_penalty']:.1f}")
    print(f"  Cost (USD)              = {int(baseline['cost_usd']):,}")
    print(f"  ROI                     = {baseline['roi']:.3f}")
    print(f"  Score                   = {baseline['score']:.3f}")
    
    # Run optimization
    print("\nRunning optimization...")
    opt_cfg = OptimizationConfig(
        method=args.optimizer,
        wwr_bounds=(args.wwr_min, args.wwr_max),
        d_bounds_m=(args.d_min, args.d_max),
        w1_energy=args.w1_energy,
        w2_cost=args.w2_cost,
        w3_discomfort=args.w3_discomfort,
        grid_n=4,
    )
    
    best = run_optimization(engine, opt_cfg)
    
    # Run final simulation with optimal parameters
    retrofit = engine.simulate(
        wwr=best.params["wwr"],
        overhang_depths_m={"N": best.params["dN"], "E": best.params["dE"], "S": best.params["dS"], "W": best.params["dW"]},
        label="retrofit_best"
    )
    
    # Export hourly data for baseline
    output_path_baseline = os.path.join(tables_dir, "task2_hourly_baseline.csv")
    engine.export_hourly_data(
        wwr=args.baseline_wwr,
        overhang_depths_m={"N": 0.0, "E": 0.0, "S": 0.0, "W": 0.0},
        output_path=output_path_baseline
    )
    
    # Export hourly data for best retrofit
    output_path_best = os.path.join(tables_dir, "task2_hourly_best.csv")
    engine.export_hourly_data(
        wwr=best.params["wwr"],
        overhang_depths_m={"N": best.params["dN"], "E": best.params["dE"], "S": best.params["dS"], "W": best.params["dW"]},
        output_path=output_path_best
    )
    
    # Print retrofit results
    print("\n================= BEST RETROFIT =================")
    print(f"Label: {retrofit['label']}")
    print(f"  WWR={retrofit['wwr']:.3f} | dN={retrofit['overhang_depths_m']['N']:.2f} dE={retrofit['overhang_depths_m']['E']:.2f} dS={retrofit['overhang_depths_m']['S']:.2f} dW={retrofit['overhang_depths_m']['W']:.2f}")
    print(f"  Annual Heating (kWh)    = {int(retrofit['annual_heating_kwh']):,}")
    print(f"  Annual Cooling (kWh)    = {int(retrofit['annual_cooling_kwh']):,}")
    print(f"  Total Energy Cost (USD) = {int(retrofit['total_energy_cost']):,}")
    print(f"  ASE proxy (hours/year)  = {int(retrofit['ase_hours']):,}")
    print(f"  Discomfort penalty      = {retrofit['discomfort_penalty']:.1f}")
    print(f"  Cost (USD)              = {int(retrofit['cost_usd']):,}")
    print(f"  ROI                     = {retrofit['roi']:.3f}")
    print(f"  Score                   = {retrofit['score']:.3f}")
    
    # Calculate savings
    heating_savings_rate = calculate_energy_savings(
        baseline['annual_heating_kwh'],
        retrofit['annual_heating_kwh']
    )
    
    cooling_savings_rate = calculate_energy_savings(
        baseline['annual_cooling_kwh'],
        retrofit['annual_cooling_kwh']
    )
    
    total_energy_savings_rate = calculate_energy_savings(
        baseline['annual_heating_kwh'] + baseline['annual_cooling_kwh'],
        retrofit['annual_heating_kwh'] + retrofit['annual_cooling_kwh']
    )
    
    print(f"\nHeating savings rate vs baseline: {heating_savings_rate:.3f}%")
    print(f"Cooling savings rate vs baseline: {cooling_savings_rate:.3f}%")
    print(f"Total energy savings rate vs baseline: {total_energy_savings_rate:.3f}%")
    print(f"Baseline hourly data exported to: {output_path_baseline}")
    print(f"Best retrofit hourly data exported to: {output_path_best}")


if __name__ == "__main__":
    main()
