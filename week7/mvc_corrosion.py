#################################################################################
# WaterTAP Copyright (c) 2020-2024, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory, Oak Ridge National Laboratory,
# National Renewable Energy Laboratory, and National Energy Technology
# Laboratory (subject to receipt of any required approvals from the U.S. Dept.
# of Energy). All rights reserved.
#
# Please see the files COPYRIGHT.md and LICENSE.md for full copyright and license
# information, respectively. These files are also available online at the URL
# "https://github.com/watertap-org/watertap/"
#################################################################################

from pyomo.environ import (
    ConcreteModel,
    value,
    Constraint,
    Objective,
    Var,
    Expression,
    Set,
    TransformationFactory,
    units as pyunits,
    check_optimal_termination,
    assert_optimal_termination,
    Param,
)
from pyomo.network import Arc, SequentialDecomposition

import pyomo.environ as pyo
from pyomo.util.calc_var_value import calculate_variable_from_constraint
from idaes.core import FlowsheetBlock
from watertap.core.solvers import get_solver
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.core.util.initialization import propagate_state
from idaes.models.unit_models import Feed, Separator, Mixer, Product
from idaes.models.unit_models.translator import Translator
from idaes.models.unit_models.separator import SplittingType
from idaes.models.unit_models.mixer import MomentumMixingType
from idaes.models.unit_models.heat_exchanger import (
    HeatExchanger,
    HeatExchangerFlowPattern,
)
from idaes.core import UnitModelCostingBlock
import idaes.core.util.scaling as iscale
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from idaes.core.surrogate.surrogate_block import SurrogateBlock
import idaes.logger as idaeslog

from watertap.unit_models.mvc.components import Evaporator, Compressor, Condenser
from watertap.unit_models.mvc.components.lmtd_chen_callback import (
    delta_temperature_chen_callback,
)
from watertap.unit_models.pressure_changer import Pump
import watertap.property_models.seawater_prop_pack as props_sw
import watertap.property_models.water_prop_pack as props_w
from watertap.costing import WaterTAPCosting
import math
import numpy as np
import pandas as pd
import time

def single_run(material='Stainless steel 316',
               wf=100,
               rr=0.5,
               do=0,
               ph=7.5,
               output_level='ERROR',
               save=False,
               results_folder='results_higher_resolution',
               save_name=None,
               display=False):

    # build, set operating conditions, initialize for simulation
    m = build(material=material)
    set_operating_conditions(m)
    add_Q_ext(m, time_point=m.fs.config.time)
    initialize_system(m,output_level=output_level)
    scale_costs(m) # rescale costs after initialization because scaling depends on flow rates
    fix_outlet_pressures(m)  # outlet pressure are initially unfixed for initialization

    # set up for minimizing Q_ext in first solve - should be 1 DOF because Q_ext is unfixed
    print("DOF after initialization: ", degrees_of_freedom(m))
    m.fs.objective = Objective(expr=m.fs.Q_ext[0])

    print("\n***---First solve - minimizing Q_ext---***")
    solver = get_solver()
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    assert_optimal_termination(results)
    if display:
        display_metrics(m)
        display_design(m)

    print("\n***---Second solve - optimization---***")
    add_evap_hx_material_factor_equal_constraint(m)
    m.fs.Q_ext[0].fix(0)  # no longer want external heating in evaporator
    del m.fs.objective
    set_up_optimization(m)
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    if display:
        display_metrics(m)
        display_design(m)

    print("\n***---Third solve - optimization with corrosion rate surrogate, new conditions---***")
    add_corrosion_rate_surrogate(m)
    set_surrogate_conditions(m,do,ph)
    m.fs.evaporator.properties_vapor[0].temperature.setub(95 + 273.15)
    m.fs.evaporator.properties_vapor[0].temperature.setlb(45 + 273.15)
    m.fs.recovery[0].fix(rr)
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].fix(wf/1e3)
    results = solve(m, solver=solver, tee=False)
    print("Termination condition: ", results.solver.termination_condition)
    if display:
        display_metrics(m)
        display_design(m)
        display_corrosion(m)

    if save:
        save_single_run(m, material, wf, rr, do, ph, results=results, results_folder=results_folder, save_name=save_name)

    return m, results


def build(material):
    # _log = idaeslog.getLogger(name='framework', level=50)
    # flowsheet set up
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.material = Param(initialize=material, within=[material])

    # Properties
    m.fs.properties_feed = props_sw.SeawaterParameterBlock()
    m.fs.properties_vapor = props_w.WaterParameterBlock()

    # Unit models
    m.fs.feed = Feed(property_package=m.fs.properties_feed)

    m.fs.pump_feed = Pump(property_package=m.fs.properties_feed)

    m.fs.separator_feed = Separator(
        property_package=m.fs.properties_feed,
        outlet_list=["hx_distillate_cold", "hx_brine_cold"],
        split_basis=SplittingType.totalFlow,
    )

    m.fs.hx_distillate = HeatExchanger(
        hot_side_name="hot",
        cold_side_name="cold",
        hot={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        cold={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        delta_temperature_callback=delta_temperature_chen_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
    )
    # Set lower bound of approach temperatures
    m.fs.hx_distillate.delta_temperature_in.setlb(0)
    m.fs.hx_distillate.delta_temperature_out.setlb(0)
    m.fs.hx_distillate.area.setlb(10)

    m.fs.hx_brine = HeatExchanger(
        hot_side_name="hot",
        cold_side_name="cold",
        hot={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        cold={"property_package": m.fs.properties_feed, "has_pressure_change": True},
        delta_temperature_callback=delta_temperature_chen_callback,
        flow_pattern=HeatExchangerFlowPattern.countercurrent,
    )
    # Set lower bound of approach temperatures
    m.fs.hx_brine.delta_temperature_in.setlb(0)
    m.fs.hx_brine.delta_temperature_out.setlb(0)
    m.fs.hx_brine.area.setlb(10)

    m.fs.mixer_feed = Mixer(
        property_package=m.fs.properties_feed,
        momentum_mixing_type=MomentumMixingType.equality,
        inlet_list=["hx_distillate_cold", "hx_brine_cold"],
    )
    m.fs.mixer_feed.pressure_equality_constraints[0, 2].deactivate()

    m.fs.evaporator = Evaporator(
        property_package_feed=m.fs.properties_feed,
        property_package_vapor=m.fs.properties_vapor,
    )

    m.fs.compressor = Compressor(property_package=m.fs.properties_vapor)

    m.fs.condenser = Condenser(property_package=m.fs.properties_vapor)

    m.fs.tb_distillate = Translator(
        inlet_property_package=m.fs.properties_vapor,
        outlet_property_package=m.fs.properties_feed,
    )

    # Translator block to convert distillate exiting condenser from water to seawater prop pack
    @m.fs.tb_distillate.Constraint()
    def eq_flow_mass_comp(blk):
        return (
            blk.properties_in[0].flow_mass_phase_comp["Liq", "H2O"]
            == blk.properties_out[0].flow_mass_phase_comp["Liq", "H2O"]
        )

    @m.fs.tb_distillate.Constraint()
    def eq_temperature(blk):
        return blk.properties_in[0].temperature == blk.properties_out[0].temperature

    @m.fs.tb_distillate.Constraint()
    def eq_pressure(blk):
        return blk.properties_in[0].pressure == blk.properties_out[0].pressure

    m.fs.pump_brine = Pump(property_package=m.fs.properties_feed)

    m.fs.pump_distillate = Pump(property_package=m.fs.properties_feed)

    m.fs.distillate = Product(property_package=m.fs.properties_feed)

    m.fs.brine = Product(property_package=m.fs.properties_feed)

    # Connections and connect condenser and evaporator
    m.fs.s01 = Arc(source=m.fs.feed.outlet, destination=m.fs.pump_feed.inlet)
    m.fs.s02 = Arc(source=m.fs.pump_feed.outlet, destination=m.fs.separator_feed.inlet)
    m.fs.s03 = Arc(
        source=m.fs.separator_feed.hx_distillate_cold,
        destination=m.fs.hx_distillate.cold_inlet,
    )
    m.fs.s04 = Arc(
        source=m.fs.separator_feed.hx_brine_cold, destination=m.fs.hx_brine.cold_inlet
    )
    m.fs.s05 = Arc(
        source=m.fs.hx_distillate.cold_outlet,
        destination=m.fs.mixer_feed.hx_distillate_cold,
    )
    m.fs.s06 = Arc(
        source=m.fs.hx_brine.cold_outlet, destination=m.fs.mixer_feed.hx_brine_cold
    )
    m.fs.s07 = Arc(
        source=m.fs.mixer_feed.outlet, destination=m.fs.evaporator.inlet_feed
    )
    m.fs.s08 = Arc(
        source=m.fs.evaporator.outlet_vapor, destination=m.fs.compressor.inlet
    )
    m.fs.s09 = Arc(source=m.fs.compressor.outlet, destination=m.fs.condenser.inlet)
    m.fs.s10 = Arc(
        source=m.fs.evaporator.outlet_brine, destination=m.fs.pump_brine.inlet
    )
    m.fs.s11 = Arc(source=m.fs.pump_brine.outlet, destination=m.fs.hx_brine.hot_inlet)
    m.fs.s12 = Arc(source=m.fs.hx_brine.hot_outlet, destination=m.fs.brine.inlet)
    m.fs.s13 = Arc(source=m.fs.condenser.outlet, destination=m.fs.tb_distillate.inlet)
    m.fs.s14 = Arc(
        source=m.fs.tb_distillate.outlet, destination=m.fs.pump_distillate.inlet
    )
    m.fs.s15 = Arc(
        source=m.fs.pump_distillate.outlet, destination=m.fs.hx_distillate.hot_inlet
    )
    m.fs.s16 = Arc(
        source=m.fs.hx_distillate.hot_outlet, destination=m.fs.distillate.inlet
    )

    TransformationFactory("network.expand_arcs").apply_to(m)

    m.fs.evaporator.connect_to_condenser(m.fs.condenser)

    # Add costing
    add_costing(m)

    # Add recovery ratio
    m.fs.recovery = Var(m.fs.config.time, initialize=0.5, bounds=(0, 1))
    m.fs.recovery_equation = Constraint(
        expr=m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"]
        == m.fs.recovery[0]
        * (
            m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
            + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]
        )
    )

    # Make split ratio equal to recovery
    m.fs.split_ratio_recovery_equality = Constraint(
        expr=m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"]
        == m.fs.recovery[0]
    )

    # Scaling
    # properties
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )
    m.fs.properties_feed.set_default_scaling(
        "flow_mass_phase_comp", 1e2, index=("Liq", "TDS")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Vap", "H2O")
    )
    m.fs.properties_vapor.set_default_scaling(
        "flow_mass_phase_comp", 1, index=("Liq", "H2O")
    )

    # unit model values
    # pumps
    iscale.set_scaling_factor(m.fs.pump_feed.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.pump_brine.control_volume.work, 1e-3)
    iscale.set_scaling_factor(m.fs.pump_distillate.control_volume.work, 1e-3)

    # distillate HX
    iscale.set_scaling_factor(m.fs.hx_distillate.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_distillate.cold.heat, 1e-3)
    iscale.set_scaling_factor(
        m.fs.hx_distillate.overall_heat_transfer_coefficient, 1e-3
    )

    iscale.set_scaling_factor(m.fs.hx_distillate.area, 1e-1)
    iscale.constraint_scaling_transform(
        m.fs.hx_distillate.cold_side.pressure_balance[0], 1e-5
    )
    iscale.constraint_scaling_transform(
        m.fs.hx_distillate.hot_side.pressure_balance[0], 1e-5
    )

    # brine HX
    iscale.set_scaling_factor(m.fs.hx_brine.hot.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_brine.cold.heat, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_brine.overall_heat_transfer_coefficient, 1e-3)
    iscale.set_scaling_factor(m.fs.hx_brine.area, 1e-1)
    iscale.constraint_scaling_transform(
        m.fs.hx_brine.cold_side.pressure_balance[0], 1e-5
    )
    iscale.constraint_scaling_transform(
        m.fs.hx_brine.hot_side.pressure_balance[0], 1e-5
    )

    # evaporator
    iscale.set_scaling_factor(m.fs.evaporator.area, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.U, 1e-3)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_in, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.delta_temperature_out, 1e-1)
    iscale.set_scaling_factor(m.fs.evaporator.lmtd, 1e-1)

    # compressor
    iscale.set_scaling_factor(m.fs.compressor.control_volume.work, 1e-6)

    # condenser
    iscale.set_scaling_factor(m.fs.condenser.control_volume.heat, 1e-6)

    # calculate and propagate scaling factors
    iscale.calculate_scaling_factors(m)

    return m


def add_corrosion_rate_surrogate(m):
    # Need to replace with path to surrogate model
    surrogate_dir = f"/path/to/surrogate/models/{m.fs.material.value}/"

    # 1. Add indexed version of inputs: temperature, brine salinity, dissolved oxygen
    m.fs.temperature_indexed = Var(
        [0],
        initialize=m.fs.evaporator.properties_brine[0].temperature.value,
        units=pyunits.dimensionless
    )
    m.fs.eq_temperature_indexed = Constraint(
        expr=m.fs.evaporator.properties_brine[0].temperature == m.fs.temperature_indexed[0] + 273.15
    )

    brine_salt = m.fs.evaporator.properties_brine[0].flow_mass_phase_comp['Liq','TDS'].value
    brine_water = m.fs.evaporator.properties_brine[0].flow_mass_phase_comp['Liq','H2O'].value
    m.fs.brine_salinity_indexed = Var(
        [0],
        initialize=brine_salt/(brine_water + brine_salt),
        units=pyunits.dimensionless,
    )
    m.fs.eq_brine_salinity_indexed = Constraint(
        expr=m.fs.evaporator.properties_brine[0].mass_frac_phase_comp['Liq','TDS'] == m.fs.brine_salinity_indexed[0]
    )

    m.fs.dissolved_oxygen_index = Var(
        [0],
        initialize=0,
        units=pyunits.dimensionless
    )

    m.fs.pH_index = Var(
        [0],
        initialize=7.5,
        units=pyunits.dimensionless
    )

    # 2. Add corrosion rate and potential difference outputs
    m.fs.corrosion_rate_indexed = Var(
        [0],
        initialize=0.1,
        units=pyunits.dimensionless
    )
    iscale.set_scaling_factor(m.fs.corrosion_rate_indexed[0], 1e2)

    m.fs.corrosion_rate = Var(
        initialize=0.1,
        bounds=(0, 0.1), # cannot exceed 0.1
        units=pyunits.mm / pyunits.year
    )
    iscale.set_scaling_factor(m.fs.corrosion_rate, 1e2)
    m.fs.eq_corrosion_rate_indexed = Constraint(
        expr=m.fs.corrosion_rate==m.fs.corrosion_rate_indexed[0]
    )

    m.fs.potential_difference_indexed = Var(
        [0],
        initialize=0.0,
        units=pyunits.dimensionless
    )
    iscale.set_scaling_factor(m.fs.potential_difference_indexed[0], 1e3)

    m.fs.potential_difference = Var(
        initialize=0,
        bounds=(0,100),
        units=pyunits.dimensionless
    )
    iscale.set_scaling_factor(m.fs.potential_difference, 1e3)

    m.fs.eq_potential_difference_indexed = Constraint(
        expr=m.fs.potential_difference == m.fs.potential_difference_indexed[0]
    )
    # 3. Add corrosion rate surrogate - input order:'Temperature', 'Brine salinity', 'Dissolved oxygen mgO2'
    filename = surrogate_dir + "final_surrogate/corrosion_rate.json" # replace with correct path
    corrosion_rate_surrogate = PysmoSurrogate.load_from_file(filename)
    m.fs.corrosion_rate_surrogate = SurrogateBlock(concrete=True)
    m.fs.corrosion_rate_surrogate.build_model(corrosion_rate_surrogate,
                                              input_vars=[m.fs.temperature_indexed[0],
                                                          m.fs.brine_salinity_indexed[0],
                                                          m.fs.dissolved_oxygen_index[0],
                                                          m.fs.pH_index[0]],
                                              output_vars=[m.fs.corrosion_rate_indexed[0]])
    # check value
    calculate_variable_from_constraint(m.fs.corrosion_rate_indexed[0], m.fs.corrosion_rate_surrogate.pysmo_constraint['corrosionRateMmPerYear'])

    # 4. Add potential different surrogate
    filename = surrogate_dir + "final_surrogate/potential_difference.json" # replace with correct path
    potential_difference_surrogate = PysmoSurrogate.load_from_file(filename)
    m.fs.potential_difference_surrogate = SurrogateBlock(concrete=True)
    m.fs.potential_difference_surrogate.build_model(potential_difference_surrogate,
                                              input_vars=[m.fs.temperature_indexed[0],
                                                          m.fs.brine_salinity_indexed[0],
                                                          m.fs.dissolved_oxygen_index[0],
                                                          m.fs.pH_index[0]],
                                              output_vars=[m.fs.potential_difference_indexed[0]])
    # check value
    calculate_variable_from_constraint(m.fs.potential_difference_indexed[0], m.fs.potential_difference_surrogate.pysmo_constraint['repassivation_corrosion_potential_difference'])
    m.fs.brine_salinity_indexed[0].setub(0.26)

def set_surrogate_conditions(m,do=0, pH=7.5):
    # fix dissolved oxygen
    m.fs.dissolved_oxygen_index[0].fix(do)
    # fix pH
    m.fs.pH_index[0].fix(pH)
    # fix material factor corresponding to surrogate
    material_factor = {
        "Carbon steel 1018": 1,
        "Stainless steel 304": 3.0,
        "Stainless steel 316": 3.2,
        "Duplex stainless 2205": 3.5,
        "Duplex stainless 2507": 4.0,
        "Nickel alloy 825": 5.0,
        "Nickel alloy 625": 6.0
    }
    m.fs.costing.evaporator.material_factor_cost.fix(material_factor[m.fs.material.value])

def add_Q_ext(m, time_point=None):
    # Allows additional heat to be added to evaporator so that an initial feasible solution can be found as a starting
    # guess for optimization in case physically infeasible simulation is proposed

    if time_point is None:
        time_point = m.fs.config.time
    m.fs.Q_ext = Var(time_point, initialize=0, units=pyunits.J / pyunits.s)
    m.fs.Q_ext[0].setlb(0)
    m.fs.evaporator.eq_energy_balance.deactivate()
    m.fs.evaporator.eq_energy_balance_with_additional_Q = Constraint(
        expr=m.fs.evaporator.heat_transfer
        + m.fs.Q_ext[0]
        + m.fs.evaporator.properties_feed[0].enth_flow
        == m.fs.evaporator.properties_brine[0].enth_flow
        + m.fs.evaporator.properties_vapor[0].enth_flow_phase["Vap"]
    )
    iscale.set_scaling_factor(m.fs.Q_ext, 1e-6)

def add_costing(m):
    m.fs.costing = WaterTAPCosting()
    m.fs.pump_feed.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.pump_distillate.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.pump_brine.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.hx_distillate.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.hx_brine.costing = UnitModelCostingBlock(flowsheet_costing_block=m.fs.costing)
    m.fs.mixer_feed.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.evaporator.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )
    m.fs.compressor.costing = UnitModelCostingBlock(
        flowsheet_costing_block=m.fs.costing
    )

    m.fs.costing.cost_process()
    m.fs.costing.add_annual_water_production(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_LCOW(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.add_specific_energy_consumption(m.fs.distillate.properties[0].flow_vol)
    m.fs.costing.base_currency = pyo.units.USD_2020

    # Add costing expressions
    m.fs.costing.MVC_components = Set(initialize=["feed_pump",
                                                  "distillate_pump",
                                                  "brine_pump",
                                                  "hx_distillate",
                                                  "hx_brine",
                                                  "mixer",
                                                  "evaporator",
                                                  "compressor"])
    # Percentage of capital costs
    m.fs.costing.MVC_capital_cost_percentage = Expression(m.fs.costing.MVC_components)
    m.fs.costing.MVC_capital_cost_percentage["feed_pump"] = (
            m.fs.pump_feed.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["distillate_pump"] = (
            m.fs.pump_distillate.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["brine_pump"] = (
            m.fs.pump_brine.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["hx_distillate"] = (
            m.fs.hx_distillate.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["hx_brine"] = (
            m.fs.hx_brine.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["mixer"] = (
            m.fs.mixer_feed.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["evaporator"] = (
            m.fs.evaporator.costing.capital_cost / m.fs.costing.aggregate_capital_cost)
    m.fs.costing.MVC_capital_cost_percentage["compressor"] = (
            m.fs.compressor.costing.capital_cost / m.fs.costing.aggregate_capital_cost)

    # Percentage of costs normalized to LCOW
    m.fs.costing.annual_operating_costs = Expression(
        expr=m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor + m.fs.costing.total_operating_cost)
    m.fs.costing.MVC_LCOW_comp = Set(initialize=["feed_pump",
                                                 "distillate_pump",
                                                 "brine_pump",
                                                 "hx_distillate",
                                                 "hx_brine",
                                                 "mixer",
                                                 "evaporator",
                                                 "compressor",
                                                 "electricity",
                                                 "MLC",
                                                 "capital_costs",
                                                 "operating_costs",
                                                 "capex_opex_ratio"])
    m.fs.costing.LCOW_percentage = Expression(m.fs.costing.MVC_LCOW_comp)
    m.fs.costing.LCOW_percentage["feed_pump"] = (
            m.fs.pump_feed.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["distillate_pump"] = (
            m.fs.pump_distillate.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["brine_pump"] = (
            m.fs.pump_brine.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["hx_distillate"] = (
            m.fs.hx_distillate.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["hx_brine"] = (
            m.fs.hx_brine.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["mixer"] = (
            m.fs.mixer_feed.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["evaporator"] = (
            m.fs.evaporator.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["compressor"] = (
            m.fs.compressor.costing.capital_cost * m.fs.costing.total_investment_factor * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage['electricity'] = (m.fs.costing.aggregate_flow_costs[
                                                       'electricity'] * m.fs.costing.utilization_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage['MLC'] = (
            m.fs.costing.maintenance_labor_chemical_operating_cost / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["capital_costs"] = (
            m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage["operating_costs"] = (
            m.fs.costing.total_operating_cost / m.fs.costing.annual_operating_costs)
    m.fs.costing.LCOW_percentage['capex_opex_ratio'] = (
            m.fs.costing.total_capital_cost * m.fs.costing.capital_recovery_factor / m.fs.costing.total_operating_cost)

def add_evap_hx_material_factor_equal_constraint(m):
    m.fs.costing.evaporator.material_factor_cost.fix()
    m.fs.costing.heat_exchanger.material_factor_cost.unfix()
    # make HX material factor equal to evaporator material factor
    m.fs.costing.hx_material_factor_constraint = Constraint(
        expr=m.fs.costing.heat_exchanger.material_factor_cost == m.fs.costing.evaporator.material_factor_cost)

def set_operating_conditions(m):
    # Feed inlet
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].fix(0.1)
    m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].fix(40)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].pressure.fix(101325)

    m.fs.recovery[0].fix(0.5)

    # Feed pump
    m.fs.pump_feed.efficiency_pump[0].fix(0.8)
    m.fs.pump_feed.control_volume.deltaP[0].fix(7e3)

    # Separator
    m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"] = m.fs.recovery[0].value

    # Distillate HX
    m.fs.hx_distillate.overall_heat_transfer_coefficient[0].fix(2e3)
    m.fs.hx_distillate.area.fix(125)
    m.fs.hx_distillate.cold.deltaP[0].fix(7e3)
    m.fs.hx_distillate.hot.deltaP[0].fix(7e3)

    # Brine HX
    m.fs.hx_brine.overall_heat_transfer_coefficient[0].fix(2e3)
    m.fs.hx_brine.area.fix(115)
    m.fs.hx_brine.cold.deltaP[0].fix(7e3)
    m.fs.hx_brine.hot.deltaP[0].fix(7e3)

    # Evaporator
    m.fs.evaporator.inlet_feed.temperature[0] = 50 + 273.15  # provide guess
    m.fs.evaporator.outlet_brine.temperature[0].fix(70 + 273.15)
    m.fs.evaporator.U.fix(3e3)  # W/K-m^2
    m.fs.evaporator.area.setub(1e4)  # m^2

    # Compressor
    m.fs.compressor.pressure_ratio.fix(1.6)
    m.fs.compressor.efficiency.fix(0.8)

    # Brine pump
    m.fs.pump_brine.efficiency_pump[0].fix(0.8)
    m.fs.pump_brine.control_volume.deltaP[0].fix(4e4)

    # Distillate pump
    m.fs.pump_distillate.efficiency_pump[0].fix(0.8)
    m.fs.pump_distillate.control_volume.deltaP[0].fix(4e4)

    # Fix 0 TDS
    m.fs.tb_distillate.properties_out[0].flow_mass_phase_comp["Liq", "TDS"].fix(1e-5)

    # Costing
    m.fs.costing.TIC.fix(2)
    m.fs.costing.electricity_cost = 0.1  # 0.15
    m.fs.costing.heat_exchanger.material_factor_cost.fix(5)
    m.fs.costing.evaporator.material_factor_cost.fix(5)
    m.fs.costing.compressor.unit_cost.fix(1 * 7364)

    # Temperature bounds
    m.fs.evaporator.properties_brine[0].temperature.setlb(45 + 273.15)
    m.fs.evaporator.properties_brine[0].temperature.setub(75 + 273.15) # assumed when not accounting for corrosion
    m.fs.compressor.control_volume.properties_out[0].temperature.setub(450)

    # Pressure ratio bounds
    m.fs.compressor.pressure_ratio.setub(4)

    # check degrees of freedom
    print("DOF after setting operating conditions: ", degrees_of_freedom(m))

def initialize_system(m, solver=None,output_level='ERROR'):
    if solver is None:
        solver = get_solver()
    optarg = solver.options

    # Touch feed mass fraction property
    m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"]
    solver.solve(m.fs.feed)

    # Propagate vapor flow rate based on given recovery
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp[
        "Vap", "H2O"
    ] = m.fs.recovery[0] * (
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
        + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]
    )
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Liq", "H2O"] = 0

    # Propagate brine salinity and flow rate
    m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"] = (
        m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"]
        / (1 - m.fs.recovery[0])
    )
    m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "H2O"] = (
        1 - m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"].value
    )
    m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"] = (
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"]
    )
    m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"] = (
        m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"]
        - m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"]
    )

    # initialize feed pump
    propagate_state(m.fs.s01)
    m.fs.pump_feed.initialize(optarg=optarg, solver="ipopt-watertap", outlvl=output_level)

    # initialize separator
    propagate_state(m.fs.s02)
    # Touch property for initialization
    m.fs.separator_feed.mixed_state[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"].fix(
        m.fs.recovery[0].value
    )
    m.fs.separator_feed.mixed_state.initialize(optarg=optarg, solver="ipopt-watertap",  outlvl=output_level)
    # Touch properties for initialization
    m.fs.separator_feed.hx_brine_cold_state[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.separator_feed.hx_distillate_cold_state[0].mass_frac_phase_comp["Liq", "TDS"]
    m.fs.separator_feed.initialize(optarg=optarg, solver="ipopt-watertap", outlvl=output_level)
    m.fs.separator_feed.split_fraction[0, "hx_distillate_cold"].unfix()

    # initialize distillate heat exchanger
    propagate_state(m.fs.s03)
    m.fs.hx_distillate.cold_outlet.temperature[0] = (
        m.fs.evaporator.inlet_feed.temperature[0].value
    )
    m.fs.hx_distillate.cold_outlet.pressure[0] = m.fs.evaporator.inlet_feed.pressure[
        0
    ].value
    m.fs.hx_distillate.hot_inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = (
        m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value
    )
    m.fs.hx_distillate.hot_inlet.flow_mass_phase_comp[0, "Liq", "TDS"] = 1e-4
    m.fs.hx_distillate.hot_inlet.temperature[0] = (
        m.fs.evaporator.outlet_brine.temperature[0].value
    )
    m.fs.hx_distillate.hot_inlet.pressure[0] = 101325
    m.fs.hx_distillate.initialize(solver="ipopt-watertap",  outlvl=output_level)

    # initialize brine heat exchanger
    propagate_state(m.fs.s04)
    m.fs.hx_brine.cold_outlet.temperature[0] = m.fs.evaporator.inlet_feed.temperature[
        0
    ].value
    m.fs.hx_brine.cold_outlet.pressure[0] = m.fs.evaporator.inlet_feed.pressure[0].value
    m.fs.hx_brine.hot_inlet.flow_mass_phase_comp[0, "Liq", "H2O"] = (
        m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "H2O"]
    )
    m.fs.hx_brine.hot_inlet.flow_mass_phase_comp[0, "Liq", "TDS"] = (
        m.fs.evaporator.properties_brine[0].flow_mass_phase_comp["Liq", "TDS"]
    )
    m.fs.hx_brine.hot_inlet.temperature[0] = m.fs.evaporator.outlet_brine.temperature[
        0
    ].value
    m.fs.hx_brine.hot_inlet.pressure[0] = 101325
    m.fs.hx_brine.initialize(solver="ipopt-watertap",  outlvl=output_level)

    # initialize mixer
    propagate_state(m.fs.s05)
    propagate_state(m.fs.s06)
    m.fs.mixer_feed.initialize(solver="ipopt-watertap", outlvl=output_level)
    m.fs.mixer_feed.pressure_equality_constraints[0, 2].deactivate()

    # initialize evaporator
    propagate_state(m.fs.s07)
    m.fs.Q_ext[0].fix()
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].fix()
    # fixes and unfixes those values
    m.fs.evaporator.initialize(delta_temperature_in=60, solver="ipopt-watertap", outlvl=output_level)
    m.fs.Q_ext[0].unfix()
    m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].unfix()

    # initialize compressor
    propagate_state(m.fs.s08)
    m.fs.compressor.initialize(solver="ipopt-watertap", outlvl=output_level)

    # initialize condenser
    propagate_state(m.fs.s09)
    m.fs.condenser.initialize(
        heat=-m.fs.evaporator.heat_transfer.value, solver="ipopt-watertap", outlvl=output_level
    )

    # initialize brine pump
    propagate_state(m.fs.s10)
    m.fs.pump_brine.initialize(optarg=optarg, solver="ipopt-watertap",  outlvl=output_level)

    # initialize distillate pump
    propagate_state(m.fs.s13)  # to translator block
    propagate_state(m.fs.s14)  # from translator block to pump
    m.fs.pump_distillate.control_volume.properties_in[0].temperature = (
        m.fs.condenser.control_volume.properties_out[0].temperature.value
    )
    m.fs.pump_distillate.control_volume.properties_in[0].pressure = (
        m.fs.condenser.control_volume.properties_out[0].pressure.value
    )
    m.fs.pump_distillate.initialize(optarg=optarg, solver="ipopt-watertap",  outlvl=output_level)

    # propagate brine state
    propagate_state(m.fs.s12)
    propagate_state(m.fs.s16)

    seq = SequentialDecomposition(tear_solver="cbc") #cbc
    seq.options.log_info = False
    seq.options.iterLim = 5

    def func_initialize(unit):
        if unit.local_name == "feed":
            pass
        elif unit.local_name == "condenser":
            unit.initialize(
                heat=-unit.flowsheet().evaporator.heat_transfer.value,
                optarg=solver.options,
                solver="ipopt-watertap",
                outlvl=output_level
            )
        elif unit.local_name == "evaporator":
            unit.flowsheet().Q_ext[0].fix()
            unit.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].fix()
            unit.initialize(delta_temperature_in=60, solver="ipopt-watertap",  outlvl=output_level)
            unit.flowsheet().Q_ext[0].unfix()
            unit.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].unfix()
        elif unit.local_name == "separator_feed":
            unit.split_fraction[0, "hx_distillate_cold"].fix(
                unit.flowsheet().recovery[0].value
            )
            unit.initialize(solver="ipopt-watertap",  outlvl=output_level)
            unit.split_fraction[0, "hx_distillate_cold"].unfix()
        elif unit.local_name == "mixer_feed":
            unit.initialize(solver="ipopt-watertap",  outlvl=output_level)
            unit.pressure_equality_constraints[0, 2].deactivate()
        else:
            unit.initialize(solver="ipopt-watertap",  outlvl=output_level)

    seq.run(m, func_initialize)

    m.fs.costing.initialize()

    results = solver.solve(m, tee=False)
    print('Initialization termination condition: ', results.solver.termination_condition)

def fix_outlet_pressures(m):
    # Only fixing the brine outlet pressure to solve for the brine pump head
    # Distillate outlet pressure remains unfixed so there is not an implicit upper bound on the compressed vapor pressure

    # Unfix brine pump head
    m.fs.pump_brine.control_volume.deltaP[0].unfix()

    # Fix brine outlet pressure
    m.fs.brine.properties[0].pressure.fix(101325)
    return

def calculate_cost_sf(cost):
    sf = 10 ** -(math.log10(abs(cost.value)))
    iscale.set_scaling_factor(cost, sf)

def scale_costs(m):
    calculate_cost_sf(m.fs.hx_distillate.costing.capital_cost)
    calculate_cost_sf(m.fs.hx_brine.costing.capital_cost)
    calculate_cost_sf(m.fs.mixer_feed.costing.capital_cost)
    calculate_cost_sf(m.fs.evaporator.costing.capital_cost)
    calculate_cost_sf(m.fs.compressor.costing.capital_cost)
    calculate_cost_sf(m.fs.costing.aggregate_capital_cost)
    calculate_cost_sf(m.fs.costing.aggregate_flow_costs["electricity"])
    calculate_cost_sf(m.fs.costing.total_capital_cost)
    calculate_cost_sf(m.fs.costing.total_operating_cost)

    iscale.calculate_scaling_factors(m)

    print("Scaled costs")

def solve(model, solver=None, tee=False, raise_on_failure=False):
    # ---solving---
    if solver is None:
        solver = get_solver()

    results = solver.solve(model, tee=tee)
    if check_optimal_termination(results):
        return results
    msg = (
        "The current configuration is infeasible. Please adjust the decision variables."
    )
    if raise_on_failure:
        raise RuntimeError(msg)
    else:
        print(msg)
        return results

def set_up_optimization(m):
    m.fs.objective = Objective(expr=m.fs.costing.LCOW)
    m.fs.Q_ext[0].fix(0)
    m.fs.evaporator.area.unfix()
    m.fs.evaporator.outlet_brine.temperature[0].unfix()
    m.fs.compressor.pressure_ratio.unfix()
    m.fs.hx_distillate.area.unfix()
    m.fs.hx_brine.area.unfix()

    print("DOF for optimization: ", degrees_of_freedom(m))

def display_demo(m):
    print(
        "Levelized cost of water:                  %.2f $/m3" % value(m.fs.costing.LCOW)
    )
    print(
        "Evaporator (brine, vapor) temperature:    %.2f C"
        % (m.fs.evaporator.properties_brine[0].temperature.value - 273.15)
    )
    print(
        "Evaporator material factor:               %.2f " % m.fs.costing.evaporator.material_factor_cost.value
    )
    if m.fs.find_component('potential_difference') is not None:
        print(
            "Dissolved oxygen:                         %.2f mg/L"
            % m.fs.dissolved_oxygen_index[0].value
        )
        print(
            "Potential difference:                     %.4f V"
            % m.fs.potential_difference.value
        )


def display_metrics(m):
    print("\nSystem metrics")
    print(
        "Feed flow rate:                           %.2f kg/s"
        % (
            m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value
            + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
        )
    )
    print(
        "Feed salinity:                            %.2f g/kg"
        % (m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].value * 1e3)
    )
    print(
        "Brine salinity:                           %.2f g/kg"
        % (
            m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"].value
            * 1e3
        )
    )
    print(
        "Product flow rate:                        %.2f kg/s"
        % m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value
    )
    print(
        "Recovery:                                 %.2f %%"
        % (m.fs.recovery[0].value * 100)
    )
    print(
        "Specific energy consumption:              %.2f kWh/m3"
        % value(m.fs.costing.specific_energy_consumption)
    )
    print(
        "Levelized cost of water:                  %.2f $/m3" % value(m.fs.costing.LCOW)
    )
    print(
        "External Q:                               %.2f W" % m.fs.Q_ext[0].value
    )  # should be 0 for optimization

def display_design(m):
    print("\nState variables")
    print(
        "Preheated feed temperature:               %.2f K, %.2f C"
        % (m.fs.evaporator.properties_feed[0].temperature.value, m.fs.evaporator.properties_feed[0].temperature.value-273.15)
    )
    print(
        "Evaporator (brine, vapor) temperature:    %.2f K, %.2f C"
        % (m.fs.evaporator.properties_brine[0].temperature.value, m.fs.evaporator.properties_brine[0].temperature.value-273.15)
    )
    print(
        "Evaporator (brine, vapor) pressure:       %.2f kPa"
        % (m.fs.evaporator.properties_vapor[0].pressure.value * 1e-3)
    )
    print(
        "Compressed vapor temperature:             %.2f K, %.2f C"
        % (m.fs.compressor.control_volume.properties_out[0].temperature.value, m.fs.compressor.control_volume.properties_out[0].temperature.value-273.15)
    )
    print(
        "Compressed vapor pressure:                %.2f kPa"
        % (m.fs.compressor.control_volume.properties_out[0].pressure.value * 1e-3)
    )
    print(
        "Condensed vapor temperature:              %.2f K, %.2f C"
        % (m.fs.condenser.control_volume.properties_out[0].temperature.value, m.fs.condenser.control_volume.properties_out[0].temperature.value - 273.15)
    )

    print("\nDesign variables")
    print(
        "Brine heat exchanger area:                %.2f m2" % m.fs.hx_brine.area.value
    )
    print(
        "Distillate heat exchanger area:           %.2f m2"
        % m.fs.hx_distillate.area.value
    )
    print(
        "Compressor pressure ratio:                %.2f"
        % m.fs.compressor.pressure_ratio.value
    )
    print(
        "Evaporator area:                          %.2f m2" % m.fs.evaporator.area.value
    )
    print(
        "Evaporator LMTD:                          %.2f K" % m.fs.evaporator.lmtd.value
    )
    print(
        "Evaporator material factor:               %.2f " % m.fs.costing.evaporator.material_factor_cost.value
    )

def display_corrosion(m):
    print('\nCorrosion results')
    print(f'Material:                            {m.fs.material.value}')
    print(
        "Corrosion rate:                          %.4f mm/yr"
        % m.fs.corrosion_rate.value
    )
    print(
        "Corrosion rate (surrogate input):        %.4f mm/yr"
        % m.fs.corrosion_rate_indexed[0].value
    )
    print(
        "Potential difference:                    %.4f V"
        % m.fs.potential_difference.value
    )
    print(
        "Potential difference (surrogate input):  %.4f V"
        % m.fs.potential_difference_indexed[0].value
    )
    print(
        "Dissolved oxygen:                        %.2f mg/L"
        % m.fs.dissolved_oxygen_index[0].value
    )
    print(
        "pH:                                      %.2f"
        % m.fs.pH_index[0].value
    )
    print(
        "Brine salinity (surrogate input):        %.3f kg/kg"
        % m.fs.brine_salinity_indexed[0].value
    )

def build_results_dict():
    res_dict = {}
    res_dict['Feed salinity'] = []
    res_dict['Recovery'] = []
    res_dict['Material'] = []
    res_dict['Evaporator temperature'] = []
    res_dict['Feed flow rate'] = []
    res_dict['Brine salinity'] = []
    res_dict['Product flow rate'] = []
    res_dict['SEC'] = []
    res_dict['LCOW'] = []
    res_dict['External Q'] = []
    res_dict['Preheated feed temperature'] = []
    res_dict['Evaporator vapor pressure'] = []
    res_dict['Compressed vapor temperature'] = []
    res_dict['Compressed vapor pressure'] = []
    res_dict['Condensed vapor temperature'] = []
    res_dict['Compressor pressure ratio'] = []
    res_dict['Evaporator area'] = []
    res_dict['Evaporator LMTD'] = []
    res_dict['Evaporator material factor'] = []
    res_dict['Corrosion rate'] = []
    res_dict['Potential difference'] = []
    res_dict['Dissolved oxygen']= []
    res_dict['pH'] = []
    res_dict['capex_opex_ratio'] = []
    res_dict['Termination condition'] = []

    return res_dict

def save_single_run(m, mat, wf, rr, do, ph, results_folder='results',save_name=None,results=None):
    # save_dir = f"analysis_waterTAP/analysisWaterTAP/analysis_scripts/mvc_corrosion/{results_folder}/"
    save_dir = f"src/paper_figures/{results_folder}/"
    if save_name is not None:
        filename = save_dir + save_name
    elif 'sensitivity' in results_folder:
        filename = save_dir + f'{mat}_wf_{wf}_rr_{rr}_ph_{ph}_do_{do}.csv'
    elif results_folder == 'results_dissolved_oxygen':
        filename = save_dir + f'wf_{wf}_rr_{rr}_ph_{ph}/single_runs/{mat}_wf_{wf}_rr_{rr}_do_{do}_ph_{ph}.csv'
    else:
        filename = save_dir + f'do_{do}_ph_{ph}/single_cases/{mat}_wf_{wf}_rr_{rr}.csv'

    results_dict = single_run_results_dict(m, results)
    df = pd.DataFrame(results_dict, index=[0])
    df.to_csv(filename, index=False)

def single_run_results_dict(m, results=None):
    res_dict = {}
    res_dict['Feed salinity'] = m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].value*1000
    res_dict['Feed mass fraction'] = m.fs.feed.properties[0].mass_frac_phase_comp["Liq", "TDS"].value
    res_dict['Recovery'] = m.fs.recovery[0].value
    res_dict['Material'] = m.fs.material.value
    res_dict['Evaporator temperature'] = m.fs.evaporator.properties_brine[0].temperature.value
    res_dict['Feed flow rate'] = m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "H2O"].value + m.fs.feed.properties[0].flow_mass_phase_comp["Liq", "TDS"].value
    res_dict['Brine salinity'] = m.fs.evaporator.properties_brine[0].mass_frac_phase_comp["Liq", "TDS"].value * 1e3
    res_dict['Product flow rate'] = m.fs.evaporator.properties_vapor[0].flow_mass_phase_comp["Vap", "H2O"].value
    res_dict['SEC']=value(m.fs.costing.specific_energy_consumption)
    res_dict['LCOW']=value(m.fs.costing.LCOW)
    res_dict['External Q']=value(m.fs.Q_ext[0])
    res_dict['Preheated feed temperature'] = m.fs.evaporator.properties_feed[0].temperature.value
    res_dict['Evaporator vapor pressure']=m.fs.evaporator.properties_vapor[0].pressure.value
    res_dict['Compressed vapor temperature']=m.fs.compressor.control_volume.properties_out[0].temperature.value
    res_dict['Compressed vapor pressure']=m.fs.compressor.control_volume.properties_out[0].pressure.value
    res_dict['Condensed vapor temperature']=m.fs.condenser.control_volume.properties_out[0].temperature.value
    res_dict['Distillate hx outlet temperature'] = value(m.fs.hx_distillate.hot_outlet.temperature[0])
    res_dict['Distillate hx area'] = value(m.fs.hx_distillate.area)
    res_dict['Distillate overall heat transfer coefficient'] = value(m.fs.hx_distillate.overall_heat_transfer_coefficient[0])
    res_dict['Brine hx outlet temperature'] = value(m.fs.hx_brine.hot_outlet.temperature[0])
    res_dict['Brine hx area'] = value(m.fs.hx_brine.area)
    res_dict['Brine overall heat transfer coefficient'] = value(m.fs.hx_brine.overall_heat_transfer_coefficient[0])
    res_dict['Compressor pressure ratio'] = value(m.fs.compressor.pressure_ratio)
    res_dict['Compressor efficiency'] = value(m.fs.compressor.efficiency)
    res_dict['Compressor cost'] = value(m.fs.costing.compressor.unit_cost)
    res_dict['Evaporator area'] = value(m.fs.evaporator.area)
    res_dict['Evaporator LMTD'] = value(m.fs.evaporator.lmtd)
    res_dict['Evaporator overall heat transfer coefficient'] = value(m.fs.evaporator.U)
    res_dict['Evaporator material factor']=m.fs.costing.evaporator.material_factor_cost.value
    res_dict['Corrosion rate'] = m.fs.corrosion_rate.value
    res_dict['Potential difference'] =m.fs.potential_difference.value
    res_dict['Dissolved oxygen'] = m.fs.dissolved_oxygen_index[0].value
    res_dict['pH']= m.fs.pH_index[0].value
    res_dict['capex_opex_ratio']=value(m.fs.costing.LCOW_percentage['capex_opex_ratio'])
    res_dict['CAPEX percentage feed pump'] = value(m.fs.costing.MVC_capital_cost_percentage["feed_pump"])
    res_dict['CAPEX percentage distillate pump'] = value(m.fs.costing.MVC_capital_cost_percentage["distillate_pump"])
    res_dict['CAPEX percentage brine pump'] = value(m.fs.costing.MVC_capital_cost_percentage["brine_pump"])
    res_dict['CAPEX percentage distillate hx'] = value(m.fs.costing.MVC_capital_cost_percentage["hx_distillate"])
    res_dict['CAPEX percentage brine hx'] = value(m.fs.costing.MVC_capital_cost_percentage["hx_brine"])
    res_dict['CAPEX percentage mixer'] = value(m.fs.costing.MVC_capital_cost_percentage["mixer"])
    res_dict['CAPEX percentage evaporator'] = value(m.fs.costing.MVC_capital_cost_percentage["evaporator"])
    res_dict['CAPEX percentage compressor'] = value(m.fs.costing.MVC_capital_cost_percentage["compressor"])
    res_dict['Annual operating costs'] = value(m.fs.costing.annual_operating_costs)
    res_dict['LCOW percentage feed pump'] = value(m.fs.costing.LCOW_percentage["feed_pump"])
    res_dict['LCOW percentage distillate pump'] = value(m.fs.costing.LCOW_percentage["distillate_pump"])
    res_dict['LCOW percentage brine pump'] = value(m.fs.costing.LCOW_percentage["brine_pump"])
    res_dict['LCOW percentage distillate hx'] = value(m.fs.costing.LCOW_percentage["hx_distillate"])
    res_dict['LCOW percentage brine hx'] = value(m.fs.costing.LCOW_percentage["hx_brine"])
    res_dict['LCOW percentage mixer'] = value(m.fs.costing.LCOW_percentage["mixer"])
    res_dict['LCOW percentage evaporator'] = value(m.fs.costing.LCOW_percentage["evaporator"])
    res_dict['LCOW percentage compressor'] = value(m.fs.costing.LCOW_percentage["compressor"])
    res_dict['LCOW percentage electricity'] = value(m.fs.costing.LCOW_percentage['electricity'])
    res_dict['LCOW percentage MLC'] = value(m.fs.costing.LCOW_percentage['MLC'])
    res_dict['LCOW percentage CAPEX'] = value(m.fs.costing.LCOW_percentage["capital_costs"])
    res_dict['LCOW percentage OPEX'] = value(m.fs.costing.LCOW_percentage["operating_costs"])
    if results is not None:
        res_dict['Termination condition'] = results.solver.termination_condition.value
    return res_dict


if __name__ == "__main__":
    ph = 7.5
    do = 0.5
    wf = 35
    rr = 0.78
    mat = 'Nickel alloy 625'
    m, results = single_run(material=mat, wf=wf, rr=rr, do=do, ph=ph, save=False)
    display_metrics(m)
    display_design(m)
    display_corrosion(m)