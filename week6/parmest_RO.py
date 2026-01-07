import logging
from io import StringIO
from numpy import dtype
import pandas as pd
import os
from pyomo.environ import ConcreteModel, TerminationCondition, \
    value, Constraint, Set, Param, Var, Reals, Block, Objective, TransformationFactory, assert_optimal_termination
from pyomo.common.log import LoggingIntercept
from pyomo.util.infeasible import (log_active_constraints, log_close_to_bounds,
                               log_infeasible_bounds,
                               log_infeasible_constraints)
from pyomo.util.check_units import assert_units_consistent, assert_units_equivalent, check_units_equivalent
import pyomo.contrib.parmest.parmest as parmest
from pyomo.network import Arc
import pyomo.util.infeasible as infeas


from idaes.core.util.initialization import propagate_state
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.models.unit_models import Feed, Separator
from idaes.core import (FlowsheetBlock)
from idaes.core.util.model_statistics import (degrees_of_freedom,
                                            number_variables,
                                            number_total_constraints,
                                            fixed_variables_set,
                                            activated_constraints_set,
                                            number_unused_variables)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

from watertap.unit_models.reverse_osmosis_0D import ReverseOsmosis0D as RO, ConcentrationPolarizationType, MassTransferCoefficient
from watertap.property_models import seawater_prop_pack as props
from watertap.core.solvers import get_solver


# ##############################################
# ParmEst for RO Parameter Estimation
# ##############################################


solver= get_solver()
    
# Conversion factors
psi_to_pascal = 6894.75 # Pressure conversion
gpm_to_m3ps = 6.309e-005 # Vol. flow conversion
mS_to_massfrac = (0.67 * 1000) / 1e6 # Conductivity to mass fraction for seawater based on SB data provided.
pressure_atmospheric = 101325
uS_to_massfrac = 0.67 / 1e6
flow_vol_scale =  1/1.909522782100071e-05 #14318 # stdev of permeate flow #  1e2
mass_comp_scale = 1/2.4388414228794023e-06 #379457 # stdev of permeate salinity #  1e4 

CP = ConcentrationPolarizationType.none
MT = MassTransferCoefficient.none


def ro_parmest(data):
    # Define concrete model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic= False)
    m.fs.properties = props.SeawaterParameterBlock()
    m.fs.feed = Feed(property_package= m.fs.properties)
    m.fs.RO = RO(property_package=m.fs.properties,
                 has_pressure_change=True,
                 concentration_polarization_type=CP,
                 mass_transfer_coefficient=MT)

    m.fs.s00 = Arc(source=m.fs.feed.outlet, destination=m.fs.RO.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Initialize model
    m.fs.feed.properties[0].flow_vol_phase.fix(gpm_to_m3ps * 8)
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].pressure.fix(psi_to_pascal * 188)
    m.fs.feed.properties[0].mass_frac_phase_comp['Liq', 'TDS'].fix(0.001)

    pressure_atmospheric = 101325
    m.fs.RO.area.fix(28.8) # Data is for 4 membrane elements
    m.fs.RO.permeate.pressure[0].fix(pressure_atmospheric)
    m.fs.RO.deltaP.fix(-psi_to_pascal * 24.6)

    if m.fs.RO.config.concentration_polarization_type == ConcentrationPolarizationType.calculated:
        m.fs.RO.feed_side.spacer_porosity.fix(0.85)
        m.fs.RO.feed_side.channel_height.fix(0.0008636)
        m.fs.RO.recovery_vol_phase[0, 'Liq'].fix(float(data['RO_recovery_%'])/100)
    
    # Initial A and B values
    m.fs.RO.A_comp[0, 'H2O'].fix(5e-12)
    m.fs.RO.B_comp[0, 'TDS'].fix(4e-8)
    

    # Set scaling factors
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e1, index=('Liq','H2O'))
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e6, index=('Liq', 'TDS'))
    iscale.calculate_scaling_factors(m)

    # Initialize problem by: (1) solving properties_in, (2) initializing properties_out and (3) solving properties_permeate
    m.fs.RO.feed_side.properties_in[0].flow_mass_phase_comp['Liq', 'H2O'].setub(1000) # Bounds on flow rate too low!
    m.fs.RO.feed_side.properties_out[0].flow_mass_phase_comp['Liq', 'H2O'].setub(1000)

    # Initialize
    res = solver.solve(m.fs.feed)
    assert_optimal_termination(res)

    propagate_state(m.fs.s00)
    m.fs.RO.initialize()
    
    # Fix at actual conditions
    m.fs.feed.properties[0].flow_vol_phase.fix(gpm_to_m3ps * float(data['Qr+Qp+Qb_gpm']))
    m.fs.feed.properties[0].pressure.fix(psi_to_pascal * float(data['Pf_psi']))
    m.fs.feed.properties[0].mass_frac_phase_comp['Liq', 'TDS'].fix(float(data['RO_feed_mass_frac']))
    m.fs.RO.deltaP.fix(-psi_to_pascal * float(data['RO_deltaP_psi']))

    assert degrees_of_freedom(m) == 0

    return m


def SSE(m, data):
    expr = (
         10*flow_vol_scale*((gpm_to_m3ps*float(data['Qp_gpm']) - m.fs.RO.mixed_permeate[0.0].flow_vol_phase['Liq'])**2) 
         + 1000*mass_comp_scale*((1e-6*float(data['Cp_ppm'])- m.fs.RO.mixed_permeate[0.0].mass_frac_phase_comp['Liq', 'TDS'])**2)
            )
    return expr


def sb_ro_runsystem(data, params):
    # Define concrete model
    m = ConcreteModel()
    m.fs = FlowsheetBlock(dynamic=False)
    m.fs.properties = props.SeawaterParameterBlock()
    m.fs.feed = Feed(property_package=m.fs.properties)
    m.fs.RO = RO(property_package=m.fs.properties,
                 has_pressure_change=True,
                 concentration_polarization_type=CP,
                 mass_transfer_coefficient=MT)


    m.fs.s00 = Arc(source=m.fs.feed.outlet, destination=m.fs.RO.inlet)
    TransformationFactory("network.expand_arcs").apply_to(m)

    # Initialize model
    m.fs.feed.properties[0].flow_vol_phase.fix(gpm_to_m3ps * float(data['Qr+Qp+Qb_gpm']))
    m.fs.feed.properties[0].temperature.fix(273.15 + 25)
    m.fs.feed.properties[0].pressure.fix(psi_to_pascal * float(data['Pf_psi']))
    m.fs.feed.properties[0].mass_frac_phase_comp['Liq', 'TDS'].fix(float(data['RO_feed_mass_frac']))


    m.fs.RO.area.fix(28.8)
    m.fs.RO.permeate.pressure[0].fix(pressure_atmospheric)
    m.fs.RO.deltaP.fix(-psi_to_pascal * float(data['RO_deltaP_psi']))
    m.fs.RO.A_comp.fix(params[0])
    m.fs.RO.B_comp.fix(params[1])
    if m.fs.RO.config.concentration_polarization_type == ConcentrationPolarizationType.calculated:
        m.fs.RO.feed_side.spacer_porosity.fix(0.85)
        m.fs.RO.feed_side.channel_height.fix(0.008636)
        m.fs.RO.recovery_vol_phase[0, 'Liq'].fix(float(data['RO_recovery_%'])/100)
    
    # Provide upper bounds and set scaling factors
    m.fs.RO.feed_side.properties_in[0].flow_mass_phase_comp['Liq', 'H2O'].setub(1000) # Bounds on flow rate too low!
    m.fs.RO.feed_side.properties_out[0].flow_mass_phase_comp['Liq', 'H2O'].setub(1000)
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e1, index=('Liq','H2O'))
    m.fs.properties.set_default_scaling('flow_mass_phase_comp', 1e6, index=('Liq', 'TDS'))
    iscale.calculate_scaling_factors(m)
    assert degrees_of_freedom(m) == 0

    # Initialize
    res = solver.solve(m.fs.feed)
    assert res.solver.termination_condition == TerminationCondition.optimal
    propagate_state(m.fs.s00)
    m.fs.RO.initialize(optarg=solver.options)


    results = solver.solve(m, tee=False)

    assert_optimal_termination(results)

    return m

def load_data(fpath):
    # # Load data   
    full_data = pd.read_excel(fpath, 
                            #   sheet_name="Clean Data"
                              )

    dt = full_data[[
        'Time',
        'RO Overall Permeate Flowrate (gpm)',
        'RO Overall Concentrate Flowrate (gpm)',
        'Concentrate Recirculation Flowrate Set Point (gpm)',
        'CF Effluent EC (uS/cm)',
        'CF Effluent EC (ppm)',
        'RO Overall Concentrate EC (uS/cm)',
        'RO Overall Concentrate EC (ppm)',
        'RO Overall Permeate EC (uS/cm)',
        'RO Overall Permeate EC (ppm)',
        'RO IN Feedwater Pressure (psi)',
        'RO OUT Concentrate Pressure (psi)',
        'CF Effluent Temperature (C)',
        'Recovery (%)',
        ]]
    dt.rename(columns={'Concentrate Recirculation Flowrate Set Point (gpm)': "Qr_gpm"}, inplace=True)
    # RO pressure drop, psi
    dt['RO_deltaP_psi'] = dt['RO IN Feedwater Pressure (psi)'] - dt['RO OUT Concentrate Pressure (psi)']
    #Influent feed volumetric flowrate, gpm
    dt['Qp+Qb_gpm'] = dt['RO Overall Permeate Flowrate (gpm)'] + dt['RO Overall Concentrate Flowrate (gpm)']
    #RO feed volumetric flowrate, gpm
    dt['Qr+Qp+Qb_gpm'] = dt['Qp+Qb_gpm'] + dt['Qr_gpm']
    
    
    # RO feed mass frac, ppm
    dt['RO_feed_mass_frac'] = (1e-6*(dt['RO Overall Concentrate EC (ppm)']*dt['Qr_gpm'] + dt['CF Effluent EC (ppm)']*dt['Qp+Qb_gpm'])
                               / dt['Qr+Qp+Qb_gpm']) 
    # Column naming from legacy naming by Tim/Mayo:
    dt.rename(columns = {
        'RO Overall Permeate Flowrate (gpm)': "Qp_gpm",
        'RO Overall Concentrate Flowrate (gpm)': "Qb_gpm",
        'RO Overall Concentrate EC (uS/cm)': "Cb_ec_uS_cm",
        'RO Overall Concentrate EC (ppm)': "Cb_ppm",
        'RO Overall Permeate EC (uS/cm)': "Cp_ec_uS_cm",
        'RO Overall Permeate EC (ppm)': "Cp_ppm",
        'RO IN Feedwater Pressure (psi)': "Pf_psi",
        'RO OUT Concentrate Pressure (psi)': "Pb_psi",
        'Recovery (%)': 'system_recovery_%',
        

              }, inplace=True)
    dt['RO_recovery_%'] = dt['Qp_gpm'] / dt['Qr+Qp+Qb_gpm']
        
    data = dt
    return data, full_data

def estimate_ro_membrane_parameters(data,tee=True):
    # Create a list of vars to estimate
    variable_names = ["fs.RO.A_comp[0, 'H2O']", "fs.RO.B_comp[0, 'TDS']"]

    # # Initialize a parameter estimation object

    pest = parmest.Estimator(ro_parmest, data, variable_names, SSE, tee=tee)

    # # Run parameter estimation using all data
    obj_value, parameters = pest.theta_est()

    # Print results 
    print("The SSE at the optimal solution is %0.6f" % (obj_value))
    print("\nThe values for the parameters are as follows:")
    for k,v in parameters.items():
        print(k, "=", v)
    
    return pest, obj_value, parameters

def evaluate_runs(data, parameters, startrow=None, prefix=""):
    if startrow is None:
        startrow=0
    # Evaluate individual runs with estimated parameters and document results in a dataframe 
    res_df = pd.DataFrame(columns=['Time','Aw', 'B','Permeate FR - actual', 'Permeate FR - sim', 'Permeate mass frac - actual', 'Permeate mass frac - sim'])
    for i in range(startrow, startrow+data.shape[0]):
        test_data = {'Time':data['Time'][i] , 
                     'Aw': parameters[0],'B':parameters[1],'Pf_psi':data['Pf_psi'][i], 'Qr+Qp+Qb_gpm':data['Qr+Qp+Qb_gpm'][i], 'RO_feed_mass_frac':data['RO_feed_mass_frac'][i],\
                  'RO_deltaP_psi':data['RO_deltaP_psi'][i], 'RO_recovery_%':data['RO_recovery_%'][i]}
        m = sb_ro_runsystem(test_data, list(parameters))
        res_df = pd.concat([res_df,pd.DataFrame([{'Time': data['Time'][i],'Aw': parameters[0],'B':parameters[1],'Permeate FR - actual': gpm_to_m3ps*data['Qp_gpm'][i],
                                'Permeate FR - sim': value(m.fs.RO.mixed_permeate[0.0].flow_vol_phase['Liq']),
                                'Pct Error Permeate Flow': (value(m.fs.RO.mixed_permeate[0.0].flow_vol_phase['Liq'])-gpm_to_m3ps*data['Qp_gpm'][i])/(gpm_to_m3ps*data['Qp_gpm'][i])*100,
                                'Permeate mass frac - actual': data['Cp_ppm'][i]*1e-6,
                                'Permeate mass frac - sim': m.fs.RO.mixed_permeate[0.0].mass_frac_phase_comp['Liq', 'TDS'](),
                                'Pct Error Perm mass frac':(value(m.fs.RO.mixed_permeate[0.0].mass_frac_phase_comp['Liq', 'TDS']) -data['Cp_ppm'][i]*1e-6)/(data['Cp_ppm'][i]*1e-6)*100
                                }])],      ignore_index = True)
    
        print("\n'Pct Error Permeate Flow':", (value(m.fs.RO.mixed_permeate[0.0].flow_vol_phase['Liq'])-gpm_to_m3ps*data['Qp_gpm'][i])/(gpm_to_m3ps*data['Qp_gpm'][i])*100)
        print("'Pct Error Perm mass frac':", (value(m.fs.RO.mixed_permeate[0.0].mass_frac_phase_comp['Liq', 'TDS']) -data['Cp_ppm'][i]*1e-6)/(data['Cp_ppm'][i]*1e-6)*100)

    
    if CP == ConcentrationPolarizationType.none:
        msg="wo_CP"
    elif CP == ConcentrationPolarizationType.calculated:
        msg="w_CP"
    else:
        raise RuntimeError("Invalid entry for 'CP'")
    fname = f'{prefix}parmest_output_{msg}.xlsx'
    if not os.path.isfile(fname):
         with pd.ExcelWriter(fname) as writer:
            res_df.to_excel(writer,sheet_name='Sheet1', header=True, index=False,startrow=startrow) 

    else:
        
        with pd.ExcelWriter(fname, mode='a', if_sheet_exists="overlay") as writer:
            res_df.to_excel(writer,sheet_name='Sheet1',header=False, index=False,startrow=startrow+1) 

    return m

if __name__ == "__main__":

    data_file=r"Pilot Data_9-10DEC2024_24112523_post_process.xlsx"

    # data_file=r"C:\Users\Adam\Box\WaterTAP (protected by NDA)\nawi (NDA protected)\WaterTAP-5.09 MF-RO-UV\data\UCI_Pilot\raw_operational\Pilot Data_9-10DEC2024_24112523_post_process.xlsx"
    data, full_data = load_data(fpath=data_file)

    # averaged_df = data.drop(columns="Time")
    # averaged_df = averaged_df.groupby(averaged_df.index // 49).mean()
    start = 2*49
    steps = 49
    end = 1000

    # print("end = ", end)
    # assert False
    for start in range(start, end , steps):
        data_chunk = data.iloc[start:start+steps,:]
        try:
            pest, obj_val, params = estimate_ro_membrane_parameters(data_chunk)
            m = evaluate_runs(data_chunk, params, startrow=start, prefix="latest_")
        except:
            continue
    print(" ALL DONE !")
