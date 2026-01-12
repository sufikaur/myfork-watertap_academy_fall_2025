from pyomo.environ import Reals
from idaes.models.unit_models import Feed, Separator
from watertap.unit_models.reverse_osmosis_0D import (
    ReverseOsmosis0D as RO,
    ConcentrationPolarizationType,
    MassTransferCoefficient,
)
from watertap.property_models import seawater_prop_pack as props
from watertap.core.solvers import get_solver


def load_data(full_data):
    # # Load data
    psi_to_pascal = 6894.75  # Pressure conversion
    gpm_to_m3ps = 6.309e-005  # Vol. flow conversion

    dt = full_data[
        [
            "Time",
            "RO Overall Permeate Flowrate (gpm)",
            "RO Overall Concentrate Flowrate (gpm)",
            "Concentrate Recirculation Flowrate Set Point (gpm)",
            "CF Effluent EC (uS/cm)",
            "CF Effluent EC (ppm)",
            "RO Overall Concentrate EC (uS/cm)",
            "RO Overall Concentrate EC (ppm)",
            "RO Overall Permeate EC (uS/cm)",
            "RO Overall Permeate EC (ppm)",
            "RO IN Feedwater Pressure (psi)",
            "RO OUT Concentrate Pressure (psi)",
            "CF Effluent Temperature (C)",
            "Recovery (%)",
        ]
    ]
    dt.rename(
        columns={"Concentrate Recirculation Flowrate Set Point (gpm)": "Qr_gpm"},
        inplace=True,
    )
    # RO pressure drop, psi
    dt["RO_deltaP_psi"] = (
        dt["RO IN Feedwater Pressure (psi)"] - dt["RO OUT Concentrate Pressure (psi)"]
    )
    # Influent feed volumetric flowrate, gpm
    dt["Qp+Qb_gpm"] = (
        dt["RO Overall Permeate Flowrate (gpm)"]
        + dt["RO Overall Concentrate Flowrate (gpm)"]
    )
    # RO feed volumetric flowrate, gpm
    dt["Qr+Qp+Qb_gpm"] = dt["Qp+Qb_gpm"] + dt["Qr_gpm"]

    # RO feed mass frac, ppm
    dt["RO_feed_mass_frac"] = (
        1e-6
        * (
            dt["RO Overall Concentrate EC (ppm)"] * dt["Qr_gpm"]
            + dt["CF Effluent EC (ppm)"] * dt["Qp+Qb_gpm"]
        )
        / dt["Qr+Qp+Qb_gpm"]
    )
    # Column naming from legacy naming by Tim/Mayo:
    dt.rename(
        columns={
            "RO Overall Permeate Flowrate (gpm)": "Qp_gpm",
            "RO Overall Concentrate Flowrate (gpm)": "Qb_gpm",
            "RO Overall Concentrate EC (uS/cm)": "Cb_ec_uS_cm",
            "RO Overall Concentrate EC (ppm)": "Cb_ppm",
            "RO Overall Permeate EC (uS/cm)": "Cp_ec_uS_cm",
            "RO Overall Permeate EC (ppm)": "Cp_ppm",
            "RO IN Feedwater Pressure (psi)": "Pf_psi",
            "RO OUT Concentrate Pressure (psi)": "Pb_psi",
            "Recovery (%)": "system_recovery_%",
        },
        inplace=True,
    )
    dt["RO_recovery_%"] = dt["Qp_gpm"] / dt["Qr+Qp+Qb_gpm"]
    # dt["pressure_in"] = dt["Pf_psi"] * psi_to_pascal
    # dt["flow_vol_permeate"] = dt["Qp_gpm"] * gpm_to_m3ps
    dt["mass_frac_TDS_permeate"] = dt["Cp_ppm"] * 1e-6

    dt.rename(
        columns={
            "Qr+Qp+Qb_gpm": "flow_vol_in",
            "Pf_psi": "pressure_in",
            "RO_feed_mass_frac": "mass_frac_TDS_in",
            "RO_deltaP_psi": "deltaP",
            "Qp_gpm": "flow_vol_permeate",
            "Cp_ppm": "conc_frac_TDS_permeate",
        },
        inplace=True,
    )

    subset_cols = [
        "flow_vol_in",
        "mass_frac_TDS_in",
        "pressure_in",
        "deltaP",
        "flow_vol_permeate",
        # "conc_frac_TDS_permeate",
        "mass_frac_TDS_permeate",
    ]
    dt = dt[subset_cols]

    full_data = dt.iloc[11:61]

    data = dt.iloc[31:41]

    return data, full_data
