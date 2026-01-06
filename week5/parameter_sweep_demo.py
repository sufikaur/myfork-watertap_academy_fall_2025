import pandas as pd
import matplotlib.pyplot as plt

from pyomo.environ import value, assert_optimal_termination
from watertap.core.solvers import get_solver

from parameter_sweep import (
    LinearSample,
    ParameterSweep,
)
import RO_with_ERD as ro_erd


def solve(blk, solver=None, tee=False, check_termination=True):

    if solver is None:
        solver = get_solver()
    results = solver.solve(blk, tee=tee)
    if check_termination:
        assert_optimal_termination(results)

    return results


def build_sweep_params(m, num_samples=1, parameter="A_comp"):
    sweep_params = {}

    if parameter == "A_comp":
        sweep_params["A_comp"] = LinearSample(
            m.fs.RO.A_comp, 1.0e-12, 6e-12, num_samples
        )
    elif parameter == "B_comp":
        sweep_params["B_comp"] = LinearSample(
            m.fs.RO.B_comp, 1.0e-8, 8.0e-8, num_samples
        )
    elif parameter == "recovery":
        sweep_params["recovery"] = LinearSample(
            m.fs.RO.recovery_vol_phase[0, "Liq"], 0.1, 0.65, num_samples
        )
    else:
        raise NotImplementedError

    return sweep_params


def build_outputs(m):
    outputs = {}
    # for b in m.fs.costing.component_objects([Var, Expression], descend_into=True):
    #     if b.is_indexed():
    #         # continue
    #         for i, bb in b.items():
    #             if bb.name in outputs.keys():
    #                 continue
    #             outputs[bb.name] = bb
    #     else:
    #         if b.name in outputs.keys():
    #             continue
    #         outputs[b.name] = b
    cols = [
        "fs.costing.LCOW",
        "fs.costing.LCOW_component_direct_capex['fs.pump']",
        "fs.costing.LCOW_component_direct_capex['fs.RO']",
        "fs.costing.LCOW_component_direct_capex['fs.erd']",
        "fs.costing.LCOW_component_indirect_capex['fs.pump']",
        "fs.costing.LCOW_component_indirect_capex['fs.RO']",
        "fs.costing.LCOW_component_indirect_capex['fs.erd']",
        "fs.costing.LCOW_component_fixed_opex['fs.pump']",
        "fs.costing.LCOW_component_fixed_opex['fs.RO']",
        "fs.costing.LCOW_component_fixed_opex['fs.erd']",
        "fs.costing.LCOW_component_variable_opex['fs.pump']",
        "fs.costing.LCOW_component_variable_opex['fs.RO']",
        "fs.costing.LCOW_component_variable_opex['fs.erd']",
    ]
    for c in cols:
        outputs[c] = m.find_component(c)
    return outputs


def param_sweep(parameter="A_comp"):
    m = ro_erd.build()
    ro_erd.scale_system(m)
    ro_erd.add_costing(m)
    ro_erd.initialize_system(m)
    results = ro_erd.solve_system(m)
    if parameter in ["A_comp", "B_comp"]:
        m.fs.RO.recovery_vol_phase.unfix()
        m.fs.RO.length.fix(value(m.fs.RO.length))

    if parameter == "recovery":
        m.fs.pump.control_volume.properties_out[0].pressure.unfix()
        m.fs.RO.length.fix(value(m.fs.RO.length))

    return m


if __name__ == "__main__":

    num_samples = 4
    num_procs = 4

    solver = get_solver()

    for parameter in ["A_comp", "B_comp", "recovery"]:
    # for parameter in ["recovery"]:
        kwargs_dict = {
            "h5_results_file_name": f"test_{parameter}.h5",
            "build_model": param_sweep,
            "build_model_kwargs": {"parameter": parameter},
            "build_sweep_params": build_sweep_params,
            "build_sweep_params_kwargs": {
                "num_samples": num_samples,
                "parameter": parameter,
            },
            "build_outputs": build_outputs,
            "build_outputs_kwargs": {},
            "optimize_function": solve,
            "number_of_subprocesses": num_procs,
            "csv_results_file_name": f"test_{parameter}.csv",  # For storing results as CSV
        }

        ps = ParameterSweep(**kwargs_dict)

        results_array, results_dict = ps.parameter_sweep(
            kwargs_dict["build_model"],
            kwargs_dict["build_sweep_params"],
            build_outputs=kwargs_dict["build_outputs"],
            build_outputs_kwargs=kwargs_dict["build_outputs_kwargs"],
            num_samples=num_samples,
            build_model_kwargs=kwargs_dict["build_model_kwargs"],
            build_sweep_params_kwargs=kwargs_dict["build_sweep_params_kwargs"],
        )

        df = pd.read_csv(f"test_{parameter}.csv")

        stacked_cols = list()
        colors = list()
        hatch = list()
        units = ["fs.RO", "fs.erd", "fs.pump"]

        for u in units:
            stacked_cols.append(
                df[f"fs.costing.LCOW_component_direct_capex['{u}']"]
                + df[f"fs.costing.LCOW_component_indirect_capex['{u}']"]
            )
            stacked_cols.append(df[f"fs.costing.LCOW_component_fixed_opex['{u}']"])
            stacked_cols.append(df[f"fs.costing.LCOW_component_variable_opex['{u}']"])
            colors.extend(["#1f77b4", "#ff7f0e", "#2ca02c"])
            hatch.extend(["..", "//", "xx"])


        x = df[f"# {parameter}"]

        fig, ax = plt.subplots()
        ax.stackplot(
            x,
            stacked_cols,
            colors=colors,
            labels=["RO", "ERD", "Pump"],
            edgecolor="black",
            hatch=hatch,
        )
        ax.plot(
            x, df["fs.costing.LCOW"], color="red", label="Total LCOW", linewidth=2
        )
        ax.legend()
        ax.axhline(0, linewidth=2, color="black")
        ax.set_title(f"Parameter Sweep: {parameter}")
        plt.show()
