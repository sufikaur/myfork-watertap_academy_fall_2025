import ray
import numpy as np
import json
import time
import configparser
import os
import copy
import OLIPy as oli


def oli_api_connect(config):
    username = config['OLIAPI']['username']
    password = config['OLIAPI']['password']
    access_key = config['OLIAPI']['access_key']
    oliapi = oli.OLIApi(username=username, password=password, access_key=access_key)
    if password is None or not password:
        return oliapi
    else:
        if oliapi.login():
            print("Login successful")
            return oliapi
        else:
            print("Login failed")
            quit()

def read_json_from_file(folder_path, file_name):
    file_path = os.path.join(folder_path, file_name)
    print('Filepath: ', file_path)
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error decoding JSON from file: {file_path}")
        return None

def get_chemistry_json_for_dbs(inflows, redox=None):
    formatted_inflows = []
    for inflow in inflows:
        formatted_inflows.append({"name": inflow})
    dbs_data = {
        "method": "chemistrybuilder.generateDBS",
        "params": {
            "thermodynamicFramework": "Aqueous (H+ ion)",
            "modelName": "waterTAPModel",
            "phases": [
                "liquid1",
                "vapor",
                "solid",
                "liquid2"
            ],
            "inflows": formatted_inflows,
            "redox": redox
        }
    }
    return dbs_data

@ray.remote
def process_condition(T, ph, rec, do, chemistry_file_id, base_corrosion_spec,
                      CONTACT_SURFACE, output_folder, original_H2O, original_salts, original_volume):
    try:
        # Each worker re-reads the config and reinitializes its own API connection.
        config = configparser.ConfigParser()
        config.read('corrosion_example/config.ini')
        username = config['OLIAPI']['username']
        password = config['OLIAPI']['password']
        access_key = config['OLIAPI']['access_key']
        oliapi = oli.OLIApi(username=username, password=password) #, access_key=access_key)
        if password and not oliapi.login():
            print(f"Login failed in worker for T={T}, ph={ph}, rec={rec}, do={do}")
            return None

        # Calculate effective H2O (g) using the derived equation:
        effective_H2O = original_H2O * (1 - rec) - original_salts * rec

        # Calculate O2 mass in grams from dissolved oxygen (do in mg/L)
        O2_g = do * original_volume * 1e-3

        # Build flash input for the "setph" API call.
        flash_input = {
            "params": {
                "temperature": {"value": T, "unit": "°C"},
                "pressure": {"value": 1, "unit": "atm"},
                "targetPH": {"value": ph, "unit": ""},
                "pHAcidTitrant": "HCL",
                "pHBaseTitrant": "NAOH",
                "inflows": {
                    # Using absolute mass units ("g") as in your code.
                    "unit": "g",
                    "values": {
                        "H2O": effective_H2O,
                        "B2O3": 0.0146370,
                        "CACO3": 1.97586e-3,
                        "CAO": 0.558576,
                        "CO2": 0.100109,
                        "KCL": 0.724571,
                        "MGCL2": 3.15923,
                        "MGO": 0.755392,
                        "NABR": 0.0837018,
                        "NACL": 26.8241,
                        "SO3": 2.20781,
                        "SRCL2": 0.0235202,
                        "FEEL": 0,
                        "HCL": 0.0,
                        "NAOH": 0.0,
                        "O2": O2_g
                    },
                    # The total feed mass remains the sum of water and salts.
                    "totalAmount": {"value": effective_H2O + original_salts, "unit": "g"}
                },
                "unitSetInfo": {"combined_phs_comp": "g"}
            }
        }

        # Run the flash calculation using the "setph" API call.
        result_flash_calc = oliapi.call("setph", chemistry_file_id, flash_input)

        # Save the flash output.
        flash_filename = os.path.join(output_folder,
            f"flash_output_temp_{T}_ph_{ph}_recovery_{rec}_do_{do}.json")
        with open(flash_filename, 'w', encoding='utf-8-sig') as f:
            json.dump(result_flash_calc, f, ensure_ascii=False, indent=4)
        # Extract molecular concentration from flash result.
        try:
            molecular_concentration = result_flash_calc['data']['result']['total']['molecularConcentration']
        except Exception as e:
            print(f"Error extracting molecular concentration for T={T}, ph={ph}, rec={rec}, do={do}: {e}")
            return None

        # Prepare corrosion calculation input by copying the base specification.
        corrosion_spec = copy.deepcopy(base_corrosion_spec)
        corrosion_spec['params']['temperature']['value'] = T
        corrosion_spec['params']['pH'] = {"value": ph, "unit": ""}
        corrosion_spec['params']['inflows'] = molecular_concentration
        corrosion_spec['params']['corrosionParameters']['contactSurface'] = CONTACT_SURFACE

        # Run the corrosion rate calculation.
        result_corrosion_rate_calc = oliapi.call("corrosion-rates", chemistry_file_id, corrosion_spec)

        # Save the corrosion output.
        corrosion_filename = os.path.join(output_folder,
            f"corrosion_rate_output_temp_{T}_ph_{ph}_recovery_{rec}_do_{do}.json")
        with open(corrosion_filename, 'w', encoding='utf-8-sig') as f:
            json.dump(result_corrosion_rate_calc, f, ensure_ascii=False, indent=4)

        # Return summary information.
        return {
            "temperature": T,
            "pH": ph,
            "water_recovery": rec,
            "dissolved_oxygen": do,
            "contact_surface": CONTACT_SURFACE,
            "flash_file": flash_filename,
            "corrosion_file": corrosion_filename
        }
    except:
        print('Task failed')
        return {
            "temperature": T,
            "pH": ph,
            "water_recovery": rec,
            "dissolved_oxygen": do,
            "contact_surface": CONTACT_SURFACE,
            "flash_file": np.nan,
            "corrosion_file": np.nan
        }

def run_engine_survey(config, oliapi, CONTACT_SURFACE="Duplex stainless 2205", cases=None):
    # Create the output directory for this contact surface.
    output_folder = os.path.join("corrosion_example", "data", "result", CONTACT_SURFACE)
    os.makedirs(output_folder, exist_ok=True)

    # Get (or create) the chemistry file.
    chemistry_file_id = config['OLIAPI']['chemistry_ID']
    if not chemistry_file_id:
        inflows = ["H2O", "B2O3", "CACL2", "CO2", "FEIICL2", "KCL",
                   "MGCL2", "MGO", "NABR", "NACL", "SO3", "SRCL2",
                   "BASO4", "CACO3", "O2", "FEEL", "KF"]
        redox = {
            "enabled": True,
            "subSystems": [
                {
                    "enabled": True,
                    "name": "Iron",
                    "valenceStates": [
                        {"enabled": True, "name": "Fe(0)"},
                        {"enabled": True, "name": "Fe(+2)"},
                        {"enabled": True, "name": "Fe(+3)"}
                    ]
                }
            ]
        }
        chemistry_payload = get_chemistry_json_for_dbs(inflows, redox)
        result = oliapi.generate_chemistry_file("chemistry-builder", "", chemistry_payload)
        print(result)
        if result["status"] == "SUCCESS":
            chemistry_file_id = result["data"]["id"]
            print(f'Chemistry file ID: {chemistry_file_id}')
        else:
            print(f'Error generating chemistry file: {result["message"]}')
            return
    else:
        print(f'Chemistry file ID available in config.ini: {chemistry_file_id}')

    # Load the base corrosion specification.
    base_corrosion_spec = read_json_from_file(os.path.join("corrosion_example", "data", "specification"),
                                               'corrosion_survey_input.json')

    # Set original feed values - from OLI Studio
    original_H2O = 988.267 # g
    original_salts = 34.45362286 # g
    original_volume= 1 # L

    # Define survey ranges and build list of Ray tasks
    tasks = []
    if cases is None:
        temperatures = range(70, 96, 5)
        pH_values = [round(x, 2) for x in np.arange(4.0, 8.01, 0.5)]
        recovery_values = [round(x, 2) for x in np.arange(0.0, 0.8 + 0.001, 0.05)]
        do_values = [round(x, 1) for x in np.arange(1.0, 7.1, 2)]

        total_tasks = len(temperatures) * len(pH_values) * len(recovery_values) * len(do_values)

        for T in temperatures:
            for ph in pH_values:
                for rec in recovery_values:
                    for do in do_values:
                        tasks.append(
                            process_condition.remote(T, ph, rec, do, chemistry_file_id,
                                                       base_corrosion_spec, CONTACT_SURFACE,
                                                       output_folder, original_H2O, original_salts, original_volume)
                        )
    else:
        print('Running individual cases')
        temperatures = cases['temperature']
        pH_values = cases['ph']
        recovery_values = cases['recovery']
        do_values = cases['do']

        total_tasks = len(temperatures)
        for i in range(len(temperatures)):
            tasks.append(
                process_condition.remote(temperatures[i], pH_values[i], recovery_values[i], do_values[i], chemistry_file_id,
                                         base_corrosion_spec, CONTACT_SURFACE,
                                         output_folder, original_H2O, original_salts, original_volume)
            )

    print(f"Total tasks to run: {total_tasks}")

    # Run all tasks
    results = ray.get(tasks)

    # Summary
    survey_summary = [res for res in results if res is not None]
    summary_filename = os.path.join(output_folder, "survey_results_summary.json")
    with open(summary_filename, 'w', encoding='utf-8-sig') as f:
        json.dump(survey_summary, f, ensure_ascii=False, indent=4)
    print(f"Survey summary saved to {summary_filename}")
    print("All calculations complete.")

def run(contact_surface, cases=None):
    print(f'Running {contact_surface}')
    ray.init(num_cpus=8)
    print("Available resources:", ray.available_resources())
    config = configparser.ConfigParser()
    config.read('config.ini')
    oliapi = oli_api_connect(config)
    start_time = time.time()
    run_engine_survey(config, oliapi, CONTACT_SURFACE=contact_surface, cases=cases)
    end_time = time.time()
    time_elapsed = end_time - start_time
    print("Total elapsed time (parallel): ", time_elapsed)

if __name__ == "__main__":
    contact_surface = 'Duplex stainless 2507'

    # Run full survey for a material
    # run("Stainless steel 304")

    # Run missing cases
    # missing_cases = find_missing_cases(contact_surface)
    # run(contact_surface, cases=missing_cases)
