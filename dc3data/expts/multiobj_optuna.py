import json
import subprocess
import os
import tempfile
import optuna
import numpy as np

# Load the base configuration from the default JSON file.
with open("configs/default.json", "r") as f:
    base_config = json.load(f)

def objective(trial):
    # Create a new configuration based on default, updating only the parameters to optimize.
    config = base_config.copy()
    
    # Optimize the selected parameters:
    # 1. initial_learning_rate: Sample in log-space; default 0.01 → 
    config["initial_learning_rate"] = trial.suggest_loguniform("initial_learning_rate", 1e-6, 1e-3)
    # 2. lr_supression: Sample in log-space; default 0.1 → range from 0.01 to 1.0
    config["lr_supression"] = trial.suggest_loguniform("lr_supression", 0.01, 1.0)
    # 3. decay_rate: Sample in log-space; default 0.0001 → range from 1e-6 to 1e-3
    config["decay_rate"] = trial.suggest_loguniform("decay_rate", 1e-7, 1e-3)
    # 4. sandwich_rounds: Integer parameter; default 10 → range from 1 to 20
    config["sandwich_rounds"] = trial.suggest_int("sandwich_rounds", 1, 20)
    # 5. width: Integer parameter; default 120
    config["width"] = trial.suggest_int("width",1,35)
    # 6. num_layers: Integer parameter; default 1 → range from 1 to 5
    config["num_layers"] = trial.suggest_int("num_layers", 1, 4)
    
    # Write the updated configuration to a temporary file.
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as temp:
        json.dump(config, temp)
        temp_file = temp.name

    try:
        # Execute your evaluation script.
        # This example assumes your script is called "sandwich_optuna.py" and takes a "--config" flag.
        result = subprocess.run(
            ["python", "sandwich_optuna.py", "--config", temp_file],
            capture_output=True,
            text=True,
            check=True
        )
        
        # The script might print several unneeded lines before the final JSON output.
        # Split the stdout into lines and iterate from the end, looking for valid JSON.
        output_lines = result.stdout.strip().splitlines()
        last_valid_line = None
        for line in reversed(output_lines):
            try:
                parsed = json.loads(line)
                last_valid_line = parsed
                break  # Use the first valid JSON (from the end).
            except json.JSONDecodeError:
                continue
        if last_valid_line is None:
            raise ValueError("No valid JSON output found in evaluation output.")
        
        # Extract the two losses.
        loss1 = last_valid_line.get("loss1", float("inf"))
        loss2 = last_valid_line.get("loss2", float("inf"))
    except subprocess.CalledProcessError as e:
        print("Error during evaluation:", e)
        loss1, loss2 = float("inf"), float("inf")
    finally:
        os.remove(temp_file)
    
    return loss1, loss2

def save_pareto_configs(study, trial):
    # Retrieve the Pareto front – the set of current best (non-dominated) trials.
    pareto_trials = study.best_trials  # List of Pareto-optimal trials
    pareto_configs = []
    for t in pareto_trials:
        params = t.params
        best_conf = base_config.copy()
        best_conf["initial_learning_rate"] = params["initial_learning_rate"]
        best_conf["lr_supression"] = params["lr_supression"]
        best_conf["decay_rate"] = params["decay_rate"]
        best_conf["sandwich_rounds"] = int(params["sandwich_rounds"])
        best_conf["width"] = int(params["width"])
        best_conf["num_layers"] = int(params["num_layers"])
        pareto_configs.append(best_conf)
    
    # Write the Pareto optimal configurations to best_config.json.
    with open("configs/best_config.json", "w") as f:
        json.dump(pareto_configs, f, indent=4)
    
    print(f"Trial {trial.number}: Updated best_config.json with Pareto-optimal configurations.")

if __name__ == "__main__":
    # Persist study results using SQLite storage for dashboard visualization.
    storage_url = "sqlite:///optuna_study_70var_light.db"
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        storage=storage_url,
        study_name="pareto_hyperopt_study_70var_light",
        load_if_exists=False
    )
    
    # Run optimization for, e.g., 50 trials with a callback saving best config after each trial.
    study.optimize(objective, n_trials=50, callbacks=[save_pareto_configs])
    
    print("Optimization complete!")
    # Display Pareto-optimal trials.
    for t in study.best_trials:
        print(f"Trial {t.number}: loss1 = {t.values[0]:.6f}, loss2 = {t.values[1]:.6f}, params = {t.params}")
    
    print("\nTo visualize the study, run:")
    print(f"optuna-dashboard {storage_url}")
