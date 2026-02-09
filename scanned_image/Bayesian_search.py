#!/usr/bin/env python
# coding=utf-8
# optimize_from_yaml.py
# Usage examples:
#   1) Create a starter YAML (once), then exit:
#        python optimize_from_yaml.py --config config.yaml --init_config
#   2) Run BO using your YAML:
#        python optimize_from_yaml.py --config config.yaml
#
# Notes:
# - You must have evaluate_parameters available: from evaluate_parameters import evaluate_parameters
# - YAML drives:
#     - bo_settings: init_points, n_iter, seed
#     - eval_args: target_samples, with_model, candidate_models, testing
#     - pbounds: lower/upper per parameter (the parameters from your original script)
#     - output.best_settings_yaml: where the best results YAML will be written

import sys
import time
import argparse
import copy
import yaml
from bayes_opt import BayesianOptimization
from bayes_opt.util import UtilityFunction

# Your original evaluation function
from evaluate_parameters import evaluate_parameters


# ---- Int casting control (so BO can search continuously but we evaluate with ints where needed) ----
INT_PARAMS = {
    # image quality / sizes
    "id_resized_shape1", "id_resized_shape2",
    "save_quality1", "save_quality2",
    # shadow + blur
    "shadow_offset1", "shadow_offset2", "shadow_color", "shadow_blur_radius",
    # perspective points
    "top_left1", "top_left2", "top_right1", "top_right2",
    "bottom_left1", "bottom_left2", "bottom_right1", "bottom_right2",
}
# You can add more here if your evaluate_parameters requires them as integers.


def cast_params_for_eval(params: dict) -> dict:
    """Cast BO's float params to ints for specific keys, leave others as-is."""
    casted = {}
    for k, v in params.items():
        if k in INT_PARAMS:
            casted[k] = int(round(v))
        else:
            casted[k] = float(v)
    return casted


def run_bo(pbounds: dict, fixed_args: dict, bo_settings: dict):
    """
    Run Bayesian Optimization with:
      - pbounds: {param: (lower, upper)}
      - fixed_args: passed directly into evaluate_parameters
      - bo_settings: {init_points, n_iter, seed}
    """
    # During BO we evaluate on validation (testing=False), and optionally run test after.
    testing_requested = bool(fixed_args.get("testing", False))
    fixed_args = copy.deepcopy(fixed_args)
    fixed_args["testing"] = False

    def objective_function(**params):
        # Cast for evaluation (e.g., coordinates, sizes, and qualities should be ints)
        coerced = cast_params_for_eval(params)
        return evaluate_parameters(**coerced, **fixed_args)

    optimizer = BayesianOptimization(
        f=objective_function,
        pbounds=pbounds,
        verbose=2,
        random_state=int(bo_settings.get("seed", 1)),
    )
    optimizer.set_gp_params(normalize_y=True)
    utility = UtilityFunction(kind="ei", kappa=2.5, xi=0.0)

    optimizer.maximize(
        init_points=int(bo_settings.get("init_points", 20)),
        n_iter=int(bo_settings.get("n_iter", 50)),
        acquisition_function=utility,
    )

    best_params_float = optimizer.max["params"]
    best_params_eval = cast_params_for_eval(best_params_float)

    # Evaluate the best parameters explicitly on validation once more (optional)
    best_val = evaluate_parameters(**best_params_eval, **fixed_args)

    # Evaluate on testing data if requested by YAML
    best_test = None
    if testing_requested:
        fixed_args_test = copy.deepcopy(fixed_args)
        fixed_args_test["testing"] = True
        best_test = evaluate_parameters(**best_params_eval, **fixed_args_test)

    return optimizer, best_params_float, best_params_eval, best_val, best_test


def build_pbounds_from_initial_margin(param_dict):
    pbounds = {}
    for name, spec in param_dict.items():
        if not isinstance(spec, dict) or "initial" not in spec or "margin" not in spec:
            raise ValueError(f"Parameter {name} must have 'initial' and 'margin'")
        initial = spec["initial"]
        margin = spec["margin"]
        pbounds[name] = (initial - margin, initial + margin)
    return pbounds

def main():
    parser = argparse.ArgumentParser(description="Bayesian Optimization driven by YAML config.")
    parser.add_argument("--config", type=str, default="scan_config.yaml",
                        help="Path to YAML with bo_settings, eval_args, pbounds, output.best_settings_yaml")
    args = parser.parse_args()


    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    bo_settings = cfg.get("bo_settings", {})
    eval_args   = cfg.get("eval_args", {})
    parameters  = cfg.get("parameters", {})
    pbounds = build_pbounds_from_initial_margin(parameters)
    out_cfg     = cfg.get("output", {})
    best_yaml   = out_cfg.get("best_settings_yaml", "best_settings.yaml")

    # Backward compatibility: allow candidate_models as str or list
    cand = eval_args.get("candidate_models", [])
    if isinstance(cand, str):
        eval_args["candidate_models"] = [cand]

    # Sanity checks
    required_eval = {"target_samples", "with_model"}
    missing = [k for k in required_eval if k not in eval_args]
    if missing:
        raise ValueError(f"Missing eval_args keys in YAML: {missing}")
    if eval_args.get("with_model", 0) and not eval_args.get("candidate_models"):
        raise ValueError("with_model=1 but candidate_models is empty in YAML.")

    print("candidate_models:", eval_args.get("candidate_models", []))

    since = time.time()
    optimizer, best_params_float, best_params_eval, best_val, best_test = run_bo(
        pbounds=pbounds,
        fixed_args=eval_args,
        bo_settings=bo_settings
    )

    # Mirror original prints
    print("Best Parameters (raw/float):", best_params_float)
    print("Best Parameters (eval-cast):", best_params_eval)
    print("Best Evaluation on validation data:", best_val)
    if best_test is not None:
        print("Best Evaluation on testing data:", best_test)

    best_params_float = {k: float(v) for k, v in optimizer.max["params"].items()}
    best_params_eval  = {k: (int(v) if k in INT_PARAMS else float(v)) for k, v in best_params_eval.items()}
    best_val  = float(best_val)  # if evaluate_parameters returns a numpy scalar
    best_test = None if best_test is None else float(best_test)

    # Persist results to YAML
    results_yaml = {
        "bo_settings": bo_settings,
        "eval_args": eval_args,
        "best": {
            "params_raw_float": best_params_float,
            "params_eval_casted": best_params_eval,
            "val_score": best_val,
            "test_score": best_test,
        }
    }
    with open(best_yaml, "w") as f:
        yaml.safe_dump(results_yaml, f, sort_keys=False)

    elapsed = time.time() - since
    print('Bayesian complete in {:.0f}m {:.0f}s'.format(elapsed // 60, elapsed % 60))
    print(f"Wrote best settings YAML to: {best_yaml}")


if __name__ == "__main__":
    main()

