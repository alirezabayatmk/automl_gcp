from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Mapping, Optional
from functools import partial
import time
import json 

import numpy as np
import matplotlib.pyplot as plt
import torch
import optuna 

from ConfigSpace import (
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
    Constant,
    InCondition,
    Categorical
)
from ConfigSpace.read_and_write import json as cs_json
from ConfigSpace.read_and_write import pcs_new, pcs

from sklearn.model_selection import StratifiedKFold
from smac.facade.multi_fidelity_facade import MultiFidelityFacade as SMAC4MF
from smac.intensifier.hyperband import Hyperband
from smac.scenario import Scenario
from smac.facade import AbstractFacade
from torch.utils.data import DataLoader, Subset
from dask.distributed import get_worker

from cnn import Model

from datasets import load_deep_woods, load_fashion_mnist

logger = logging.getLogger(__name__)

CV_SPLIT_SEED = 42

def configuration_space(
        device: str,
        dataset: str,
        cv_count: int = 3,
        budget_type: str = "img_size",
        datasetpath: str | Path = Path("."),
        cs_file: Optional[str | Path] = None
) -> ConfigurationSpace:
    """Build Configuration Space which defines all parameters and their ranges."""
    if cs_file is None:
        # This serves only as an example of how you can manually define a Configuration Space
        # To illustrate different parameter types;
        # we use continuous, integer and categorical parameters.
        cs = ConfigurationSpace(
            {
                "n_conv_layers": Integer("n_conv_layers", (1, 3), default=3),
                "use_BN": Categorical("use_BN", [True, False], default=True),
                "global_avg_pooling": Categorical("global_avg_pooling", [True, False], default=True),
                "n_channels_conv_0": Integer("n_channels_conv_0", (32, 512), default=512, log=True),
                "n_channels_conv_1": Integer("n_channels_conv_1", (16, 512), default=512, log=True),
                "n_channels_conv_2": Integer("n_channels_conv_2", (16, 512), default=512, log=True),
                "n_fc_layers": Integer("n_fc_layers", (1, 3), default=3),
                "n_channels_fc_0": Integer("n_channels_fc_0", (32, 512), default=512, log=True),
                "n_channels_fc_1": Integer("n_channels_fc_1", (16, 512), default=512, log=True),
                "n_channels_fc_2": Integer("n_channels_fc_2", (16, 512), default=512, log=True),
                "batch_size": Integer("batch_size", (1, 1000), default=200, log=True),
                "learning_rate_init": Float(
                    "learning_rate_init",
                    (1e-5, 1.0),
                    default=1e-3,
                    log=True,
                ),
                "kernel_size": Constant("kernel_size", 3),
                "dropout_rate": Constant("dropout_rate", 0.2),
                "device": Constant("device", device),
                "dataset": Constant("dataset", dataset),
                "datasetpath": Constant("datasetpath", str(datasetpath.absolute())),
            }
        )

        # Add conditions to restrict the hyperparameter space
        use_conv_layer_2 = InCondition(cs["n_channels_conv_2"], cs["n_conv_layers"], [3])
        use_conv_layer_1 = InCondition(cs["n_channels_conv_1"], cs["n_conv_layers"], [2, 3])

        use_fc_layer_2 = InCondition(cs["n_channels_fc_2"], cs["n_fc_layers"], [3])
        use_fc_layer_1 = InCondition(cs["n_channels_fc_1"], cs["n_fc_layers"], [2, 3])

        # Add multiple conditions on hyperparameters at once:
        cs.add_conditions([use_conv_layer_2, use_conv_layer_1, use_fc_layer_2, use_fc_layer_1])
    
    else:
        with open(cs_file, "r") as fh:
            cs_string = fh.read()
            if cs_file.suffix == ".json":
                cs = cs_json.read(cs_string)
            elif cs_file.suffix in [".pcs", ".pcs_new"]:
                cs = pcs_new.read(pcs_string=cs_string)
        logging.info(f"Loaded configuration space from {cs_file}")

        if "device" not in cs:
            cs.add_hyperparameter(Constant("device", device))
        if "dataset" not in cs:
            cs.add_hyperparameter(Constant("dataset", dataset))
        if "cv_count" not in cs:
            cs.add_hyperparameter(Constant("cv_count", cv_count))
        if "budget_type" not in cs:
            cs.add_hyperparameter(Constant("budget_type", budget_type))
        if "datasetpath" not in cs:
            cs.add_hyperparameter(Constant("datasetpath", str(datasetpath.absolute())))
        logging.debug(f"Configuration space:\n{cs}")

    return cs

def get_optimizer_and_criterion(
        cfg: Mapping[str, Any]
) -> tuple[
    type[torch.optim.AdamW | torch.optim.Adam],
    type[torch.nn.MSELoss | torch.nn.CrossEntropyLoss],
]:
    if cfg["optimizer"] == "AdamW":
        model_optimizer = torch.optim.AdamW
    else:
        model_optimizer = torch.optim.Adam

    if cfg["train_criterion"] == "mse":
        train_criterion = torch.nn.MSELoss
    else:
        train_criterion = torch.nn.CrossEntropyLoss

    return model_optimizer, train_criterion

def cnn_from_cfg(
        cfg: Configuration,
        seed: int = 0,
        budget: int = 20,
) -> float:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This is the function-call we try to optimize. Chosen values are stored in
    the configuration (cfg).

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    val_accuracy cross validation accuracy
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = False

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # determine fidelity and used budget
    epochs = max(3, int(np.floor(budget)))  

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "fashion_mnist" in dataset:
        input_shape, train_val, _ = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, _ = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(16, 16),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv = StratifiedKFold(n_splits=3, random_state=CV_SPLIT_SEED, shuffle=True)

    score = []
    cv_splits = cv.split(train_val, train_val.targets)
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        train_data = Subset(train_val, list(train_idx))
        val_data = Subset(train_val, list(valid_idx))

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
        )

        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_val.classes),
        )
        model = model.to(model_device)

        # summary(model, input_shape, device=device)

        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(epochs):  # 20 epochs
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{epochs}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            # logging.info(f"Worker:{worker_id} => Train accuracy {train_score:.3f} | loss {train_loss}")

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy {val_score:.3f}")
        score.append(val_score)

    val_error = 1 - np.mean(score)  # because minimize

    results = val_error
    return results

def optimize_smac_hyperparameters(trial):
    # Optimize parameters of the random forest model
    n_trees = trial.suggest_int("n_trees", 10, 25)
    ratio_features = trial.suggest_float("ratio_features", 0.7, 1.0)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

    return {
        "n_trees": n_trees,
        "ratio_features": ratio_features,
        "min_samples_split": min_samples_split,
        "min_samples_leaf": min_samples_leaf,
    }

def plot_main_trajectory(facades: list[AbstractFacade], plot_name: str = 'epoch') -> None:
    """Plots the trajectory (incumbents) of the optimization process."""
    plt.figure()
    plt.title("Trajectory")
    plt.xlabel("Wallclock time [s]")

    for facade in facades:
        X, Y = [], []
        for item in facade.intensifier.trajectory:
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        plt.plot(X, Y, label=plot_name)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()
    plt.savefig(f"visualizations/trajectory_{plot_name}.png")

def plot_optuna_trajectories(dictionary):
    plt.figure()
    plt.title("Trajectories")
    plt.xlabel("Wallclock time [s]")


    for trial_key, data in dictionary.items():
        trial_no = data['trial_no']
        trial_params = data['trial_params']
        trajectory = data['trajectory']

        X, Y = [], []
        for item in trajectory.intensifier.trajectory:
            assert len(item.config_ids) == 1
            assert len(item.costs) == 1

            y = item.costs[0]
            x = item.walltime

            X.append(x)
            Y.append(y)

        legend_label = f"{trial_no}. {trial_params}"
        plt.plot(X, Y, label=legend_label)
        plt.scatter(X, Y, marker="x")

    plt.legend()
    plt.show()
    plt.savefig(f"visualizations/trajectory_optuna.png")

def final_training(
        cfg: Configuration,
        seed: int
) -> float:
    """
    Creates an instance of the torch_model and fits the given data on it.
    This trains the given configuration on the train set and also gives the validation error
    that might be used for early stopping. Finally the test accuracy of the final model is put out.

    :param cfg: Configuration (basically a dictionary)
        configuration chosen by smac
    :param seed: int or RandomState
        used to initialize the rf's random generator
    :param budget: float
        used to set max iterations for the MLP
    Returns
    -------
    test accuracy on test set
    """
    try:
        worker_id = get_worker().name
    except ValueError:
        worker_id = 0

    # If data already existing on disk, set to False
    download = False

    lr = cfg["learning_rate_init"]
    dataset = cfg["dataset"]
    device = cfg["device"]
    batch_size = cfg["batch_size"]
    ds_path = cfg["datasetpath"]

    # unchangeable constants that need to be adhered to, the maximum fidelities
    img_size = 32  # example fidelity to use

    # Device configuration
    torch.manual_seed(seed)
    model_device = torch.device(device)

    if "fashion_mnist" in dataset:
        input_shape, train_val, test = load_fashion_mnist(datadir=Path(ds_path, "FashionMNIST"))
    elif "deepweedsx" in dataset:
        input_shape, train_val, test = load_deep_woods(
            datadir=Path(ds_path, "deepweedsx"),
            resize=(img_size, img_size),
            balanced="balanced" in dataset,
            download=download,
        )
    else:
        raise NotImplementedError

    # returns the cross-validation accuracy
    # to make CV splits consistent
    cv = StratifiedKFold(n_splits=3, random_state=CV_SPLIT_SEED, shuffle=True)

    score = []
    cv_splits = cv.split(train_val, train_val.targets)
    for cv_index, (train_idx, valid_idx) in enumerate(cv_splits, start=1):
        logging.info(f"Worker:{worker_id} ------------ CV {cv_index} -----------")
        train_data = Subset(train_val, list(train_idx))
        val_data = Subset(train_val, list(valid_idx))

        train_loader = DataLoader(
            dataset=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            dataset=val_data,
            batch_size=batch_size,
            shuffle=False,
        )

        test_loader = DataLoader(
            dataset=test,
            batch_size=batch_size,
            shuffle=False,
        )

        model = Model(
            config=cfg,
            input_shape=input_shape,
            num_classes=len(train_val.classes),
        )
        model = model.to(model_device)

        # summary(model, input_shape, device=device)

        model_optimizer, train_criterion = get_optimizer_and_criterion(cfg)
        optimizer = model_optimizer(model.parameters(), lr=lr)
        train_criterion = train_criterion().to(device)

        for epoch in range(20):  # 20 epochs
            logging.info(f"Worker:{worker_id} " + "#" * 50)
            logging.info(f"Worker:{worker_id} Epoch [{epoch + 1}/{20}]")
            train_score, train_loss = model.train_fn(
                optimizer=optimizer,
                criterion=train_criterion,
                loader=train_loader,
                device=model_device
            )
            logging.info(f"Worker:{worker_id} => Train accuracy epoch {train_score:.3f} | loss {train_loss}")
            val_score_epoch = model.eval_fn(val_loader, device)
            logging.info(f"Worker:{worker_id} => Val accuracy epoch {val_score_epoch:.3f}")

        val_score = model.eval_fn(val_loader, device)
        logging.info(f"Worker:{worker_id} => Val accuracy cv_splits {val_score:.3f}")
        score.append(val_score)

    val_error = 1 - np.mean(score)  # because minimize
    results = val_error
    print("results (avg val_error):", results)
    test_score = model.eval_fn(test_loader, device)
    logging.info(f"Worker:{worker_id} => Test accuracy {test_score:.3f}")

    return test_score



if __name__ == "__main__":


    #################################################################################################################
    #                                           Setting up the arguments                                            #
    #################################################################################################################


    parser = argparse.ArgumentParser(description="MF example using BOHB.")
    parser.add_argument(
        "--dataset",
        choices=["deepweedsx", "deepweedsx_balanced", "fashion_mnist"],
        default="deepweedsx_balanced",
        help="dataset to use (task for the project: deepweedsx_balanced)",
    )
    parser.add_argument(
        "--working_dir",
        default="./tmp",
        type=str,
        help="directory where intermediate results are stored",
    )
    parser.add_argument(
        "--runtime",
        default=15000,
        type=int,
        help="Running time (seconds) allocated to run the algorithm",
    )
    parser.add_argument(
        "--max_budget",
        type=float,
        default=20,
        help="maximal budget (epochs) to use with BOHB",
    )
    parser.add_argument(
        "--min_budget", type=float, default=3, help="Minimum budget (epochs) for BOHB"
    )
    parser.add_argument("--eta", type=int, default=2, help="eta for BOHB")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--device", type=str, default="cpu", help="device to run the models"
    )
    parser.add_argument(
        "--workers", type=int, default=16, help="num of workers to use with BOHB"
    )
    parser.add_argument(
        "--n_trials", type=int, default=500, help="Number of iterations to run SMAC for"
    )
    parser.add_argument(
        "--cv_count",
        type=int,
        default=2,
        help="Number of cross validations splits to create. "
             "Will not have an effect if the budget type is cv_splits",
    )
    parser.add_argument(
        "--log_level",
        choices=[
            "NOTSET"
            "CRITICAL",
            "FATAL",
            "ERROR",
            "WARN",
            "WARNING",
            "INFO",
            "DEBUG",
        ],
        default="NOTSET",
        help="Logging level",
    )
    parser.add_argument('--configspace', type=Path, default="minimal_configspace.json",
                        help='Path to file containing the configuration space')
    parser.add_argument('--datasetpath', type=Path, default=Path('./data/'),
                        help='Path to directory containing the dataset')
    args = parser.parse_args()
    
    logging.basicConfig(level=args.log_level)

    configspace = configuration_space(
        device=args.device,
        dataset=args.dataset,
        cv_count=args.cv_count,
        datasetpath=args.datasetpath,
        cs_file=args.configspace
    )

    optuna_facades: list[AbstractFacade] = []
    facades: list[AbstractFacade] = []

    #################################################################################################################
    #                                 Plot Multi-fidelity plot                                                      #
    #################################################################################################################




    #################################################################################################################
    #                                 OPTUNA optimization process and visualizations                                #
    #################################################################################################################    

    try:
        start_time = time.time()

        # Optuna applied for optimizing the hyperparameters of the random forest model of SMAC
        def objective(trial):
            rf_params = optimize_smac_hyperparameters(trial)

            scenario = Scenario(
                name="OptunaSMAC",
                configspace=configspace,
                deterministic=True,
                output_directory=args.working_dir,
                seed=args.seed,
                n_trials=100,
                max_budget=args.max_budget,
                min_budget=args.min_budget,
                n_workers=16,
                walltime_limit=180,
            )

            smac = SMAC4MF(
                target_function=cnn_from_cfg,
                scenario=scenario,
                initial_design=SMAC4MF.get_initial_design(scenario=scenario, n_configs=3),
                intensifier=Hyperband(
                    scenario=scenario,
                    incumbent_selection="highest_budget",
                    eta=2,
                ),
                model=SMAC4MF.get_model(scenario, **rf_params),
                overwrite=True,
                logging_level=args.log_level,  
            )

            incumbent = smac.optimize()
            optuna_facades.append(smac)

            return smac.validate(incumbent)
        
        
        # create Optuna study object and optimize the objective function
        logging.info("Optuna Optimization started")
        study = optuna.create_study(direction="minimize", study_name="SMAC_HPO")
        study.optimize(objective, n_trials=20, n_jobs=-1)

        best_params = study.best_params
        logging.info(f"Best optuna parameters: {best_params}")

        with open(Path(args.working_dir, "optuna_best_params.json"), "w") as fh:
            json.dump(best_params, fh)

        optuna_viz_dic = {}
        for smc_obj, trial in zip(optuna_facades, study.trials):
            optuna_viz_dic['trial_no'] = trial.number
            optuna_viz_dic['trial_params'] = trial.params
            optuna_viz_dic['trajectory'] = smc_obj


    except Exception as e:
        logging.error(f"Error in Optuna optimization: {e}")
        raise e
    

    try:       
        plot_optuna_trajectories(optuna_viz_dic)

        # creating optuna hyperparameter optimization progress plots
        fig = optuna.visualization.plot_contour(study)
        fig.show()
        fig.write_image("optuna_contour.png")

        fig = optuna.visualization.plot_optimization_history(study)
        fig.show()
        fig.write_image("optuna_optimization_history.png")

        fig = optuna.visualization.plot_timeline(study)
        fig.show()
        fig.write_image("optuna_timeline.png")

        fig = optuna.visualization.plot_param_importances(study)
        fig.show()
        fig.write_image("optuna_param_importances.png")

        fig = optuna.visualization.plot_parallel_coordinate(study)
        fig.show()
        fig.write_image("optuna_parallel_coordinate.png")

        fig = optuna.visualization.plot_edf([study])
        fig.show()
        fig.write_image("optuna_edf.png")

        logging.info(f"--- Optuna optimization took: {time.time() - start_time} seconds ---")

    except Exception as e:
        logging.error(f"Error in Optuna visualizations: {e}")
        raise e

    #################################################################################################################
    #                          SMAC optimization process with the best parameters from Optuna                       #
    #################################################################################################################
    

    try:
        logging.info("Final SMAC Optimization started")

        start_time = time.time()

        best_scenario = Scenario(
            name="SMACPlotting",
            configspace=configspace,
            deterministic=True,
            output_directory=args.working_dir,
            seed=args.seed,
            n_trials=args.n_trials,
            max_budget=args.max_budget,
            min_budget=args.min_budget,
            n_workers=args.workers,
            walltime_limit=args.runtime
        )

        best_smac = SMAC4MF(
            target_function=cnn_from_cfg,
            scenario=best_scenario,
            initial_design=SMAC4MF.get_initial_design(scenario=best_scenario, n_configs=3),
            intensifier=Hyperband(
                scenario=best_scenario,
                incumbent_selection="highest_budget",
                eta=args.eta,
            ),
            model=SMAC4MF.get_model(best_scenario, **best_params),
            overwrite=True,
            logging_level=args.log_level,  
        )

        best_incumbent = best_smac.optimize()

        best_default_cost = best_smac.validate(configspace.get_default_configuration())
        logging.info(f"Best Default cost ({best_smac.intensifier.__class__.__name__}): {best_default_cost}")

        best_incumbent_cost = best_smac.validate(best_incumbent)
        logging.info(f"Best Incumbent cost ({best_smac.intensifier.__class__.__name__}): {best_incumbent_cost}")

        facades.append(best_smac)

        plot_main_trajectory(facades)

        logging.info(f"--- Final SMAC optimization took: {time.time() - start_time} seconds ---")

    except Exception as e:
        logging.error(f"Error in final SMAC optimization: {e}")
        raise e

    
    try:
        # train the final CNN model with the best incumbent
        start_time = time.time()
        final_training(best_incumbent, args.seed)
        logging.info(f"--- Final CNN training with best incumbent took: {time.time() - start_time} seconds ---")

    except Exception as e:  
        logging.error(f"Error in final training: {e}")
        raise e