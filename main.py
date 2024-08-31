import sys

from irene.data_utils import Sampler
import warnings
sys.path.insert(0, "HPO-B/")
import importlib
import yaml
benchmark_plot = importlib.import_module('HPO-B.benchmark_plot')
from irene.hpo_diffusion import Network, NoiseAdder, Trainer
import numpy as np
from irene.methods import RandomSearch, MyAlgorithm
from irene.data_utils import generate_results
from irene.data_utils import extract_history
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'HPO-B'))
from hpob_handler import HPOBHandler

warnings.filterwarnings("ignore")

if __name__ == "__main__":
    # load the configurations
    config_file = "configs/config_basic.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    # step 0 - dataset generation
    hpob_handler = HPOBHandler(root_dir="HPO-B/hpob-data/", mode="v3-test")
    # history extraction
    H = []
    for dataset_id in ["10093", "3954", "43", "34536", "9970", "6566"]:
        h = extract_history(hpob_handler, "HPO-B/hpob-data/", '5971', dataset_id)
        H.extend(h)
    # Train the network
    sampler = Sampler(H)
    noise_adder = NoiseAdder()
    model = Network(h_len=len(H), context_dim=16)
    trainer = Trainer(model, noise_adder, sampler, lr=0.001)

    # Entrenamiento
    trainer.train(epochs=100)
    # step 2 - plot generation
    name = "benchmark_plot"
    experiments = ["RandomSearch", "MyAlgorithm", "GP", "DGP"]
    methods = [MyAlgorithm(H), RandomSearch()]
    new_method_name = 'MyAlgorithm.json'
    # generate the results of the algorithms inside methods in a way we can plot using HPO-B
    generate_results(config, methods, new_method_name)
    # plot the results from HPO-B
    benchmark_plotter = benchmark_plot.BenchmarkPlotter(experiments=experiments,
                                                        name=name,
                                                        n_trials=config['general']['n_trials'],
                                                        results_path=config['general']['results_path'],
                                                        output_path=config['general']['output_path'],
                                                        data_path=config['general']['data_path'])
    benchmark_plotter.search_spaces = config['general']['search_spaces']
    benchmark_plotter.plot()
    print("Finished")
