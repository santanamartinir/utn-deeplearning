import sys
sys.path.insert(0, "HPO-B/")
import importlib
import yaml
benchmark_plot = importlib.import_module('HPO-B.benchmark_plot')
from irene.methods import RandomSearch, MyAlgorithm
from irene.data_utils import generate_results

if __name__ == "__main__":
    # load the configurations
    config_file = "configs/config_basic.yaml"
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    name = "benchmark_plot"
    experiments = ["RandomSearch", "GP"]
    methods = [RandomSearch()]
    new_method_name = 'RandomSearch.json'
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
