import sys
import importlib

sys.path.insert(0, "HPO-B/")
benchmark_plot = importlib.import_module('HPO-B.benchmark_plot')


def generate_results(config, methods, new_method_name):
    benchmark_plotter = benchmark_plot.BenchmarkPlotter(experiments=[],
                                                        name='',
                                                        n_trials=config['general']['n_trials'],
                                                        results_path=config['general']['results_path'],
                                                        output_path=config['general']['output_path'],
                                                        data_path=config['general']['data_path'])
    benchmark_plotter.search_spaces = config['general']['search_spaces']
    # generate results from the algorithm
    for method in methods:
        benchmark_plotter.generate_results(method, n_trials=config['general']['n_trials'],
                                           new_method_name=new_method_name,
                                           search_spaces=config['general']['search_spaces'],
                                           seeds=config['general']['seeds'])