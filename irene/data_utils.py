import sys
import importlib
import random
import torch

sys.path.insert(0, "HPO-B/")
benchmark_plot = importlib.import_module('HPO-B.benchmark_plot')


def generate_results(config, methods, new_method_name):
    benchmark_plotter = benchmark_plot.BenchmarkPlotter(experiments=[],
                                                        name='',
                                                        n_trials=config['n_trials'],
                                                        results_path=config['results_path'],
                                                        output_path=config['output_path'],
                                                        data_path=config['data_path'])
    benchmark_plotter.search_spaces = config['search_spaces']
    # generate results from the algorithm
    for method in methods:
        benchmark_plotter.generate_results(method, n_trials=config['n_trials'],
                                           new_method_name=new_method_name,
                                           search_spaces=config['search_spaces'],
                                           seeds=config['seeds'])

class Sampler:
    def __init__(self, H):
        self.H = H  # Historia de evaluaciones (lista de pares (configuración, rendimiento))

    def sample(self):
        # Selecciona una configuración aleatoria y su rendimiento
        (x, y) = random.choice(self.H)
        # Selecciona un subconjunto aleatorio de la historia de evaluaciones
        C = random.sample(self.H, random.randint(1, len(self.H)))
        # Changing the configurations to tensors
        config = []
        for c_tuple in C:
            temp = []
            temp.extend(c_tuple[0])
            temp.extend(c_tuple[1])
            config.append(temp)
        # Determina si x es mejor que todas las configuraciones en C
        I = 1 if all(y > y_c for (_, y_c) in C) else 0
        # Selecciona un paso de tiempo aleatorio
        t = random.randint(0, 999)  # Suponiendo T=1000 pasos de tiempo
        return x, I, torch.tensor(config), t