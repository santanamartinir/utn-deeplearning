from irene.hpo_diffusion import Sampler, NoiseAdder, Network, Trainer, Inference
import random
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MyAlgorithm:
    """This is our proposed Algorithm"""
    def __init__(self, H):
        self.sampler = Sampler(H)
        self.noise_adder = NoiseAdder()
        self.model = Network(h_len=len(H), context_dim=16)
        self.trainer = Trainer(self.model, self.noise_adder, self.sampler, lr=0.001)
        self.inference = Inference(self.model, self.noise_adder)

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):
        if X_pen is not None:
            # Aquí podrías usar un proceso simple como random search para discrete space
            size_pending_eval = len(X_pen)
            idx = random.randint(0, size_pending_eval - 1)
            return idx
        else:
            # Para continuous space, usa el modelo para sugerir la próxima configuración
            # to make the dimension to 17
            X_obs_copy = X_obs.copy()
            X_obs_list = []
            for x in X_obs_copy:
                x_list = list(x)
                x_list.append(1)
                X_obs_list.append(x_list)
            X_obs = np.array(X_obs_list)
            C = torch.tensor(X_obs, dtype=torch.float32).to(device)
            C = torch.reshape(C, (1, C.shape[0], C.shape[1]))

            x_recommend = self.inference.recommend(C)
            return x_recommend


class RandomSearch:
    """Derived from HPO-B/methods/random_search.py for sake of testing"""
    def __init__(self):

        print("Using random search method...")

    def observe_and_suggest(self, X_obs, y_obs, X_pen=None):

        if X_pen is not None:
            size_pending_eval = len(X_pen)
            idx = random.randint(0, size_pending_eval-1)
            return idx

        else:
            dim = len(X_obs[0])
            bounds = tuple([(0,1) for i in range(dim)])
            x_new = np.array([random.uniform(lower, upper) for upper, lower in bounds]).reshape(-1, dim)

            return x_new