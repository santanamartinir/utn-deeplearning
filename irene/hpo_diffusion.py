import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import json
import yaml
from transformers import AutoModel, AutoTokenizer

# Configuración del dispositivo (GPU si está disponible, de lo contrario CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Sampler:
    def __init__(self, H, configs: dict):
        self.H = H  # Historia de evaluaciones (lista de pares (configuración, rendimiento))
        self.configs = configs
    def sample(self):
        # Selecciona una configuración aleatoria y su rendimiento
        (x, y) = random.choice(self.H)
        # Selecciona un subconjunto aleatorio de la historia de evaluaciones
        C = random.sample(self.H, random.randint(1, len(self.H)))
        # Determina si x es mejor que todas las configuraciones en C
        I = 1 if all(y > y_c for (_, y_c) in C) else 0
        # Selecciona un paso de tiempo aleatorio
        t = random.randint(0, self.configs['training']['diffusion']['timesteps'])  # Suponiendo T=1000 pasos de tiempo
        return x, I, C, t

class NoiseAdder:
    def __init__(self, configs):
        # Initialise beta values linearly between beta_start and beta_end
        self.betas = torch.linspace(configs['training']['diffusion']['beta_start'],
                                    configs['training']['diffusion']['beta_end'],
                                    configs['training']['diffusion']['timesteps'])

    def add_noise(self, x, t):
        # Gets the beta corresponding to the time step t
        beta_t = self.betas[t]
        # Generates Gaussian noise
        noise = torch.randn_like(x)
        # Calculate the noisy configuration using NoiseAdder formula
        x_noisy = torch.sqrt(1 - beta_t) * x + torch.sqrt(beta_t) * noise
        return x_noisy, noise


class Network(nn.Module):
    def __init__(self, input_dim, context_dim, hidden_dim):
        super(Network, self).__init__()
        # Cargar un modelo transformer preentrenado
        self.transformer = AutoModel.from_pretrained("bert-base-uncased")
        # Capa totalmente conectada para combinar la configuración ruidosa y el contexto
        self.fc1 = nn.Linear(input_dim + context_dim, hidden_dim)
        # Capa de salida para predecir el ruido
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x_noisy, I, C):
        # Obtener embeddings del transformer
        context_embeddings = self.transformer(input_ids=C, attention_mask=None)[0]
        context_embedding = context_embeddings.mean(dim=1)  # Media sobre la dimensión de la secuencia
        
        # Ajustar x_noisy a las dimensiones correctas
        if x_noisy.dim() == 1:
            x_noisy = x_noisy.unsqueeze(0).repeat(context_embedding.size(0), 1)
        
        # Comprobar las formas de los tensores
        print("x_noisy shape:", x_noisy.shape)
        print("context_embedding shape:", context_embedding.shape)
        
        # Asegúrate de que x_noisy y context_embedding tengan las dimensiones correctas
        if x_noisy.shape[0] != context_embedding.shape[0]:
            raise ValueError("Batch sizes do not match for x_noisy and context_embedding")

        # Concatenar la configuración ruidosa y la incrustación del contexto
        x_cat = torch.cat((x_noisy, context_embedding), dim=1)
        # Pasar por las capas totalmente conectadas
        h = torch.relu(self.fc1(x_cat))
        noise_pred = self.fc2(h)
        return noise_pred


class TrainLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        self.history = []

    def log(self, hyperparameters, performance):
        # Log hyperparameters and performance
        self.history.append((hyperparameters, performance))
        self.save()

    def save(self):
        with open(self.log_file, 'w') as f:
            json.dump(self.history, f)

    def load(self):
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                self.history = json.load(f)
        return [(tuple(h), p) for h, p in self.history]

class Trainer:
    def __init__(self, model, noise_adder, sampler, configs, logger):
        self.device = configs['training']['device']
        self.model = model.to(self.device)  # Mover el modelo al dispositivo
        self.noise_adder = noise_adder
        self.sampler = sampler
        self.optimizer = optim.Adam(self.model.parameters(), lr=configs['training']['lr'])
        self.loss_fn = nn.MSELoss() # TODO: eval. metric?
        self.logger = logger

    def train_step(self):
        self.model.train()
        # Obtener una muestra del sampler
        x, I, C, t = self.sampler.sample()
        x = torch.tensor(x, dtype=torch.float32).to(self.device)
        I = torch.tensor([I], dtype=torch.float32).to(self.device)
        # Example adjustment if C should be indices
        C = torch.tensor(np.array([c[0] for c in C]), dtype=torch.long).to(self.device)
        # Añadir ruido a la configuración
        x_noisy, noise = self.noise_adder.add_noise(x, t)
        # Predecir el ruido usando el modelo
        noise_pred = self.model(x_noisy, I, C)
        # Calcular la pérdida
        loss = self.loss_fn(noise, noise_pred)
        # Actualizar los pesos del modelo
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Log hyperparameters and loss
        hyperparameters = {"x": x.tolist(), "I": I.item(), "t": t}
        performance = loss.item()
        self.logger.log(hyperparameters, performance)

        return loss.item()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.train_step()
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")


class Inference:
    def __init__(self, model, noise_adder):
        self.model = model.to(device)  # Mover el modelo al dispositivo
        self.noise_adder = noise_adder

    def denoise(self, x_noisy, I, C):
        self.model.eval()
        with torch.no_grad():
            for t in reversed(range(1000)):  # Suponiendo 1000 pasos de denoising
                # Predecir el ruido en cada paso
                noise_pred = self.model(x_noisy, I, C)
                # Actualizar la configuración ruidosa eliminando el ruido predicho
                x_noisy = self.noise_adder.add_noise(x_noisy, t)[0] - noise_pred
        return x_noisy

    def recommend(self, C):
        # Inicializar una configuración aleatoria
        x_init = torch.randn((1, C.shape[1])).to(device)
        I = torch.tensor([1], dtype=torch.float32).to(device)
        # Generar una nueva configuración recomendada a partir del modelo
        x_recommend = self.denoise(x_init, I, C)
        return x_recommend.cpu().numpy()

if __name__ == "__main__":
    # load the configurations
    config_file = "../configs/config_basic.yaml"
    with open(config_file, 'r') as file:
        configs = yaml.safe_load(file)

    log_file = "training_history_data.json"
    logger = TrainLogger(log_file)

    H = logger.load()

    # TODO: init history ?!
    if not H:
        H = [[list(np.random.rand(16)), np.random.rand()] for _ in range(100)]

    # Instanciación de componentes
    sampler = Sampler(H, configs)
    noise_adder = NoiseAdder(configs)
    model = Network(input_dim=10, context_dim=768, hidden_dim=256)
    trainer = Trainer(model, noise_adder, sampler, configs)

    # Entrenamiento
    trainer.train(epochs=100)

    # Inferencia
    inference = Inference(model, noise_adder)
    C = torch.tensor([h[0] for h in H], dtype=torch.float32).to(device)
    x_recommend = inference.recommend(C)
    print("Recommended Configuration:", x_recommend)

    # TODO: save after training?
    logger.save()

    from methods import MyAlgorithm
    import sys
    import os
    import matplotlib.pyplot as plt

    # Agregar el directorio HPO-B al sys.path
    sys.path.append(os.path.join(os.path.dirname(__file__), 'HPO-B'))
    from hpob_handler import HPOBHandler


    # Load the dataset and the handler
    hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test", surrogates_dir="saved-surrogates/")

    # Get search space IDs and dataset IDs
    search_space_id = "5971"
    dataset_ids = ["10093", "3954", "43", "34536", "9970", "6566"]
    seeds = ["test0", "test1", "test2", "test3", "test4"]

    # Crear instancia de tu algoritmo
    H = [(np.random.rand(10), np.random.rand()) for _ in range(100)]  # Historial ficticio
    my_algo = MyAlgorithm(H)

    # Evaluar tu algoritmo
    results = []
    for dataset_id in dataset_ids:
        for seed in seeds:
            acc = hpob_hdlr.evaluate_continuous(
                my_algo,
                search_space_id=search_space_id,
                dataset_id=dataset_id,
                seed=seed,
                n_trials=50
            )
            results.append(acc)

    # Calcular media y desviación estándar de la métrica de rendimiento
    mean_acc = np.mean(results, axis=0)
    std_acc = np.std(results, axis=0)

    # Graficar los resultados
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(mean_acc)), mean_acc, yerr=std_acc, fmt='-o', label='Mean Performance')
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('HPO-B Evaluation Results')
    plt.legend()
    plt.show()