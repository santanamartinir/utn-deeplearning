import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from transformers import AutoModel, AutoTokenizer
from irene.data_utils import Sampler
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter()

class NoiseAdder:
    def __init__(self, beta_start=0.0001, beta_end=0.02, T=1000):
        # Initialise beta values linearly between beta_start and beta_end
        self.betas = torch.linspace(beta_start, beta_end, T)

    def add_noise(self, x, t):
        # Gets the beta corresponding to the time step t
        beta_t = self.betas[t]
        # Generates Gaussian noise
        noise = torch.randn_like(x)
        # Calculate the noisy configuration using NoiseAdder formula
        x_noisy = torch.sqrt(1 - beta_t) * x + torch.sqrt(beta_t) * noise
        return x_noisy, noise


class Network(nn.Module):
    def __init__(self, h_len, context_dim):
        super(Network, self).__init__()
        # h_len is the length of history
        self.h_len = h_len
        self.context_dim = context_dim
        # Loading a pre-trained transformer model
        encoder_layer = nn.TransformerEncoderLayer(d_model=17, nhead=1)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        # Fully connected layer to combine noisy configuration and context
        self.fc1 = nn.Linear(h_len * 17, context_dim)
        # Output layer to predict noise
        self.fc2 = nn.Linear(context_dim, 16)

    def forward(self, x_noisy, C, I, t):
        # Padding to get same length
        pad_tensor = torch.zeros(size=(C.shape[0], self.h_len - C.shape[1], C.shape[2])).to(device)
        mask = torch.ones(self.h_len, C.shape[0]).to(device)
        mask[C.shape[1]:] = 0
        C = torch.cat((C, pad_tensor), dim=1)
        context_embeddings1 = self.transformer(C, src_key_padding_mask=mask)
        context_embeddings2 = torch.relu(self.fc1(context_embeddings1.flatten()))
        t_vector = torch.tensor([(t/1000)] * self.context_dim, device=device)
        C_t_vector = context_embeddings2 + t_vector
        I_vector = torch.tensor([(I/100)] * self.context_dim, device=device)
        C_T_I_vector = C_t_vector + I_vector
        noise_pred = torch.nn.functional.tanh(self.fc2(C_T_I_vector + x_noisy))
        return noise_pred


class Trainer:
    def __init__(self, model, noise_adder, sampler, lr=0.001):
        # Move the model to the device
        self.model = model.to(device)
        self.noise_adder = noise_adder
        self.sampler = sampler
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        # Loss function: mean square error
        self.loss_fn = nn.MSELoss()

    def train_step(self):
        self.model.train()
        # Obtain a sample from the sampler
        x, I, C, t = self.sampler.sample()
        C = torch.reshape(C, (1, C.shape[0], C.shape[1])).to(device)
        x = torch.tensor(x, dtype=torch.float32).to(device)
        I = torch.tensor([I], dtype=torch.float32).to(device)
        # Add noise
        x_noisy, noise = self.noise_adder.add_noise(x, t)
        # Predict noise using model
        noise_pred = self.model(x_noisy, C, I, t)
        # Calculate loss
        loss = self.loss_fn(noise, noise_pred)
        # Update weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train(self, epochs):
        for epoch in range(epochs):
            loss = self.train_step()
            writer.add_scalar('Loss/train', loss, epoch)
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.4f}")



class Inference:
    def __init__(self, model, noise_adder):
        self.model = model.to(device)
        self.noise_adder = noise_adder

    def denoise(self, x_noisy, I, C):
        self.model.eval()
        with torch.no_grad():
            for t in reversed(range(1)):  # Assuming 1000 steps of denoising
                # Predict noise in each step
                noise_pred = self.model(x_noisy, C, I, t)
                # Update noisy configuration by removing predicted noise
                x_noisy = x_noisy - noise_pred
        return x_noisy

    def recommend(self, C):
        # Initialising a random configuration
        x_init = torch.randn((1, C.shape[2] - 1)).to(device)
        x_init = torch.reshape(x_init, (1, x_init.shape[0], x_init.shape[1])).to(device)
        I = torch.tensor([1], dtype=torch.float32).to(device)
        # Generate a new recommended configuration from the model
        x_recommend = self.denoise(x_init, I, C)
        return x_recommend.cpu().numpy()


if __name__ == "__main__":
    # Assuming a history of configurations
    H = [(np.random.rand(16).tolist(), [np.random.rand()]) for _ in range(1)]
    # Installation of components
    sampler = Sampler(H)
    print(sampler.sample())
    noise_adder = NoiseAdder()
    model = Network()
    trainer = Trainer(model, noise_adder, sampler, lr=0.001)

    # Training
    trainer.train(epochs=100)

    # Inference
    inference = Inference(model, noise_adder)
    C = torch.tensor([h[0] for h in H], dtype=torch.float32).to(device)
    x_recommend = inference.recommend(C)
    print("Recommended Configuration:", x_recommend)


    from methods import MyAlgorithm
    import sys
    import os
    import matplotlib.pyplot as plt

    sys.path.append(os.path.join(os.path.dirname(__file__), 'HPO-B'))
    from hpob_handler import HPOBHandler


    # Load the dataset and the handler
    hpob_hdlr = HPOBHandler(root_dir="hpob-data/", mode="v3-test", surrogates_dir="saved-surrogates/")

    # Get search space IDs and dataset IDs
    search_space_id = "5971"
    dataset_ids = ["10093", "3954", "43", "34536", "9970", "6566"]
    seeds = ["test0", "test1", "test2", "test3", "test4"]

    # Create instance
    H = [(np.random.rand(10), np.random.rand()) for _ in range(100)]  # Historial ficticio
    my_algo = MyAlgorithm(H)

    # Evaluate algorithm
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

    # Calculate media and deviation
    mean_acc = np.mean(results, axis=0)
    std_acc = np.std(results, axis=0)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.errorbar(range(len(mean_acc)), mean_acc, yerr=std_acc, fmt='-o', label='Mean Performance')
    plt.xlabel('Trials')
    plt.ylabel('Performance')
    plt.title('HPO-B Evaluation Results')
    plt.legend()
    plt.show()