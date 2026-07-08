import torch
import torch.nn as nn
import torchdiffeq

class ODEFunc(nn.Module):
    """
    Defines the neural ODE function.
    """
    def __init__(self, hidden_dim):
        super(ODEFunc, self).__init__()
        layers = []
        prev_dim = hidden_dim
        for _ in range(2):
            # print(prev_dim, h_dim)
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.Tanh())
            prev_dim = hidden_dim
        self.model = nn.Sequential(*layers)

    def forward(self, t, x):
        return self.model(x)

class MLP_NODE(nn.Module):
    def __init__(self, input_dim, hidden_size, output_dim):
        super(MLP_NODE, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ode_func = ODEFunc(hidden_size)
        self.ode_solver = torchdiffeq.odeint  # Using adjoint method for memory efficiency
        self.fc2 = nn.Linear(hidden_size, output_dim)

    def forward(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor except for the batch dimension
        x = self.fc1(x)
        t = torch.tensor([0, 1], dtype=torch.float32, device=x.device)  # Time interval for ODE solver, moved to the same device as x
        x = self.ode_solver(self.ode_func, x, t, rtol=1e-4, atol=1e-4)[-1]  # Get the result at the final time point
        x = self.fc2(x)
        return x

    def get_repr1(self, x):
        repr1 = self.fc1(x)
        return repr1
    
    def get_repr2(self, x):
        x = torch.flatten(x, 1)  # Flatten the input tensor except for the batch dimension
        x = self.fc1(x)
        t = torch.tensor([0, 1], dtype=torch.float32, device=x.device)  # Time interval for ODE solver, moved to the same device as x
        repr2 = self.ode_solver(self.ode_func, x, t, rtol=1e-4, atol=1e-4)[-1]  # Get the result at the final time point
        return repr2
