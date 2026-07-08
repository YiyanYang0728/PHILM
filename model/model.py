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

    def _solve_ode(self, x):
        t = torch.tensor([0, 1], dtype=x.dtype, device=x.device)
        if x.device.type == "mps":
            return self.ode_solver(
                self.ode_func, x, t, method="rk4", options={"step_size": 1.0}
            )[-1]
        return self.ode_solver(self.ode_func, x, t, rtol=1e-4, atol=1e-4)[-1]

    def forward(self, x):
        x = x.to(dtype=self.fc1.weight.dtype)
        x = torch.flatten(x, 1)  # Flatten the input tensor except for the batch dimension
        x = self.fc1(x)
        x = self._solve_ode(x)
        x = self.fc2(x)
        return x

    def get_repr1(self, x):
        x = x.to(dtype=self.fc1.weight.dtype)
        repr1 = self.fc1(x)
        return repr1
    
    def get_repr2(self, x):
        x = x.to(dtype=self.fc1.weight.dtype)
        x = torch.flatten(x, 1)  # Flatten the input tensor except for the batch dimension
        x = self.fc1(x)
        repr2 = self._solve_ode(x)
        return repr2
