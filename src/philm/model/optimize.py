import optuna
import torch, os
from .trainer import train_model

def determine_nodelayer_no(input_num, output_num):
    pool = [4, 8, 16, 32, 64, 128, 256, 320, 512, 640, 1024, 2048, 2560, 4096]
    print(input_num, output_num)
    val = input_num + output_num
    hidden_dim = min(pool, key=lambda x: abs(x - val/2))
    
    # print(hidden_dim)
    hidden_size_list = [hidden_dim]
    indices = [pool.index(hidden_dim)]
    if (pool.index(hidden_dim)-1) > 0:
        indices.append(pool.index(hidden_dim)-1)
        hidden_size_list.append(pool[pool.index(hidden_dim)-1])
    if (pool.index(hidden_dim)+1) < len(pool):
        indices.append(pool.index(hidden_dim)+1)
        hidden_size_list.append(pool[pool.index(hidden_dim)+1])
    return hidden_size_list

def objective(trial, train_X_data, train_Y_data, val_X_data, val_Y_data, batch_size, patience, num_epochs, device):
    
    input_num = train_X_data.shape[1]
    output_num = train_Y_data.shape[1]
    hidden_size_list = determine_nodelayer_no(input_num, output_num)
    print("candidates:", hidden_size_list)
    
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-2, log=True)
    hidden_size = trial.suggest_categorical("hidden_size", hidden_size_list)
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-5, log=True)

    # Ensure the directory exists
    os.makedirs("checkpoints", exist_ok=True)

    model_path = os.path.join("checkpoints", f"model_trial_{trial.number}.pth")  # Save models in a folder named 'models'
    log_path = os.path.join("checkpoints", f"train_{trial.number}.log")

    # Call the train function and get the validation loss
    best_val_loss = train_model(
        train_X_data=train_X_data,
        train_Y_data=train_Y_data, 
        val_X_data=val_X_data,
        val_Y_data=val_Y_data,
        device=device,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        hidden_size=hidden_size,
        patience=patience,
        weight_decay=weight_decay,
        model_path=model_path,
        log_path=log_path
    )

    # Store the model name in the trial's user attributes for later reference
    trial.set_user_attr("model_path", model_path)

    return best_val_loss
