import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import optuna
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
import geopandas as gpd
from osgeo import gdal
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
import random
import sys
import matplotlib.pyplot as plt
import seaborn as sns

torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"  

class Logger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()
    def __del__(self):
        if self.log:
            self.log.close()

class EarlyStopping:
    def __init__(self, patience=25, delta=0):
        self.patience = patience  
        self.delta = delta  
        self.counter = 0  
        self.best_score = None  
        self.early_stop = False  

    def __call__(self, val_loss):
        score = -val_loss  

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
class DNNDepthNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128, 64], activation='relu'):
        super(DNNDepthNet, self).__init__()

        layers = []
        in_dim = input_dim
        for hid_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(self.get_activation(activation))
            in_dim = hid_dim
        layers.append(nn.Linear(in_dim, 1))  
        self.model = nn.Sequential(*layers)

    def get_activation(self, activation):
        if activation == 'relu':
            return nn.ReLU()
        elif activation == 'tanh':
            return nn.Tanh()
        elif activation == 'LeakyReLU':
            return nn.LeakyReLU()
        else:
            raise NotImplementedError

    def forward(self, x):
        return self.model(x)

def optimize_mlp_hyperparams(X_train, Y_train, X_val, Y_val, input_dim):
    def objective(trial):
        n_layers = trial.suggest_int('n_layers', 1, 4)

        hidden_dims = []
        for i in range(n_layers):
            hidden_dim = trial.suggest_int(f'hidden_dim_{i}', 8, 256)  
            hidden_dims.append(hidden_dim)

        activation = trial.suggest_categorical('activation', ['relu', 'tanh', 'LeakyReLU'])
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = DNNDepthNet(input_dim=input_dim, hidden_dims=hidden_dims,
                            activation=activation)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_train_tensor = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_val_tensor = torch.tensor(Y_val.values.reshape(-1, 1), dtype=torch.float32).to(device)

        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, Y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, Y_val_tensor).item()

        return val_loss

    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=50)

    best_params = study.best_params
    return best_params

class CNN1DDepthNet(nn.Module):
    def __init__(self, input_dim, conv1_out=16, conv2_out=32, conv3_out=64, conv4_out=128):
        super(CNN1DDepthNet, self).__init__()
        self._feature_dim = None
        self.input_dim = input_dim
        self.conv1_out = conv1_out
        self.conv2_out = conv2_out
        self.conv3_out = conv3_out
        self.conv4_out = conv4_out

        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, conv1_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv1_out, conv2_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv2_out, conv3_out, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(conv3_out, conv4_out, kernel_size=3, padding=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy_input = torch.randn(1, self.input_dim) 
            dummy_input = dummy_input.unsqueeze(1)      
            conv_output = self.conv_layers(dummy_input)
            flattened_size = conv_output.flatten(start_dim=1).size(1)

        self.fc = nn.Linear(flattened_size, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv_layers(x)
        x = x.flatten(start_dim=1)

        return self.fc(x)

def optimize_cnn1d_hyperparams(X_train, Y_train, X_val, Y_val, input_dim):
    def objective(trial):
        conv1_channels = trial.suggest_int('conv1_channels', 8, 64)
        conv2_channels = trial.suggest_int('conv2_channels', 16, 128)
        conv3_channels = trial.suggest_int('conv3_channels', 32, 256)
        conv4_channels = trial.suggest_int('conv4_channels', 64, 512)

        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)

        model = CNN1DDepthNet(input_dim=input_dim, conv1_out=conv1_channels, conv2_out=conv2_channels, conv3_out=conv3_channels, conv4_out=conv4_channels)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.MSELoss()

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        Y_train_tensor = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)

        X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        Y_val_tensor = torch.tensor(Y_val.values.reshape(-1, 1), dtype=torch.float32).to(device)

        for epoch in range(20):
            model.train()
            optimizer.zero_grad()
            outputs = model(X_train_tensor)
            loss = criterion(outputs, Y_train_tensor)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_tensor)
            val_loss = criterion(val_pred, Y_val_tensor).item()
            
        return val_loss

    sampler = TPESampler(seed=42) 
    study = optuna.create_study(direction='minimize', sampler=sampler)
    study.optimize(objective, n_trials=30, timeout=1800)

    best_params = study.best_params
    return best_params

class LossFunction(nn.Module):
    def __init__(self, loss_type='mse', lambda_phy=1, lambda_phy_base=1.0, lambda_reg=1e-4, epsilon=1e-6):
        super(LossFunction, self).__init__()
        self.loss_type = loss_type
        self.lambda_phy = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)

    def forward(self, model, pred_depth, Y_train, X_train, scaler):
        loss_data = F.mse_loss(pred_depth, Y_train)
        loss_phy = physical_loss(pred_depth, X_train, Y_train, scaler)  

        if self.loss_type == 'mse':
            return loss_data
        elif self.loss_type == 'mse+phy':  
            return loss_data + self.lambda_phy * loss_phy.clamp(min=1e-8, max=1e8)

def physical_loss(pred_depth, X_train, Y_train, scaler): 
    blue_band = 13
    green_band = 27

    X_train_cpu = X_train.cpu().numpy() if torch.is_tensor(X_train) else X_train
    X_np = scaler.inverse_transform(X_train_cpu)
    X_np = X_np / np.pi
    Y_np = Y_train.detach().cpu().numpy()

    num_samples = min(1024, len(X_np))
    indices = np.random.choice(len(X_np), size=num_samples, replace=False)

    log_blue = np.log(X_np[indices, blue_band] * 1000)
    log_green = np.log(X_np[indices, green_band] * 1000)
    band_ratio = (log_blue / log_green).reshape(-1)
    y_train_sample = Y_np[indices].reshape(-1)

    model = LinearRegression()
    model.fit(band_ratio.reshape(-1, 1), y_train_sample)
    phy_depth_band_ratio = model.predict(band_ratio.reshape(-1, 1))  
    phy_depth_band_ratio = torch.tensor(phy_depth_band_ratio, dtype=torch.float32).to(pred_depth.device).view(-1)

    pred_sample = pred_depth[indices].view(-1)
    relative_error_br1 = torch.abs((phy_depth_band_ratio.view(-1) - pred_sample) / (pred_sample + 1e-6))
    relative_error_br = torch.mean(F.relu(phy_depth_band_ratio - pred_sample))
    fval_band_ratio = torch.norm(phy_depth_band_ratio - pred_sample) / torch.sum(pred_sample + 1e-6)

    return fval_band_ratio

def train_model(X_train, Y_train, X_test, Y_test, 
                model_type='DNN', best_params=None,
                epochs=400, batch_size=256, shuffle=True, patience=25,
                loss_type='mse+phy',
                save_path='model.pth'):

    global avg_train_loss
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train.values.reshape(-1, 1), dtype=torch.float32)

    X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test.values.reshape(-1, 1), dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    if model_type == 'DNN':
        if best_params is not None:
            n_layers = best_params['n_layers']
            hidden_dims = [best_params[f'hidden_dim_{i}'] for i in range(n_layers)]
            activation = best_params['activation']
            lr = best_params['lr']
        else:
            hidden_dims = [128, 64]
            activation = 'relu'
            lr = 1e-3
        model = DNNDepthNet(X_train.shape[1], hidden_dims=hidden_dims, activation=activation)

    elif model_type == 'CNN1D':
        if best_params:
            conv1_out = best_params['conv1_channels']
            conv2_out = best_params['conv2_channels']
            conv3_out = best_params['conv3_channels']
            conv4_out = best_params['conv4_channels']
            lr = best_params['lr']
        else:
            conv1_out = 32
            conv2_out = 64
            conv3_out = 128
            conv4_out = 64
            lr = 1e-3
        model = CNN1DDepthNet(X_train.shape[1], conv1_out=conv1_out, conv2_out=conv2_out, conv3_out=conv3_out, conv4_out=conv4_out)

    else:
        raise ValueError("Unsupported model type: must be 'DNN' or 'CNN1D'")

    loss_fn = LossFunction(loss_type=loss_type)
    model = model.to(device)

    print(f"--- {model_type},  {loss_type} ---")

    if loss_type == 'mse':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    elif loss_type == 'mse+phy': 
        optimizer = torch.optim.Adam([
                    {'params': model.parameters()},
                    {'params': loss_fn.lambda_phy, 'lr': 0.001}
                ], lr=0.001, weight_decay=1e-4)
    train_losses = []
    val_losses = []
    history = {
        'train_total_loss': [],
        'train_data_loss': [],
        'train_phy_loss': [],
        'val_total_loss': [],
        'val_data_loss': [],
        'val_phy_loss': []
    }
    early_stopping = EarlyStopping(patience=patience)
    for epoch in range(epochs):
        model.train()
        epoch_train_total = 0.0
        epoch_train_data = 0.0
        epoch_train_phy = 0.0
        total_samples = 0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            pred = model(batch_x)
            
            data_loss = F.mse_loss(pred, batch_y)
            phy_loss = physical_loss(pred, batch_x, batch_y, scaler)
            total_loss = loss_fn(model, pred, batch_y, batch_x, scaler)
 
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            batch_size = batch_x.size(0)
            epoch_train_total += total_loss.item() * batch_size
            epoch_train_data += data_loss.item() * batch_size
            epoch_train_phy += phy_loss.item() * batch_size
            total_samples += batch_size

        avg_train_total = epoch_train_total / total_samples
        avg_train_data = epoch_train_data / total_samples
        avg_train_phy = epoch_train_phy / total_samples

        model.eval()
        with torch.no_grad():
            X_test = X_test_tensor.to(device)
            Y_test = Y_test_tensor.to(device)

            pred_val = model(X_test_tensor.to(device))

            val_data_loss = F.mse_loss(pred_val, Y_test)
            val_phy_loss = physical_loss(pred_val, X_test, Y_test, scaler)
            val_total_loss = loss_fn(model, pred_val, Y_test_tensor.to(device), X_test_tensor.to(device), scaler)
                              
        train_losses.append(avg_train_total)
        val_losses.append(val_data_loss)
 
        history['train_total_loss'].append(avg_train_total)
        history['train_data_loss'].append(avg_train_data)
        history['train_phy_loss'].append(avg_train_phy)
        history['val_total_loss'].append(val_total_loss.item())
        history['val_data_loss'].append(val_data_loss.item())
        history['val_phy_loss'].append(val_phy_loss.item())

        if epoch % 20 == 0:
            print(f'Epoch {epoch + 1}:')
            print(f'  Train - Total: {avg_train_total:.6f}, Data: {avg_train_data:.6f}, Phy: {avg_train_phy:.6f}')
            print(
                f'  Val   - Total: {val_total_loss.item():.6f}, Data: {val_data_loss.item():.6f}, Phy: {val_phy_loss.item():.6f}')

        early_stopping(val_total_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    torch.cuda.empty_cache()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train_tensor.to(device)).cpu().numpy()
        test_pred = model(X_test_tensor.to(device)).cpu().numpy()
        Y_train = Y_train.cpu().numpy() if isinstance(Y_train, torch.Tensor) else Y_train
        Y_test = Y_test.cpu().numpy() if isinstance(Y_test, torch.Tensor) else Y_test

    train_mse = mean_squared_error(Y_train, train_pred)
    test_mse = mean_squared_error(Y_test, test_pred)
    train_rmse = np.sqrt(train_mse)
    test_rmse = np.sqrt(test_mse)
    train_mae = mean_absolute_error(Y_train, train_pred)
    test_mae = mean_absolute_error(Y_test, test_pred)
    train_mape = mean_absolute_percentage_error(Y_train, train_pred)
    test_mape = mean_absolute_percentage_error(Y_test, test_pred)
    train_r2 = r2_score(Y_train, train_pred)
    test_r2 = r2_score(Y_test, test_pred)

    metrics = {
        'train_rmse': train_rmse,
        'train_mae': train_mae,
        'train_mape': train_mape,
        'train_r2': train_r2,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_mape': test_mape,
        'test_r2': test_r2
    }
    return scaler, model, metrics, (train_losses, val_losses, Y_train, train_pred, Y_test, test_pred), X_test_scaled, history

def repeat_train_and_evaluate(
        image_path, image_data,
        X_train, Y_train, 
        X_test, Y_test, 
        output_path,
        model_type='DNN',
        best_params=None,
        n_repeats=5,
        loss_type='mse',
        image_predict=False,
):
    metrics_list = []
    preds_all = []
    if best_params is None:
         raise ValueError("best_params must be provided to repeat_train_and_evaluate")

    for repeat in range(n_repeats):
        seed_everything(seed=42 + repeat) 
        print(f"\n>>> [{model_type}] Training Run {repeat + 1}/{n_repeats}")

        scaler, model, metrics, (train_losses, val_losses, Y_train, train_pred, test_true, test_pred), X_test_scaled, history = train_model(
            X_train, Y_train, 
            X_test, Y_test, 
            model_type=model_type,
            best_params=best_params,
            loss_type=loss_type,
            save_path=os.path.join(output_path, f'model_{model_type}_{loss_type}_run{repeat + 1}.pth')
        )

        metrics['model'] = model_type
        metrics['run'] = repeat + 1
        metrics_list.append(metrics)

        model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            preds = model(X_test_tensor).cpu().numpy().flatten()
        preds_all.append((Y_test.values, preds))

        if image_predict:
            predict_image(scaler, model, image_path, image_data, output_path, model_type, loss_type,
                          repeat_idx=repeat + 1)

    df_metrics = pd.DataFrame(metrics_list)
    df_mean = df_metrics.groupby('model').mean().reset_index()
    df_mean['run'] = 'mean'
    
    print(model)
    print("-" * 30)

    return df_metrics


def load_data(in_situ_path, input_image_path, input_iop_path, output_path, task_description, include_iop=False):
    print(f"{task_description}")

    in_situ_data = gpd.read_file(in_situ_path)

    neg_count = (in_situ_data['h_mss'] < 0).sum()
    pos_count = (in_situ_data['h_mss'] >= 0).sum()

    if neg_count > pos_count:
        in_situ_data['h_mss'] = -in_situ_data['h_mss']
        print(f"[INFO] Detected negative dominant. h_mss has been flipped to positive.")

        in_situ_data = in_situ_data[in_situ_data['h_mss'] <= 25].reset_index(drop=True)
        print(f"[INFO] Filtered points <=22m. Remaining points: {len(in_situ_data)}")

    IOP = gdal.Open(input_iop_path).ReadAsArray()

    image = gdal.Open(input_image_path)
    image_data = image.ReadAsArray().astype('float')[:60]  

    from scipy.ndimage import median_filter
    image_data_filtered = np.zeros_like(image_data)
    for i in range(image_data.shape[0]):  
        image_data_filtered[i] = median_filter(image_data[i], size=3) 
    image_data = image_data_filtered
    if (image_data[27] > 1).any():
        image_data /= 10000
    else:
        image_data 
    image_data[image_data < 0.00002] = 0.00002

    if include_iop:
        print("[INFO] WITH IOP")
        combined_data = np.vstack([image_data, IOP]) 
    else:
        print("[INFO] NO IOP")
        combined_data = image_data 
    
    coords = np.vstack((in_situ_data['UTM49X'], in_situ_data['UTM49Y'])).T
    trans = image.GetGeoTransform()
    inv_trans = np.linalg.inv([[trans[1], trans[2]], [trans[4], trans[5]]])
    rc = np.dot(coords - np.array([trans[0], trans[3]]), inv_trans).astype(int)

    pixel_dict = {}
    for idx, (col, row) in enumerate(rc):
        if 0 <= row < image.RasterYSize and 0 <= col < image.RasterXSize:
            key = (row, col)
            if key not in pixel_dict:
                pixel_dict[key] = []
            pixel_dict[key].append(in_situ_data['h_mss'][idx])

    bands = [[] for _ in range(combined_data.shape[0])]
    H = []

    for (row, col), depths in pixel_dict.items():
        for band_index in range(combined_data.shape[0]):
            bands[band_index].append(combined_data[band_index, row, col])
        H.append(np.median(depths))

    data_dict = {f'band_{i + 1}': bands[i] for i in range(len(bands))}
    data_dict['H'] = H
    dataset = pd.DataFrame(data_dict)

    csv_path = os.path.join(output_path, f'dataset_full.csv')
    dataset.to_csv(csv_path, index=False)

    X = dataset.iloc[:, :-1] 
    Y = dataset.iloc[:, -1]

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42)

    imputer = SimpleImputer(strategy='constant', fill_value=0.0002)
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    return combined_data, X_train, Y_train,  X_test, Y_test


def predict_image(scaler, model, image_path, image_data, save_dir, model_type, loss_type='mse', repeat_idx=None,
                  batch_size=1024): 
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    height, width = image_data.shape[1:] 

    flat_data = image_data.reshape(image_data.shape[0], -1).T
    flat_data_scaled = scaler.transform(flat_data)

    preds = []
    with torch.no_grad():
        for i in range(0, len(flat_data), batch_size):
            batch = flat_data_scaled[i:i + batch_size]
            batch_tensor = torch.tensor(batch, dtype=torch.float32).to(device)
            batch_preds = model(batch_tensor).squeeze().cpu().numpy()
            preds.append(batch_preds)

    preds = np.concatenate(preds, axis=0)
    depth_map = preds.reshape(height, width)
    depth_map = np.clip(depth_map, 0, 30)

    img = gdal.Open(image_path)
    driver = gdal.GetDriverByName('GTiff')

    if repeat_idx is not None:
        save_name = f'predict_depth_{model_type}_{loss_type}_run{repeat_idx}.tif'
    else:
        save_name = f'predict_depth_{model_type}_{loss_type}.tif'

    save_path = os.path.join(save_dir, save_name)

    out_raster = driver.Create(save_path, width, height, 1, gdal.GDT_Float32)
    out_raster.SetGeoTransform(img.GetGeoTransform())
    out_raster.SetSpatialRef(img.GetSpatialRef())
    out_raster.SetProjection(img.GetProjection())
    out_raster.GetRasterBand(1).WriteArray(depth_map)
    out_raster.FlushCache()

    print(f"[INFO] Depth prediction saved to {save_path}")
    return depth_map

def plot_metrics_summary(df_total, output_path):
    df_plot = df_total[df_total['run'] != 'mean'].copy()
    df_plot['run'] = df_plot['run'].astype(int)

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='run', y='test_rmse', hue='model', style='loss_type', markers=True, dashes=False)
    plt.title('Test RMSE over Runs (All Models)', fontsize=30)
    plt.xlabel('Run', fontsize=30)
    plt.ylabel('Test RMSE', fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'test_rmse_summary_all_models.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_plot, x='run', y='test_mae', hue='model', style='loss_type', markers=True, dashes=False)
    plt.title('Test MAE over Runs (All Models)', fontsize=30)
    plt.xlabel('Run', fontsize=30)
    plt.ylabel('Test MAE', fontsize=30)
    plt.grid(True)
    plt.legend(fontsize=22)
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'test_mae_summary_all_models.png'))
    plt.close()

    for model_name in df_plot['model'].unique():
        df_model = df_plot[df_plot['model'] == model_name]

        # RMSE
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_model, x='run', y='test_rmse', hue='loss_type', markers=True, dashes=False)
        plt.title(f'{model_name} - Test RMSE over Runs', fontsize=30)
        plt.xlabel('Run', fontsize=30)
        plt.ylabel('Test RMSE', fontsize=30)
        plt.grid(True)
        plt.legend(fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{model_name}_test_rmse.png'))
        plt.close()

        # MAE
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_model, x='run', y='test_mae', hue='loss_type', markers=True, dashes=False)
        plt.title(f'{model_name} - Test MAE over Runs', fontsize=30)
        plt.xlabel('Run', fontsize=30)
        plt.ylabel('Test MAE', fontsize=30)
        plt.grid(True)
        plt.legend(fontsize=22)
        plt.tick_params(axis='both', which='major', labelsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(output_path, f'{model_name}_test_mae.png'))
        plt.close()

def plot_loss_comparison_bar(df_mean, output_path):

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_mean, x='loss_type', y='test_rmse', hue='model')
    plt.title('Average Test RMSE Comparison across Loss Types', fontsize=26)
    plt.xlabel('Loss Type', fontsize=30)
    plt.ylabel('Test RMSE', fontsize=30)
    plt.legend(fontsize=22, loc='lower left')
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'loss_comparison_rmse.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_mean, x='loss_type', y='test_mae', hue='model')
    plt.title('Average Test MAE Comparison across Loss Types', fontsize=26)
    plt.xlabel('Loss Type', fontsize=30)
    plt.ylabel('Test MAE', fontsize=30)
    plt.legend(fontsize=22, loc='lower left')
    plt.tick_params(axis='both', which='major', labelsize=24)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'loss_comparison_mae.png'))
    plt.close()

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main():

    seed_everything(42)

    task_description = "for testing PHY-SDB on EnMAP data"

    output_path = r'F:\HOPE\HOPE_L2025.04.02_Enmap\result\IOP_NN_SDB\github_test_ENMAP_LY_IS2_batch256_fval_band_ratio_IOP02_1'
    os.makedirs(output_path, exist_ok=True)

    sys.stdout = Logger(os.path.join(output_path, 'training_log.txt'))

    # Lingyang
    in_situ_path = r'Lingyang_IS2_49N.dbf'
    input_image_path = r'enmap_lingyang.tif'
    input_iop_path = r'lingyang_IOPs.tif'
    param_path = r'models_params.csv'

    image_data, X_train, Y_train,  X_test, Y_test = load_data(
        in_situ_path, input_image_path, input_iop_path, output_path, task_description, include_iop=False)

    best_params_collection = {}
    if os.path.exists(param_path):
        print("[INFO] Loading existing best model parameters...")
        best_df = pd.read_csv(param_path)
        
        for model_name in ['DNN', 'CNN1D']:
            try:
                params_str = best_df[best_df['Model'] == model_name]['Best_Parameters'].values[0]
                best_params_collection[model_name] = ast.literal_eval(params_str)
                print(f"[INFO] Loaded params for {model_name}.")
            except (IndexError, KeyError):
                print(f"[WARNING] Parameters for {model_name} not found in CSV. Will run optimization.")
                best_params_collection[model_name] = None
    else:
        print("[INFO] Parameter file not found. Optimizing all models.")
        for model_name in ['DNN', 'CNN1D']:
            best_params_collection[model_name] = None

    if best_params_collection.get('DNN') is None:
        print("\n[OPTIMIZING] DNN...")
        best_params_collection['DNN'] = optimize_mlp_hyperparams(X_train, Y_train, X_test, Y_test, input_dim=X_train.shape[1])
    if best_params_collection.get('CNN1D') is None:
        print("\n[OPTIMIZING] CNN1D...")
        best_params_collection['CNN1D'] = optimize_cnn1d_hyperparams(X_train, Y_train, X_test, Y_test, input_dim=X_train.shape[1])

    print("\n[INFO] Saving all best parameters to CSV...")
    results_df = pd.DataFrame({
        'Model': list(best_params_collection.keys()),
        'Best_Parameters': [str(p) for p in best_params_collection.values()]
    })
    results_df.to_csv(param_path, index=False)

    loss_types = ['mse', 'mse+phy']  
    n_repeats = 10
    all_metrics= [] 

    model_configs = {
        'DNN': {
            'model_type': 'DNN',
            'best_params': best_params_collection.get('DNN'),
        },
        'CNN1D': {
            'model_type': 'CNN1D',
            'best_params': best_params_collection.get('CNN1D'),
        }
    }

    models_to_run = ['DNN','CNN1D']  
    for loss_type in loss_types:
        print(f"=== Training with Loss Type: {loss_type} ===")

        loss_results = []

        for model_name in models_to_run:
            config = model_configs[model_name]

            df_model = repeat_train_and_evaluate(
                input_image_path, image_data,
                X_train, Y_train,
                X_test, Y_test,
                output_path,
                model_type=config['model_type'],
                best_params=config['best_params'],
                n_repeats=n_repeats,
                loss_type=loss_type,
                image_predict=False,
            )

            df_model['model_type'] = model_name
            loss_results.append(df_model)

        df_loss = pd.concat(loss_results, axis=0)
        df_loss['loss_type'] = loss_type  

        all_metrics.append(df_loss)  

    df_total = pd.concat(all_metrics, axis=0)

    metric_columns = [
        'train_rmse', 'train_mae', 'train_mape', 'train_r2',
        'test_rmse', 'test_mae', 'test_mape', 'test_r2'
    ]
    df_mean = df_total.groupby(['model', 'loss_type'])[metric_columns].mean().reset_index()
    df_mean['run'] = 'mean'
    df_final_summary = pd.concat([df_total, df_mean], ignore_index=True)
    df_final_summary.to_csv(os.path.join(output_path, 'metrics_summary_total.csv'), index=False)

    plot_loss_comparison_bar(df_mean, output_path)
    plot_metrics_summary(df_total, output_path)

    print("\n==== All Tasks Completed ====")


if __name__ == '__main__':

    main()



