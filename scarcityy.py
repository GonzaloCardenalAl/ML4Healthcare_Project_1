import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, accuracy_score, f1_score
from sklearn.metrics import roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score

torch.manual_seed(42)
np.random.seed(42)

def load_data(time_series_path, outcomes_path):
    time_series = pd.read_csv(time_series_path)
    outcomes = pd.read_csv(outcomes_path)
    outcomes = outcomes.rename(columns={'RecordID': 'PatientID'})
    patient_outcomes = dict(zip(outcomes['PatientID'], outcomes['In-hospital_death']))
    time_series['In-hospital_death'] = time_series['PatientID'].map(patient_outcomes)
    time_series = time_series.dropna(subset=['In-hospital_death'])
    return time_series

def prepare_patient_sequences(data, seq_length=49):
    patient_ids = data['PatientID'].unique()
    exclude_cols = ['PatientID', 'Hours', 'Gender', 'ICUType', 'In-hospital_death']
    feature_cols = [col for col in data.columns if col not in exclude_cols]

    sequences, targets, seq_patient_ids = [], [], []

    for patient_id in patient_ids:
        patient_data = data[data['PatientID'] == patient_id].sort_values('Hours')
        outcome = patient_data['In-hospital_death'].iloc[0]

        if len(patient_data) < seq_length:
            continue

        X_patient = patient_data[feature_cols].values[:seq_length]
        sequences.append(X_patient)
        targets.append(outcome)
        seq_patient_ids.append(patient_id)

    return np.array(sequences), np.array(targets), np.array(seq_patient_ids)

class LSTMBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1, dropout_prob=0.6):
        super(LSTMBinaryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=layer_dim, batch_first=True,
                            dropout=dropout_prob if layer_dim > 1 else 0)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class BiLSTMBinaryClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim=1, dropout_prob=0.6):
        super(BiLSTMBinaryClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim,
                            num_layers=layer_dim, batch_first=True,
                            bidirectional=True,
                            dropout=dropout_prob if layer_dim > 1 else 0)
        self.fc = nn.Linear(2 * hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.fc(out)
        return out

def calculate_class_weights(labels, print_info=False):
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels)
    class_counts = torch.bincount(labels.long())
    n_samples = len(labels)
    n_classes = len(class_counts)
    weights = n_samples / (n_classes * class_counts.float())
    return {i: weights[i].item() for i in range(len(weights))}

def train_model(model, train_loader, val_loader, learning_rate, class_weights=None, num_epochs=50, patience=3):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    if class_weights is not None:
        pos_weight = torch.tensor([class_weights[1] / class_weights[0] - 0.5], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        
        train_loss = train_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_val_outputs, all_val_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                all_val_outputs.append(torch.sigmoid(outputs).cpu())
                all_val_targets.append(targets.cpu())

        val_loss = val_loss / len(val_loader.dataset)
        val_losses.append(val_loss)
        
        val_outputs = torch.cat(all_val_outputs).numpy().flatten()
        val_targets = torch.cat(all_val_targets).numpy().flatten()
        val_auc = roc_auc_score(val_targets, val_outputs)
        val_auprc = average_precision_score(val_targets, val_outputs)
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}")

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_lstm_classifier.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                break

    model.load_state_dict(torch.load('best_lstm_classifier.pth'))
    return model, train_losses, val_losses

def evaluate_model_simple(model, test_loader, threshold=0.5, save_figures=True, figure_prefix='model'):
    import matplotlib.pyplot as plt
    import seaborn as sns

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

    predictions = np.concatenate(all_preds).flatten()
    true_values = np.concatenate(all_targets).flatten()
    binary_predictions = (predictions >= threshold).astype(int)

    try:
        auroc = roc_auc_score(true_values, predictions)
    except:
        auroc = np.nan
    try:
        auprc = average_precision_score(true_values, predictions)
    except:
        auprc = np.nan

    accuracy = accuracy_score(true_values, binary_predictions)
    f1 = f1_score(true_values, binary_predictions)
    cm = confusion_matrix(true_values, binary_predictions)

    if save_figures:
        # Confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{figure_prefix}_confusion_matrix.png", dpi=300)

        # ROC and PR curves
        fpr, tpr, _ = roc_curve(true_values, predictions)
        precision_curve, recall_curve, _ = precision_recall_curve(true_values, predictions)

        plt.figure(figsize=(16, 6))
        plt.subplot(1, 2, 1)
        plt.plot(fpr, tpr, label=f'AUC = {auroc:.3f}')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.title("ROC Curve")
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(recall_curve, precision_curve, label=f'AUPRC = {auprc:.3f}')
        plt.title("Precision-Recall Curve")
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{figure_prefix}_roc_pr_curves.png", dpi=300)

    metrics = {
        'AUC': auroc,
        'AUPRC': auprc,
        'Accuracy': accuracy,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }

    return metrics, predictions, binary_predictions, true_values

def subsample_patients(X, y, patient_ids, n_samples):
    np.random.seed(42)
    unique_ids = np.unique(patient_ids)
    sampled_ids = np.random.choice(unique_ids, size=n_samples, replace=False)
    mask = np.isin(patient_ids, sampled_ids)
    return X[mask], y[mask], patient_ids[mask]

def train_xgboost(X_train, y_train, X_val, y_val, X_test, y_test):
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        scale_pos_weight=(len(y_train) - sum(y_train)) / sum(y_train)
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10, verbose=True)
    preds_proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds_proba)
    auprc = average_precision_score(y_test, preds_proba)
    print(f"XGBoost Test AUROC: {auc:.4f}, AUPRC: {auprc:.4f}")
    return model

def run_lstm_experiment(train_data_path, val_data_path, test_data_path,
                        train_outcomes_path, val_outcomes_path, test_outcomes_path,
                        bidirectional=False, hidden_dim=16, layer_dim=2,
                        batch_size=2048, learning_rate=0.001, num_epochs=100,
                        num_labeled_train_patients=None, save_figures=True):

    train_data = load_data(train_data_path, train_outcomes_path)
    val_data = load_data(val_data_path, val_outcomes_path)
    test_data = load_data(test_data_path, test_outcomes_path)

    X_train, y_train, train_patient_ids = prepare_patient_sequences(train_data)
    X_val, y_val, val_patient_ids = prepare_patient_sequences(val_data)
    X_test, y_test, test_patient_ids = prepare_patient_sequences(test_data)

    if num_labeled_train_patients:
        X_train, y_train, train_patient_ids = subsample_patients(X_train, y_train, train_patient_ids, num_labeled_train_patients)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))

    class_weights = calculate_class_weights(y_train)

    train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=batch_size)
    test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=batch_size)

    input_dim = X_train.shape[2]
    model_class = BiLSTMBinaryClassifier if bidirectional else LSTMBinaryClassifier
    model = model_class(input_dim, hidden_dim, layer_dim)

    model, train_losses, val_losses = train_model(model, train_loader, val_loader, learning_rate, class_weights, num_epochs)
    metrics, predictions, binary_predictions, true_values = evaluate_model_simple(model, test_loader)

    print("Test Metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v}")

    return model, metrics

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, 
                 dropout=0.1, output_dim=1, max_seq_length=1000):
        super(TransformerClassifier, self).__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(d_model, output_dim)
    
    def forward(self, x):
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        x = x.transpose(1, 2)
        x = self.global_avg_pool(x).squeeze(-1)
        out = self.fc(x)
        return out

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

def train_transformer_model(model, train_loader, val_loader, learning_rate=0.0005, num_epochs=50, patience=5):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    pos_weight = torch.tensor([5.0], device=device)  # Or use calculate_class_weights if available
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    early_stopping_counter = 0

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)

        train_loss /= len(train_loader.dataset)
        train_losses.append(train_loss)

        model.eval()
        val_loss = 0.0
        all_val_outputs, all_val_targets = [], []

        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * inputs.size(0)
                all_val_outputs.append(torch.sigmoid(outputs).cpu())
                all_val_targets.append(targets.cpu())

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)
        val_outputs = torch.cat(all_val_outputs).numpy().flatten()
        val_targets = torch.cat(all_val_targets).numpy().flatten()
        val_auc = roc_auc_score(val_targets, val_outputs)
        val_auprc = average_precision_score(val_targets, val_outputs)
        print(f"Epoch {epoch+1}: Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}, AUPRC: {val_auprc:.4f}")
        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            torch.save(model.state_dict(), 'best_transformer_classifier.pth')
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(torch.load('best_transformer_classifier.pth'))
    return model, train_losses, val_losses

if __name__ == "__main__":
    train_data_path = "enhanced_sampled_set-a.csv"
    val_data_path = "enhanced_sampled_set-b.csv"
    test_data_path = "enhanced_sampled_set-c.csv"
    train_outcomes_path = "Outcomes-a.txt"
    val_outcomes_path = "Outcomes-b.txt"
    test_outcomes_path = "Outcomes-c.txt"

    for n in [100, 500, 1000]:
        print(f"\nRunning Transformer with {n} labeled patients")
        
        # Load and prepare data
        train_data = load_data(train_data_path, train_outcomes_path)
        val_data = load_data(val_data_path, val_outcomes_path)
        test_data = load_data(test_data_path, test_outcomes_path)
    
        X_train, y_train, train_patient_ids = prepare_patient_sequences(train_data)
        X_val, y_val, _ = prepare_patient_sequences(val_data)
        X_test, y_test, _ = prepare_patient_sequences(test_data)
    
        if n:
            X_train, y_train, _ = subsample_patients(X_train, y_train, train_patient_ids, n)
    
        # Convert to tensors
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.FloatTensor(y_train.reshape(-1, 1))
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.FloatTensor(y_val.reshape(-1, 1))
        X_test_tensor = torch.FloatTensor(X_test)
        y_test_tensor = torch.FloatTensor(y_test.reshape(-1, 1))
    
        # Create DataLoaders
        train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=2048, shuffle=True)
        val_loader = DataLoader(TensorDataset(X_val_tensor, y_val_tensor), batch_size=2048)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=2048)
    
        # Build model
        input_dim = X_train.shape[2]
        model = TransformerClassifier(
            input_dim=input_dim,
            d_model=16,
            nhead=2,
            num_layers=1,
            dim_feedforward=32,
            dropout=0.6,
            output_dim=1
        )
    
        # Train model
        print("Training Transformer model...")
        model, train_losses, val_losses = train_transformer_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            learning_rate=0.0002,
            num_epochs=50
        )
    
        # Evaluate model
        print("Evaluating Transformer model...")
        metrics, predictions, binary_predictions, true_values = evaluate_model_simple(
            model, test_loader, figure_prefix=f'transformer_{n}_samples'
        )
    
        print("\nTransformer Test Metrics:")
        for k, v in metrics.items():
            if k != 'Confusion Matrix':
                print(f"{k}: {v:.6f}")
        print("\nConfusion Matrix:")
        print(metrics['Confusion Matrix'])

    for n in [100, 500, 1000]:
        print(f"\nRunning BiLSTM with {n} labeled patients")
        run_lstm_experiment(
            train_data_path, val_data_path, test_data_path,
            train_outcomes_path, val_outcomes_path, test_outcomes_path,
            bidirectional=True, hidden_dim=24, layer_dim=2,
            batch_size=256, learning_rate=0.001, num_epochs=50,
            num_labeled_train_patients=n
        )
    for n in [100, 500, 1000]:
        print(f"\nRunning LSTM with {n} labeled patients")
        run_lstm_experiment(
                train_data_path, val_data_path, test_data_path,
                train_outcomes_path, val_outcomes_path, test_outcomes_path,
                bidirectional=False, hidden_dim=24, layer_dim=2,
                batch_size=256, learning_rate=0.001, num_epochs=50,
                num_labeled_train_patients=n
            )