import h5py
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from tqdm import tqdm
import json
from datetime import datetime
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).parent.parent))
from scripts.complex_processing import compute_centroids, compute_mapper_graph, create_simplicial_complex


class ECTNet(nn.Module):
    def __init__(self, input_shape):
        super(ECTNet, self).__init__()

        self.input_height, self.input_width = input_shape
        
        self.features = nn.ModuleList([
            
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # inplace ReLU saves memory
            nn.BatchNorm2d(16),
            nn.MaxPool2d(2, 2),

           
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2, 2),

            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
        ])

        # calculate feature dimensions
        h = self.input_height // 4
        w = self.input_width // 4
        self.fc_input_dim = 64 * h * w 

        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  
            nn.Linear(self.fc_input_dim, 256), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        for layer in self.features:
            x = layer(x)
        return self.classifier(x.view(x.size(0), -1))

class ECTClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.exemplars = {}
        self.exemplar_indices = {}
        self.output_dir = os.path.join('data', 'training_output', 
                                     "cnn_"+datetime.now().strftime('%Y%m%d_%H%M%S'))
        os.makedirs(self.output_dir, exist_ok=True)
        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        self.model = None
        self.criterion = nn.CrossEntropyLoss()

    def load_data(self, filename):
        print("Loading data...")
        with h5py.File(filename, 'r') as f:
            features = f['features'][:]
            labels = f['labels'][:]

        self.features_shape = features.shape
        # reshape for CNN input
        features = features.reshape(-1, 1, features.shape[1], features.shape[2])

        # initialize model with correct input shape
        self.model = ECTNet(input_shape=(features.shape[2], features.shape[3])).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        print(f"Input shape: {features.shape}")
        print(f"Number of classes: {len(np.unique(labels))}")

        return features, labels
    
    def plot_training_history(self):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        # accuracy plot
        ax1.plot(self.history['train_acc'], label='Training Accuracy')
        ax1.plot(self.history['val_acc'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy over Epochs')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()

        # loss plot
        ax2.plot(self.history['train_loss'], label='Training Loss')
        ax2.plot(self.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss over Epochs')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.output_dir, 'confusion_matrix.png'))
        plt.close()

    def plot_all_heatmaps(self):
        """Plot all exemplar heatmaps together with the same scale using Plotly."""
        n_exemplars = len(self.exemplars)
        n_cols = 5
        n_rows = (n_exemplars + n_cols - 1) // n_cols

        # find global min and max for consistent scale
        all_values = np.array(list(self.exemplars.values()))
        zmin = all_values.min()
        zmax = all_values.max()

        # create subplots
        fig = make_subplots(
            rows=n_rows, 
            cols=n_cols,
            subplot_titles=[f'Digit {label}' for label in sorted(self.exemplars.keys())],
            shared_yaxes=True,
            shared_xaxes=True
        )

        # plot each heatmap
        for idx, (label, exemplar) in enumerate(sorted(self.exemplars.items())):
            row = idx // n_cols + 1
            col = idx % n_cols + 1

            fig.add_trace(
                go.Heatmap(
                    z=exemplar.T,
                    colorscale='Rainbow',
                    zmin=zmin,
                    zmax=zmax,
                    showscale=(idx == 0),  # Only show colorbar for first heatmap
                ),
                row=row,
                col=col
            )

        # update layout
        fig.update_layout(
            height=300 * n_rows,
            width=1500,
            title_text="All ECT Heatmaps",
            showlegend=False,
        )

        # update axes
        for i in range(1, n_exemplars + 1):
            fig.update_xaxes(showticklabels=False, row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)
            fig.update_yaxes(showticklabels=False, row=(i-1)//n_cols + 1, col=(i-1)%n_cols + 1)

        # save the interactive plot
        fig.write_html(os.path.join(self.output_dir, 'all_heatmaps.html'))

    def train_epoch(self, train_loader, val_loader):
        # Training
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs.float())
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        train_acc = 100. * correct / total
        train_loss = train_loss / len(train_loader)

        # validation
        self.model.eval()
        val_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs.float())
                loss = self.criterion(outputs, targets)

                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        val_acc = 100. * correct / total
        val_loss = val_loss / len(val_loader)

        return train_loss, train_acc, val_loss, val_acc

    def train(self, features, labels):
        
        torch.backends.cudnn.benchmark = True
        self.optimizer = optim.AdamW(
        self.model.parameters(),
        lr=0.001,
        weight_decay=0.01
        )
        
        
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )

        # convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        # create data loaders
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
            pin_memory=True,  # faster data transfer to GPU
            num_workers=4     # parallel data loading
        )
        test_dataset = TensorDataset(X_test, y_test)
        test_loader = DataLoader(test_dataset, batch_size=32)

        print("Training CNN classifier...")
        epochs = 8
        best_val_acc = 0

        for epoch in range(epochs):
            train_loss, train_acc, val_loss, val_acc = self.train_epoch(train_loader, test_loader)

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'Train Loss: {train_loss:.3f} | Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.3f} | Val Acc: {val_acc:.2f}%')

            # save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), os.path.join(self.output_dir, 'best_model.pth'))

        # generate training visuals
        self.plot_training_history()

        # generate predictions for confusion matrix
        self.model.eval()
        with torch.no_grad():
            y_pred = []
            for inputs, _ in test_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                y_pred.extend(predicted.cpu().numpy())

        y_pred = np.array(y_pred)
        y_test_np = y_test.numpy()

        # generate confusion matrix
        self.plot_confusion_matrix(y_test_np, y_pred)

        # generate ECT image comparison
        #self.plot_all_heatmaps()

        # save training metrics
        metrics = {
            'final_train_acc': float(train_acc),
            'final_val_acc': float(val_acc),
            'best_val_acc': float(best_val_acc),
            'classification_report': classification_report(y_test_np, y_pred, output_dict=True)
        }

        with open(os.path.join(self.output_dir, 'training_metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=4)

        return X_test.numpy(), y_test_np

    def find_exemplars(self, features, labels, original_indices):
        """Finds entry closest to the mean of each class and saves it as an exemplar."""
        print("Finding exemplars for each class...")
        unique_labels = np.unique(labels)

        for label in tqdm(unique_labels, desc="Processing classes"):
            class_mask = labels == label
            class_features = features[class_mask]
            class_indices = original_indices[class_mask]

            mean_feature = np.mean(class_features, axis=0)
            distances = np.linalg.norm(class_features.reshape(len(class_features), -1) - 
                                    mean_feature.reshape(1, -1), axis=1)
            exemplar_idx = np.argmin(distances)

            self.exemplars[label] = class_features[exemplar_idx].reshape(
                self.features_shape[1], self.features_shape[2]
            )
            self.exemplar_indices[label] = class_indices[exemplar_idx]

    def save_exemplar_visualization(self, label, exemplar, points):
        # create figure with 3 subplots
        fig = make_subplots(
            rows=1, cols=3,
            specs=[[{'type': 'heatmap'}, {'type': 'scatter3d'}, {'type': 'scatter3d'}]],
            subplot_titles=(
                f'ECT Heatmap for Digit {label}',
                'Point Cloud',
                'Mapper Complex'
            )
        )

        # 1. ECT Heatmap
        fig.add_trace(
            go.Heatmap(
                z=exemplar.T,
                colorscale='Rainbow',
                showscale=True
            ),
            row=1, col=1
        )

        # 2. Point Cloud
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points[:, 2],
                    colorscale='Viridis',
                    showscale=False
                ),
                name='Point Cloud'
            ),
            row=1, col=2
        )

        # 3. Mapper Complex
        result = compute_mapper_graph(point_cloud=points, dimension=1)
        centroids = compute_centroids(point_cloud=points, mapper_result=result)
        complex = create_simplicial_complex(result=result, centroids=centroids, point_cloud=points)

        # Background points
        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color='lightgray',
                    opacity=0.5
                ),
                name='Original Points'
            ),
            row=1, col=3
        )

        # Vertices
        vertex_coords = np.array([complex.vertex_coords[v] for v in range(len(result.nodes))])
        vertex_colors = [complex.vertex_functions[v] for v in range(len(result.nodes))]

        fig.add_trace(
            go.Scatter3d(
                x=vertex_coords[:, 0],
                y=vertex_coords[:, 1],
                z=vertex_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=vertex_colors,
                    colorscale='Viridis',
                    showscale=True
                ),
                name='Complex Vertices'
            ),
            row=1, col=3
        )

        # Edges
        for simplex in complex._simplices[1]:
            v1, v2 = simplex
            p1 = complex.vertex_coords[v1]
            p2 = complex.vertex_coords[v2]

            fig.add_trace(
                go.Scatter3d(
                    x=[p1[0], p2[0]],
                    y=[p1[1], p2[1]],
                    z=[p1[2], p2[2]],
                    mode='lines',
                    line=dict(
                        color=complex.simplex_functions.get(simplex, 0),
                        width=2
                    ),
                    showlegend=False
                ),
                row=1, col=3
            )

        fig.update_layout(
            height=600,
            width=1800,
            showlegend=True,
            title_text=f"Comprehensive Visualization for Digit {label}",
            scene2=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            scene3=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            )
        )

        fig.write_html(os.path.join(self.output_dir, f'visualization_digit_{label}.html'))

    def visualize_exemplars(self, point_cloud_file):
        print("Generating visualizations...")
        with h5py.File(point_cloud_file, 'r') as f:
            for label, exemplar in tqdm(sorted(self.exemplars.items()), 
                                      desc="Generating visualizations"):
                points = f[f"points_3d/points_{self.exemplar_indices[label]}"][:]
                self.save_exemplar_visualization(label, exemplar, points)

def main():
    classifier = ECTClassifier()

    # load ECT data
    features, labels = classifier.load_data("data/mnist_mapper_ect_64.h5")
    original_indices = np.arange(len(labels))

    # train classifier and get test data
    X_test, y_test = classifier.train(features, labels)

    # find and visualize exemplars using the full dataset
    classifier.find_exemplars(features, labels, original_indices)
    
    classifier.plot_all_heatmaps()
    classifier.visualize_exemplars("data/preprocessed_data/mnist_3d_cloud_curvy.h5")

    # save the model
    model_path = os.path.join(classifier.output_dir, 'model.pth')
    torch.save(classifier.model.state_dict(), model_path)
    print(f"\nModel saved to: {model_path}")

if __name__ == "__main__":
    main()