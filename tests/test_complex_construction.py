from plotly.subplots import make_subplots
import sys
from pathlib import Path
import numpy as np
sys.path.append(str(Path(__file__).parent.parent))
from scripts.complex_processing import compute_centroids, compute_mapper_graph, create_simplicial_complex, load_single_point_cloud
import networkx as nx
import numpy as np
from sklearn.decomposition import PCA
from zen_mapper import mapper
from zen_mapper.adapters import to_networkx
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import unittest
import numpy as np
from pathlib import Path
from scripts.complex_processing import compute_centroids, compute_mapper_graph, create_simplicial_complex, load_single_point_cloud
import plotly.graph_objects as go


def visualize_results(point_cloud, label, result, centroids, complex):
    # create figure with 2 subplots
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'scatter3d'}, {'type': 'scatter3d'}]],
        subplot_titles=(
            f'MNIST Digit {label}: Point Cloud',
            'Nerve Complex',
            'Embedded Complex'
        )
    )

    # 1. Plot point cloud
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color=point_cloud[:, 2],
                colorscale='Viridis',
                showscale=False
            ),
            name='Point Cloud'
        ),
        row=1, col=1
    )

    # 2. Plot embedded complex
    # Background points
    fig.add_trace(
        go.Scatter3d(
            x=point_cloud[:, 0],
            y=point_cloud[:, 1],
            z=point_cloud[:, 2],
            mode='markers',
            marker=dict(
                size=2,
                color='lightgray',
                opacity=0.5
            ),
            name='Original Points'
        ),
        row=1, col=2
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
        row=1, col=2
    )

    # plot edges (1-simplices)
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
            row=1, col=2
        )

    fig.update_layout(
        height=600,
        width=1200,
        showlegend=True,
        title_text="Mapper Complex Visualization",
        scene=dict(
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        scene2=dict(
            aspectmode='cube',
            camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        )
    )

    return fig
    

def main():
    """Main function to process and visualize MNIST point cloud data"""
    # Set backend to avoid Wayland issues
    import os
    os.environ["QT_QPA_PLATFORM"] = "xcb"

    data_filename = "data/preprocessed_data/mnist_3d_cloud.h5"

    # load point cloud
    point_cloud, label = load_single_point_cloud(data_filename, index=116)

    print("\nData Statistics:")
    print(f"Point cloud shape: {point_cloud.shape}") 
    print("Coordinate ranges:")
    for i, dim in enumerate(["X", "Y", "Z"]):
        print(
            f"{dim} range: [{point_cloud[:,i].min():.3f},{point_cloud[:,i].max():.3f}]"
        )

    # mapper graph
    print("\nComputing Mapper graph...")
    result = compute_mapper_graph(point_cloud, dimension=1)

    print("Computing centroids...")
    centroids = compute_centroids(point_cloud, result)

    
    print("Creating simplicial complex...")
    complex = create_simplicial_complex(result, centroids, point_cloud)

    print("\nComplex Information:")
    print(f"Dimension: {complex.dimension}")
    for dim in range(complex.dimension + 1):
        print(f"{dim}-simplices: {len(complex._simplices[dim])}")

    print("\nEuler Characteristics:")
    print(f"Full: {complex.euler_characteristic()}")
    for threshold in [-0.5, 0.0, 0.25, 0.5]:
        print(f"At threshold {threshold:.2f}: {complex.euler_characteristic(threshold=threshold)}")

    # Visualize
    print("\nGenerating visualization...")
    fig = visualize_results(point_cloud, label, result, centroids, complex)

    fig.show()
    fig.write_html("mapper_visualization.html")



class TestComplexProcessing(unittest.TestCase):

    def setUp(self):
        """Set up test environment and load data."""
        self.data_filename = "data/preprocessed_data/mnist_3d_cloud.h5"
        self.index = 116  # Example index
        self.point_cloud, self.label = load_single_point_cloud(self.data_filename, self.index)

    def test_point_cloud_shape(self):
        """Test if point cloud has the expected shape."""
        self.assertEqual(self.point_cloud.shape[1], 3, "Point cloud should have 3 dimensions.")

    def test_mapper_graph(self):
        """Test the computation of the mapper graph."""
        result = compute_mapper_graph(self.point_cloud, dimension=2)
        self.assertIsNotNone(result, "Mapper graph result should not be None.")
        self.assertGreater(len(result.nodes), 0, "Mapper graph should have nodes.")

    def test_centroids(self):
        """Test the computation of centroids."""
        result = compute_mapper_graph(self.point_cloud, dimension=2)
        centroids = compute_centroids(self.point_cloud, result)
        self.assertEqual(len(centroids), len(result.nodes), "Number of centroids should match number of nodes.")

    def test_simplicial_complex(self):
        """Test the creation of the simplicial complex."""
        result = compute_mapper_graph(self.point_cloud, dimension=2)
        centroids = compute_centroids(self.point_cloud, result)
        complex = create_simplicial_complex(result, centroids, self.point_cloud)
        self.assertIsNotNone(complex, "Simplicial complex should not be None.")
        self.assertGreaterEqual(complex.dimension, 0, "Simplicial complex should have a non-negative dimension.")

    def test_euler_characteristic(self):
        """Test the Euler characteristic computation."""
        result = compute_mapper_graph(self.point_cloud, dimension=2)
        centroids = compute_centroids(self.point_cloud, result)
        complex = create_simplicial_complex(result, centroids, self.point_cloud)
        euler_char = complex.euler_characteristic()
        self.assertIsInstance(euler_char, int, "Euler characteristic should be an integer.")

    def test_visualization(self):
        """Test the visualization generation."""
        result = compute_mapper_graph(self.point_cloud, dimension=2)
        centroids = compute_centroids(self.point_cloud, result)
        complex = create_simplicial_complex(result, centroids, self.point_cloud)
        fig = visualize_results(self.point_cloud, self.label, result, centroids, complex)
        self.assertIsInstance(fig, go.Figure, "Visualization should be a Plotly Figure.")

if __name__ == "__main__":
    unittest.main()
