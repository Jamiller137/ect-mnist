from collections import defaultdict
from itertools import combinations
from typing import Any
import h5py
import numpy as np
from sklearn.cluster import AffinityPropagation
from zen_mapper import mapper
from zen_mapper.komplex import Simplex
from zen_mapper.cluster import sk_learn
from zen_mapper.cover import Width_Balanced_Cover
import tqdm

class SimplicialComplex:
    def __init__(self):
        self._simplices = defaultdict(set)
        self.vertex_coords = {}
        self.vertex_functions = {}
        self.simplex_functions = {}

    def add_vertex(self, vertex_id, coords):
        self._simplices[0].add(Simplex(vertices=[vertex_id]))
        self.vertex_coords[vertex_id] = np.array(coords)

    def add_simplex(self, vertices, dim):
        for vertex in vertices:
            if Simplex(vertices=[vertex]) not in self._simplices[0]:
                raise ValueError(f"Vertex {vertex} not found in complex")

        simplex = Simplex(vertices=vertices)
        for k in range(1, dim + 1):
            for face in combinations(vertices, k):
                self._simplices[k - 1].add(Simplex(vertices=face))

        if simplex.dim == dim:
            self._simplices[dim].add(simplex)

    def from_mapper_result(self, mapper_result, centroids):
        for i in range(len(mapper_result.nodes)):
            self.add_vertex(vertex_id=i, coords=centroids[i])
        for dim in range(mapper_result.nerve.dim + 1):
            for simplex in mapper_result.nerve[dim]:
                self.add_simplex(vertices=simplex, dim=dim + 1)
        return self

    def set_vertex_function(self, vertex_id, value):
        if Simplex([vertex_id]) not in self._simplices[0]:
            raise ValueError(f"Vertex {vertex_id} not found in complex")
        self.vertex_functions[vertex_id] = value

    def extend_function(self, method="max"):
        for dim in range(1, max(self._simplices.keys()) + 1):
            for simplex in self._simplices[dim]:
                if method == "max":
                    self.simplex_functions[simplex] = max(
                        self.vertex_functions[vertex] for vertex in simplex
                    )

    @property
    def dimension(self):
        return max(self._simplices.keys()) if self._simplices else -1

    def euler_characteristic(self, threshold=None):
        if threshold is None:
            return sum((-1) ** k * len(self._simplices[k]) for k in self._simplices)

        vertex_count = 0
        for v in self._simplices[0]:
            vertex_id = list(v)[0]
            if vertex_id in self.vertex_functions:
                # count vertices based on dot product comparison
                if self.vertex_functions[vertex_id] <= threshold:
                    vertex_count += 1
        return vertex_count

    def validate(self):
        for dim in range(1, self.dimension + 1):
            for simplex in self._simplices[dim]:
                for k in range(1, dim + 1):
                    for face in combinations(simplex, k):
                        face_simplex = Simplex(vertices=face)
                        if face_simplex not in self._simplices[k - 1]:
                            print(f"Face {face_simplex} not found in complex")
                            return False
        return True

def load_single_point_cloud(filename, index):
    with h5py.File(filename, 'r') as f:
        points = f[f"points_3d/points_{index}"][:]
        label = f[f"points_3d/points_{index}"].attrs["label"]
    return points, label

def compute_mapper_graph(point_cloud, dimension=2):
    projection = point_cloud[:, 0] - point_cloud[:, 1]
    cover_scheme = Width_Balanced_Cover(n_elements=10, percent_overlap=0.4)

    preferences = np.random.uniform(low=-0.1, high=0.1, size=point_cloud.shape[0])
    sk = AffinityPropagation(
        damping=0.9,
        max_iter=600,
        convergence_iter=50,
        preference=preferences,
        random_state=42
    )
    clusterer = sk_learn(clusterer=sk)

    result = mapper(
        data=point_cloud,
        projection=projection,
        cover_scheme=cover_scheme,
        clusterer=clusterer,
        dim=dimension
    )
    return result

def compute_centroids(point_cloud, mapper_result):
    centroids = {}
    for vertex_id in range(len(mapper_result.nodes)):
        points = mapper_result.nodes[vertex_id]
        points_in_vertex = point_cloud[points]
        centroids[vertex_id] = np.mean(points_in_vertex, axis=0)
    return centroids

def create_simplicial_complex(result, centroids, point_cloud):
    try:
        complex = SimplicialComplex()
        for vertex_id in range(len(result.nodes)):
            complex.add_vertex(vertex_id=vertex_id, coords=centroids[vertex_id])
            complex.set_vertex_function(vertex_id=vertex_id, value=centroids[vertex_id][2])

        for dim in range(result.nerve.dim + 1):
            for simplex in result.nerve[dim]:
                try:
                    complex.add_simplex(vertices=simplex, dim=dim + 1)
                except ValueError as e:
                    print(f"Error adding simplex {simplex}: {e}")

        complex.extend_function(method="max")
        return complex

    except Exception as e:
        print(f"Error creating simplicial complex: {e}")
        return None

def create_point_cloud_complex(point_cloud):
    try:
        complex = SimplicialComplex()
        for vertex_id, point in enumerate(point_cloud):
            complex.add_vertex(vertex_id=vertex_id, coords=point)

        # validate the complex
        if not complex.validate():
            print("Complex validation failed.")
            return None

        return complex

    except Exception as e:
        print(f"Error creating point cloud complex: {e}")
        return None

def create_processed_data(input_filename, output_filename, n_directions=64, n_thresholds=64, directions=Any, thresholds=Any):
    print("Opening input file...")
    with h5py.File(input_filename, "r") as f:
        n_samples = len(f["points_3d"].keys())
        print(f"Found {n_samples} samples")

        print("Creating output file...")
        with h5py.File(output_filename, "w") as out_f:
            features = out_f.create_dataset(
                "features",
                shape=(n_samples, n_directions, n_thresholds),
                dtype=np.int16,
            )
            labels = out_f.create_dataset("labels", shape=(n_samples,), dtype=np.int16)

            for idx in tqdm.tqdm(range(n_samples), desc="Processing point clouds"):
                points = f[f"points_3d/points_{idx}"][:]
                label = f[f"points_3d/points_{idx}"].attrs["label"]

                complex = create_point_cloud_complex(point_cloud=points)

                for i, direction in enumerate(directions):
                    # compute dot products for all vertices
                    dot_products = [np.dot(complex.vertex_coords[vertex_id], direction) 
                                  for vertex_id in complex.vertex_coords]

                    # set vertex functions to dot product values
                    for vertex_id, dot_prod in enumerate(dot_products):
                        complex.set_vertex_function(vertex_id=vertex_id, value=dot_prod)
                        if idx == 0:  # Print only for first sample
                            print(f"Vertex {vertex_id} dot product: {dot_prod:.6f}")

                    # compute range for thresholds
                    min_val = min(dot_products)
                    max_val = max(dot_products)

                    # create thresholds spanning the full range of dot products
                    local_thresholds = np.linspace(min_val - 1e-6, max_val + 1e-6, n_thresholds)

                    # compute Euler characteristic for each threshold
                    for j, threshold in enumerate(local_thresholds):
                        chi = complex.euler_characteristic(threshold=threshold)
                        features[idx, i, j] = chi
                        if idx == 0:  # Print only for first sample
                            print(f"Direction {i}, Threshold {threshold:.6f}, Chi: {chi}")

                labels[idx] = label

                if idx == 0:
                    print("\nFirst computed feature matrix (Euler characteristics):")
                    print(features[idx])

def fibonacci_sphere(n_points):
    """Generate points on a sphere using the Fibonacci spiral method."""
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians

    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2  # y goes from 1 to -1
        radius = np.sqrt(1 - y * y)  # radius at y

        theta = phi * i  # golden angle increment

        x = np.cos(theta) * radius
        z = np.sin(theta) * radius

        points.append([x, y, z])

    return np.array(points)

def main():
    input_filename = "data/preprocessed_data/mnist_3d_cloud_curvy.h5"
    output_filename = "data/mnist_points_ect_64.h5"

    n_directions = 64
    n_thresholds = 64

    # generate directions using Fibonacci sphere method
    directions = fibonacci_sphere(n_directions)

    # ensure unit vectors
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

    thresholds = np.linspace(start=-1, stop=1, num=n_thresholds)

    print("Creating processed data...")
    create_processed_data(
        input_filename=input_filename,
        output_filename=output_filename,
        n_directions=n_directions,
        n_thresholds=n_thresholds,
        directions=directions,
        thresholds=thresholds
    )
    print(f"Data saved to {output_filename}")


if __name__ == "__main__":
    main()
