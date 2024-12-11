from sklearn.datasets import fetch_openml
import os
import h5py
import numpy as np
import tqdm as tqdm


class MNISTLoader:
    def __init__(self):
        self.X, self.y = fetch_openml(
            name='mnist_784', 
            version=1,
            return_X_y=True,
            as_frame=False
        )
        # normalize pixel values
        self.X = self.X / 255.0
    def get_data(self, train=True):
        if train:
            return self.X[:60000], self.y[:60000]
        else:
            return self.X[60000:], self.y[60000:]
        
    def get_image(self, index):
        # reshape the 784 pixels to a 28x28 image
        return self.X[index].reshape(28, 28)
        
class ImageTo3D:
    def __init__(self):
        self.grayscale_weight = 10 # 10 because vibes (grayscale difference shouldn't matter as much)
        
    def convert_image(self, image):
        height, width = image.shape
        x,y = np.meshgrid(np.arange(width), np.arange(height))
        
        points = np.column_stack(tup=(x.ravel(), y.ravel(), image.ravel()))
        # filter out 'empty' pixels
        filtered_points = points[np.where(points[:, 2] != 0)]
        
        return self.normalize_to_ball(points=filtered_points)
    
    def normalize_to_ball(self, points):
        
        # shift x,y coords to center
        points[:, 0] -= 13.5
        points[:,1] -= 13.5 #(28-1)/2
        
        # scale
        points[:, 2] *= self.grayscale_weight
        max_radius = np.sqrt( 2 * 14 ** 2 + self.grayscale_weight ** 2)
        points /= max_radius
        
        # recenter to mean
        mean = np.mean(points, axis=0)
        points -= mean
        
        # points = manifold_smooth(point_cloud=points)
        points = gaussian_smooth_point_cloud(point_cloud=points)
        
        return points

def gaussian_smooth_point_cloud(point_cloud, sigma=0.1):
    """
    Smooth point cloud using Gaussian weights based on distance.
    """
    from scipy.spatial.distance import cdist
    
    distances = cdist(point_cloud, point_cloud)
    
    weights = np.exp(-distances**2 / (2 * sigma**2))
    weights /= weights.sum(axis=1, keepdims=True)

    # apply weighted average
    smoothed_points = weights @ point_cloud

    return smoothed_points

# not as good as gaussian smoothing it seems, here for posterity
def manifold_smooth(point_cloud, n_components=3):
    """
    Smooth using manifold learning
    """
    from sklearn.manifold import LocallyLinearEmbedding

    lle = LocallyLinearEmbedding(
        n_components=n_components,
        n_neighbors=30,
        method='standard',
        eigen_solver='dense',
    )

    smoothed_points = lle.fit_transform(point_cloud)
    return smoothed_points

def process_and_safe_data():
    save_dir = "data/preprocessed_data"
    os.makedirs(save_dir, exist_ok=True)
    
    loader = MNISTLoader()
    transformer = ImageTo3D()
    
    # get training data
    print("\nLoading training data...")
    X_train, y_train = loader.get_data(train=True)
    
    # create h5 file
    h5_filename = os.path.join(save_dir, "mnist_3d_cloud_curvy.h5")
    with h5py.File(h5_filename, 'w') as f:
        f.create_dataset("labels", data=y_train)
        
        # create groups for OG images and 3D points
        original_group = f.create_group(name="original_images")
        points_group = f.create_group(name="points_3d")
        
        # process with progress bar
        print("\nProcessing images to 3D points...")
        for idx in tqdm.tqdm(iterable=range(len(X_train))):
            image = loader.get_image(index=idx)
            points_3d = transformer.convert_image(image=image)
            
            # save to h5 file
            original_group.create_dataset(name=f"img_{idx}", data=image)
            points_group.create_dataset(name=f"points_{idx}", data=points_3d)
            
            # store the labels
            points_group[f"points_{idx}"].attrs["label"] = y_train[idx]
            
    print(f"\nData has been saved to {h5_filename}")
    return h5_filename

def load_data(filename):
    with h5py.File(filename, 'r') as f:
        labels = f["labels"][:]
        n_samples = len(labels)
        
        all_points = []
        
        # load all the point clouds
        for i in range(n_samples):
            points = f["points_3d"][f"points_{i}"][:]
            all_points.append(points)
            
    return all_points, labels


def main():
    
    print("Starting data processing and saving...")
    h5_filename = process_and_safe_data()
    
    # load and verify
    print("\nVerifying saved data...")
    points, labels = load_data(h5_filename)

    print("\nData structure:")
    print(f"Number of training examples: {len(labels)}")
    print(f"Unique labels: {np.unique(labels)}")

    # accessing some data
    print("\nExample of first point cloud:")
    print(f"Label: {labels[0]}")
    print(f"Points shape: {points[0].shape}")
    print(f"First few points:\n{points[0][:300]}")
    
if __name__ == "__main__":
    main()