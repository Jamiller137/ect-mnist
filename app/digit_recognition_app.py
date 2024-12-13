import tkinter as tk
from tkinter import ttk, filedialog
import numpy as np
from PIL import Image, ImageDraw, ImageTk
import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from app.mnist_model import ECTNet
from scripts.complex_processing_mapper import compute_centroids, compute_mapper_graph, create_simplicial_complex
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import plotly.graph_objects as go


# how we found directions
def fibonacci_sphere(n_points):
    points = []
    phi = np.pi * (3. - np.sqrt(5.)) 
    for i in range(n_points):
        y = 1 - (i / float(n_points - 1)) * 2
        radius = np.sqrt(1 - y * y)
        theta = phi * i
        x = np.cos(theta) * radius
        z = np.sin(theta) * radius
        points.append([x, y, z])
    return np.array(points)

class DrawingApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Digit Recognition App")

        self.current_points = None
        self.current_complex = None
        self.current_direction = None
        self.threshold_var = tk.DoubleVar(value=1.0)
        
        # main window frame
        main_frame = ttk.Frame(root)
        main_frame.pack(padx=10, pady=10)

        # frame for plots
        self.plot_frame = ttk.Frame(main_frame)
        self.plot_frame.grid(row=0, column=2, padx=10, pady=10)

        # initialize matplotlib figures for all plots
        self.point_cloud_fig = Figure(figsize=(5, 5))
        self.point_cloud_canvas = FigureCanvasTkAgg(self.point_cloud_fig, self.plot_frame)
        self.point_cloud_canvas.get_tk_widget().grid(row=0, column=0, pady=5)
        
        self.ect_frame = ttk.Frame(main_frame)
        self.ect_frame.grid(row=0, column=3, padx=10, pady=10)

        self.mapper_fig = Figure(figsize=(5, 5))
        self.mapper_canvas = FigureCanvasTkAgg(self.mapper_fig, self.plot_frame)
        self.mapper_canvas.get_tk_widget().grid(row=1, column=0, pady=5)



        # drawing canvas setup
        self.canvas_size = 280
        self.mnist_size = 28
        self.canvas = tk.Canvas(
            main_frame, 
            width=self.canvas_size, 
            height=self.canvas_size, 
            bg='black',
            cursor="cross"
        )
        self.canvas.grid(row=0, column=0, padx=10, pady=10)

        
        # control frame
        self.control_frame = ttk.Frame(main_frame)
        self.control_frame.grid(row=0, column=1, padx=10, pady=10, sticky='n')

        # threshold slider
        ttk.Label(self.control_frame, text="Threshold:").pack(pady=5)
        self.threshold_slider = ttk.Scale(
            self.control_frame,
            from_=-1.0,
            to=1.0,
            orient='horizontal',
            variable=self.threshold_var,
            command=lambda x: self.on_threshold_change(),
            length=200
        )
        self.threshold_slider.pack(pady=5)
        
        # model selection
        ttk.Label(self.control_frame, text="Select Model:").pack(pady=5)
        self.model_var = tk.StringVar(value="")
        self.model_select = ttk.Combobox(self.control_frame, textvariable=self.model_var)
        self.model_select['values'] = self.find_models()
        self.model_select.pack(pady=5)

        # Buttons!!!!
        ttk.Button(self.control_frame, text="Clear", command=self.clear_canvas).pack(pady=5)
        ttk.Button(self.control_frame, text="Predict", command=self.predict_digit).pack(pady=5)
        ttk.Button(self.control_frame, text="Load Model", command=self.load_model_dialog).pack(pady=5)
        
        
        


        # text widget
        self.result_text = tk.Text(self.control_frame, height=15, width=35)
        self.result_text.pack(pady=5)
        self.result_text.insert('1.0', "Draw a digit and click predict!")

        # drawing state
        self.drawing = False
        self.last_x = None
        self.last_y = None

        # bind mouse events
        self.canvas.bind('<Button-1>', self.start_drawing)
        self.canvas.bind('<B1-Motion>', self.draw)
        self.canvas.bind('<ButtonRelease-1>', self.stop_drawing)

        # model setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None

        # load default model if available
        default_model_path = Path("app/models/mapper_best_model.pth")  # Updated model filename
        if default_model_path.exists():
            self.load_model(str(default_model_path))

        if self.model_select['values']:
            self.model_var.set(self.model_select['values'][0])
            self.load_model(self.model_select['values'][0])
            
            
        # ECT heatmap functionality
        self.ect_fig = Figure(figsize=(5, 5))
        self.ect_canvas = FigureCanvasTkAgg(self.ect_fig, self.plot_frame)
        self.ect_canvas.get_tk_widget().grid(row=0, column=3, pady=5, padx=5, sticky='nsew')
            
    #  how we update visualization from threshold slider changes
    def on_threshold_change(self):
        """Update visualization when threshold changes"""
        if hasattr(self, 'current_complex') and self.current_complex and self.current_direction is not None:
            self.update_plots(self.current_complex, self.current_direction)
    
    def load_exemplar_features(self, predicted_class):
        """Load exemplar features from visualization HTML file"""
        exemplar_path = Path(f"app/models/exemplars/visualization_digit_{predicted_class}.html")

        print(f"Looking for exemplar at: {exemplar_path.absolute()}")

        if not exemplar_path.exists():
            print(f"File not found at {exemplar_path.absolute()}")
            return None

        try:
            # extract ECT features from HTML file
            import re
            with open(exemplar_path, 'r', encoding='utf-8') as f:
                html_content = f.read()

                # Look for the heatmap data in the HTML
                match = re.search(r'"z":\s*(\[\[.*?\]\])', html_content, re.DOTALL)
                if not match:
                    print("Could not find heatmap data in HTML")
                    return None

                try:
                    import json
                    exemplar_features = np.array(json.loads(match.group(1)))
                    return torch.FloatTensor(exemplar_features).unsqueeze(0).unsqueeze(0)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                    return None

        except Exception as e:
            print(f"Error loading exemplar: {e}")
            return None
    
        
    def update_ect_plot(self, features, exemplar_features=None, best_direction_idx=None):
        """Update ECT heatmap plot"""
        self.ect_fig.clear()
        ax = self.ect_fig.add_subplot(111)
        self.ect_fig.subplots_adjust(left=0.15, right=0.95, bottom=0.1, top=0.95)  # Adjust margins

        # plot the ECT heatmap
        im = ax.imshow(
            features[0, 0].numpy().T,  
            aspect='auto',
            cmap='rainbow',
            origin='lower',
            extent=[0, features[0, 0].shape[1], -1, 1] # fix threshold axes to [-1, 1] instead of [0, 64]
        )
        colorbar = self.ect_fig.colorbar(im)
        colorbar.set_label('Euler Characteristic Value')

        if exemplar_features is not None and best_direction_idx is not None:
            # highlight the most similar direction on the ECT heatmap
            ax.axvline(x=best_direction_idx, color='r', linestyle='--', alpha=0.5)

        # axes settings
        ax.set_title('ECT Features')
        ax.set_xlabel('Directions')
        ax.set_ylabel('Thresholds')
        ax.grid(True, linestyle='--', alpha=0.3)

        self.ect_canvas.draw()
            
    def find_models(self):
        """find all .pth files in the models directory"""
        models_dir = Path("app/models")
        models_dir.mkdir(parents=True, exist_ok=True)  # create parent directories if needed

        # create exemplars directory maybe
        exemplars_dir = models_dir / "exemplars"
        exemplars_dir.mkdir(exist_ok=True)

        return [str(model) for model in models_dir.glob("*.pth")]
    
    def load_model_dialog(self):
        """Open a file dialog to select and load a model"""
        filename = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=(("PyTorch models", "*.pth"), ("All files", "*.*"))
        )
        if filename:
            self.load_model(filename)

    def load_model(self, model_path):
        """Load a model from the specified path. Should be a .pth file"""
        try:
            self.model = ECTNet(input_shape=(64, 64)).to(self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Model loaded successfully!\nYou may draw a digit and then click predict...")
        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error loading model: {str(e)}")
            
    def start_drawing(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y
        
    def draw(self, event):
        if self.drawing:
            if self.last_x and self.last_y:
                # reduce line width to be proportional to MNIST scale
                self.canvas.create_line(
                    self.last_x, self.last_y, event.x, event.y,
                    width=50,  # Reduced from 20 to 10 for better scaling
                    fill='white', 
                    capstyle=tk.ROUND, 
                    smooth=tk.TRUE
                )
            self.last_x = event.x
            self.last_y = event.y

    def stop_drawing(self, event):
        self.drawing = False
        self.last_x = None
        self.last_y = None
        
    def clear_canvas(self):
        """Clear both the canvas and the prediction text"""
        self.canvas.delete("all")  # this will remove both the drawing and overlay
        if hasattr(self, 'result_text'):
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', "Draw a digit and click predict!")
        # reset state
        self.current_points = None
        self.current_complex = None
        self.current_direction = None

        # clear the plots
        if hasattr(self, 'point_cloud_fig'):
            self.point_cloud_fig.clear()
            self.point_cloud_canvas.draw()
        if hasattr(self, 'mapper_fig'):
            self.mapper_fig.clear()
            self.mapper_canvas.draw()
    
    
    
    
    def preprocess_image(self):
        """Convert canvas to image and create point cloud"""
        # create PIL image at MNIST size directly
        image = Image.new('L', (self.mnist_size, self.mnist_size), 'black')
        draw = ImageDraw.Draw(image)

        # scale coordinates from 280x280 to 28x28
        scale_factor = self.mnist_size / self.canvas_size

        # first pass: Draw with thicker lines to create core of digit
        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            scaled_coords = [coord * scale_factor for coord in coords]
            draw.line(scaled_coords, fill='white', width=2)

        # second pass: Draw with thinner, semi-transparent lines to simulate MNIST's soft edges
        overlay = Image.new('L', (self.mnist_size, self.mnist_size), 'black')
        draw_overlay = ImageDraw.Draw(overlay)

        for item in self.canvas.find_all():
            coords = self.canvas.coords(item)
            scaled_coords = [coord * scale_factor for coord in coords]
            draw_overlay.line(scaled_coords, fill=60, width=1) 

        # apply very slight blur to smooth transitions
        from PIL import ImageFilter
        overlay = overlay.filter(ImageFilter.GaussianBlur(radius=0.8))

        # combine core and overlay
        image = Image.composite(image, overlay, overlay)

        # convert to numpy array and normalize
        img_array = np.array(image)
        img_array = img_array / 255.0

        # create point cloud
        height, width = img_array.shape
        x, y = np.meshgrid(np.arange(self.mnist_size), np.arange(self.mnist_size))
        points = np.column_stack((x.ravel(), y.ravel(), img_array.ravel()))


        # filter out empty pixels
        self.current_points = points[points[:, 2] > 0]

        # Process/normalize points - should match mnist_loader.py exactly
        self.current_points[:, 0] -= 13.5  # Center x coordinates (28/2 - 0.5)
        self.current_points[:, 1] -= 13.5  # Center y coordinates (28/2 - 0.5)
        self.current_points[:, 2] *= 10    # Scale z coordinates
        max_radius = np.sqrt(2 * 14**2 + 10**2)  # Changed to match mnist_loader.py
        self.current_points /= max_radius
        mean = np.mean(self.current_points, axis=0)
        self.current_points -= mean

        # create ECT from input
        result = compute_mapper_graph(self.current_points, dimension=1)
        centroids = compute_centroids(self.current_points, result)
        complex = create_simplicial_complex(result=result, centroids=centroids, point_cloud=self.current_points)

        # update the plots
        self.update_plots(complex)

        # generate the directions. This should match the directions used in training
        n_directions = 64
        n_thresholds = 64
        directions = fibonacci_sphere(n_directions)
        directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]

        # create matrix
        features = np.zeros((1, n_directions, n_thresholds))
        standardized_thresholds = np.linspace(-1, 1, n_thresholds)

        for i, direction in enumerate(directions):
            # calculate dot products
            dot_products = [np.dot(complex.vertex_coords[v], direction) for v in range(len(result.nodes))]

            # Normalize
            if dot_products:
                max_abs = max(abs(max(dot_products)), abs(min(dot_products)))
                if max_abs > 0:
                    normalized_dots = [dp / max_abs for dp in dot_products]
                else:
                    normalized_dots = dot_products
            else:
                normalized_dots = dot_products

            # Set vertex functions
            for vertex_id, dot_prod in enumerate(normalized_dots):
                complex.set_vertex_function(vertex_id=vertex_id, value=dot_prod)

            complex.extend_function(method="max")

            # Calculate Euler characteristics
            for j, threshold in enumerate(standardized_thresholds):
                chi = complex.euler_characteristic(threshold=threshold)
                features[0, i, j] = chi

        # convert to tensor and add channel dimension
        features = torch.FloatTensor(features).unsqueeze(1)
        return features
    
    def update_plots(self, complex, direction=None):
        threshold = self.threshold_var.get()

        # clear previous plots
        self.point_cloud_fig.clear()
        self.mapper_fig.clear()

        # Point Cloud Plot
        ax1 = self.point_cloud_fig.add_subplot(111, projection='3d')

        if direction is not None:
            # calculate dot products for coloring
            dots = np.dot(self.current_points, direction)
            # filter points based on threshold
            mask = dots <= threshold
            visible_points = self.current_points[mask]
            visible_dots = dots[mask]

            scatter = ax1.scatter(
                visible_points[:, 0],
                visible_points[:, 1],
                visible_points[:, 2],
                c=visible_dots,  # Color by dot product
                cmap='viridis',
                s=2
            )
        else:
            scatter = ax1.scatter(
                self.current_points[:, 0],
                self.current_points[:, 1],
                self.current_points[:, 2],
                c='blue',
                s=2
            )

        ax1.set_title('Point Cloud')
        
        # set fixed viewing angle and scale:
        max_range = np.array([
        self.current_points[:, 0].max() - self.current_points[:, 0].min(),
        self.current_points[:, 1].max() - self.current_points[:, 1].min(),
        self.current_points[:, 2].max() - self.current_points[:, 2].min()
        ]).max() / 2.0

        mid_x = (self.current_points[:, 0].max() + self.current_points[:, 0].min()) * 0.5
        mid_y = (self.current_points[:, 1].max() + self.current_points[:, 1].min()) * 0.5
        mid_z = (self.current_points[:, 2].max() + self.current_points[:, 2].min()) * 0.5

        ax1.set_xlim(mid_x - max_range, mid_x + max_range)
        ax1.set_ylim(mid_y - max_range, mid_y + max_range)
        ax1.set_zlim(mid_z - max_range, mid_z + max_range)

        

        # Mapper Complex Plot
        ax2 = self.mapper_fig.add_subplot(111, projection='3d')

        if direction is not None:
            # calculate dot products for vertices
            vertex_coords = np.array([complex.vertex_coords[v] for v in range(len(complex._simplices[0]))])
            vertex_dots = np.dot(vertex_coords, direction)

            # filter vertices based on threshold
            mask = vertex_dots <= threshold
            visible_vertices = np.where(mask)[0]
            visible_coords = vertex_coords[mask]
            visible_dots = vertex_dots[mask]

            if len(visible_coords) > 0:
                scatter = ax2.scatter(
                    visible_coords[:, 0],
                    visible_coords[:, 1],
                    visible_coords[:, 2],
                    c=visible_dots,  # Color by dot product
                    cmap='viridis',
                    s=50
                )

            # plot edges between visible vertices
            for simplex in complex._simplices[1]:
                v1, v2 = simplex
                if v1 in visible_vertices and v2 in visible_vertices:
                    p1 = complex.vertex_coords[v1]
                    p2 = complex.vertex_coords[v2]
                    # calculate edge color based on max dot product of endpoints
                    edge_value = max(vertex_dots[v1], vertex_dots[v2])
                    if edge_value <= threshold:
                        ax2.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            [p1[2], p2[2]],
                            'b-'
                        )
        else:
            # if no direction, show all vertices and edges
            vertex_coords = np.array([complex.vertex_coords[v] for v in range(len(complex._simplices[0]))])
            ax2.scatter(
                vertex_coords[:, 0],
                vertex_coords[:, 1],
                vertex_coords[:, 2],
                c='red',
                s=50
            )

            for simplex in complex._simplices[1]:
                v1, v2 = simplex
                p1 = complex.vertex_coords[v1]
                p2 = complex.vertex_coords[v2]
                ax2.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    'b-'
                )

        ax2.set_title('Mapper Complex')
        
        #set fixed viewing angle and scale:
        vertex_coords = np.array([complex.vertex_coords[v] for v in range(len(complex._simplices[0]))])
        max_range = np.array([
            vertex_coords[:, 0].max() - vertex_coords[:, 0].min(),
            vertex_coords[:, 1].max() - vertex_coords[:, 1].min(),
            vertex_coords[:, 2].max() - vertex_coords[:, 2].min()
        ]).max() / 2.0

        mid_x = (vertex_coords[:, 0].max() + vertex_coords[:, 0].min()) * 0.5
        mid_y = (vertex_coords[:, 1].max() + vertex_coords[:, 1].min()) * 0.5
        mid_z = (vertex_coords[:, 2].max() + vertex_coords[:, 2].min()) * 0.5

        ax2.set_xlim(mid_x - max_range, mid_x + max_range)
        ax2.set_ylim(mid_y - max_range, mid_y + max_range)
        ax2.set_zlim(mid_z - max_range, mid_z + max_range)
        

        if direction is not None:
            # add direction vector visualization
            scale = 0.5
            center = np.mean(self.current_points, axis=0)

            # plot direction on point cloud and mapper
            for ax in [ax1, ax2]:
                ax.quiver(
                    center[0], center[1], center[2],
                    direction[0], direction[1], direction[2],
                    color='red',
                    length=scale,
                    normalize=True
                )

            # update canvas overlay
            dots = np.dot(self.current_points, direction)
            # normalize dot products to [0, 1] for alpha values
            alphas = (dots - dots.min()) / (dots.max() - dots.min())

            # only show points below threshold
            mask = dots <= threshold
            visible_points = self.current_points[mask]
            visible_alphas = alphas[mask]

            # create overlay image
            overlay = Image.new('RGBA', (self.canvas_size, self.canvas_size), (0, 0, 0, 0))
            draw = ImageDraw.Draw(overlay)

            # convert normalized points back to canvas coordinates
            canvas_points = visible_points.copy()
            # rescale from [-0.5, 0.5] to canvas size
            canvas_points[:, :2] = (canvas_points[:, :2] + 0.5) * self.canvas_size

            # draw red regions with varying alpha
            for point, alpha in zip(canvas_points, visible_alphas):
                x, y = point[:2]
                # ensure coordinates are within canvas bounds
                if 0 <= x < self.canvas_size and 0 <= y < self.canvas_size:
                    alpha_int = int(alpha * 128)  # max alpha of 128 (semi-transparent)
                    radius = 4.5  # size of each point
                    draw.ellipse([x-radius, y-radius, x+radius, y+radius], 
                            fill=(255, 0, 0, alpha_int))

            # convert overlay to PhotoImage and display on canvas
            overlay_photo = ImageTk.PhotoImage(overlay)
            # store reference to prevent garbage collection
            self._overlay_photo = overlay_photo
            # create/update overlay on canvas
            self.canvas.delete('overlay')  # Remove old overlay
            self.canvas.create_image(0, 0, image=overlay_photo, anchor='nw', tags='overlay')
            
        # set fixed angle
        for ax in [ax1, ax2]:
            ax.view_init(elev=-40, azim=-40)

        self.point_cloud_canvas.draw()
        self.mapper_canvas.draw()

    def normalize_points(self, points):
        """Normalize point cloud to unit ball"""
        # center x,y coordinates
        points[:, 0] -= 140  # half of 280
        points[:, 1] -= 140

        # scale z coordinates
        points[:, 2] *= 10  # similar to original preprocessing

        # calculate max radius for normalization
        max_radius = np.sqrt(140**2 + 140**2 + 10**2)
        points /= max_radius

        # recenter to mean
        mean = np.mean(points, axis=0)
        points -= mean

        return points

    
    def predict_digit(self):
        if not self.model:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', "Please load a model first!")
            return

        try:
            # get preprocessed features and create point cloud
            features = self.preprocess_image()

            # prediction
            self.model.eval()
            with torch.no_grad():
                output = self.model(features.to(self.device))
                probabilities = torch.softmax(output, dim=1)[0]
                predicted_class = torch.argmax(probabilities).item()

                # predicted_class should be an integer from 0-9
                predicted_class = int(predicted_class)
                if not (0 <= predicted_class <= 9):
                    raise ValueError(f"Invalid predicted class: {predicted_class}")

                # load exemplar features
                exemplar_features = self.load_exemplar_features(predicted_class)
                if exemplar_features is not None:
                    # both features should be numpy arrays and properly oriented 
                    # removed taking a transposed and fixed it at the beginning
                    current_features = features[0, 0].cpu().numpy()  # (n_directions, n_thresholds)
                    exemplar_features = exemplar_features[0, 0].cpu().numpy()
                    
                    # calculate differences for each direction
                    differences = []
                    for i in range(current_features.shape[0]):
                        diff = np.linalg.norm(current_features[i] - exemplar_features[i])
                        differences.append(diff)

                    best_direction_idx = np.argmin(differences)

                    # get the corresponding direction vector
                    directions = fibonacci_sphere(64)
                    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
                    best_direction = directions[best_direction_idx]

                    # update plots with direction vector and current threshold setting from slider
                    self.current_direction = best_direction
                    result = compute_mapper_graph(self.current_points, dimension=1)
                    centroids = compute_centroids(self.current_points, result)
                    complex = create_simplicial_complex(result=result, centroids=centroids, point_cloud=self.current_points)
                    self.current_complex = complex
                    self.update_plots(complex, direction=best_direction)

                    # update ECT plot
                    self.update_ect_plot(features, exemplar_features, best_direction_idx)
                    
                    
                

                # get all probabilities and sort them
                probs_and_classes = [(prob.item(), idx) for idx, prob in enumerate(probabilities)]
                probs_and_classes.sort(reverse=True)

                # create result text with all probabilities
                result_text = "Predictions:\n"
                for prob, class_idx in probs_and_classes:
                    digit = str(class_idx)
                    if class_idx == probs_and_classes[0][1]:
                        digit = f"**{digit}**"
                    result_text += f"{digit}: {prob:.1%}\n"

                # update text widget toshow estimated probabilities
                self.result_text.delete('1.0', tk.END)
                self.result_text.insert('1.0', result_text)

                # and bold the predicted digit
                start = '1.0'
                while True:
                    start = self.result_text.search('**', start, tk.END)
                    if not start:
                        break
                    end = self.result_text.search('**', start + '+2c', tk.END)
                    if not end:
                        break
                    self.result_text.tag_add('bold', start + '+2c', end)
                    self.result_text.tag_config('bold', font=('TkDefaultFont', 10, 'bold'))
                    self.result_text.delete(end, end + '+2c')
                    self.result_text.delete(start, start + '+2c')
                    start = end + '+2c'

        except Exception as e:
            self.result_text.delete('1.0', tk.END)
            self.result_text.insert('1.0', f"Error predicting digit: {str(e)}")

            
    def create_3d_plots(self, points):

        # create point cloud plot
        point_cloud_fig = go.Figure(data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode='markers',
                marker=dict(
                    size=2,
                    color=points[:, 2],
                    colorscale='viridis',
                    showscale=True
                )
            )
        ])
        point_cloud_fig.update_layout(
            title='Processed Point Cloud',
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=400,
            height=400
        )

        # create mapper complex plot
        result = compute_mapper_graph(point_cloud=points, dimension=1)
        centroids = compute_centroids(point_cloud=points, mapper_result=result)
        complex = create_simplicial_complex(result=result, centroids=centroids, point_cloud=points)

        mapper_fig = go.Figure()

        # add background points
        mapper_fig.add_trace(
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
            )
        )

        # add vertices
        vertex_coords = np.array([complex.vertex_coords[v] for v in range(len(result.nodes))])
        vertex_colors = [complex.vertex_functions[v] for v in range(len(result.nodes))]

        mapper_fig.add_trace(
            go.Scatter3d(
                x=vertex_coords[:, 0],
                y=vertex_coords[:, 1],
                z=vertex_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=vertex_colors,
                    colorscale='viridis',
                    showscale=True
                ),
                name='Complex Vertices'
            )
        )

        # add edges
        for simplex in complex._simplices[1]:
            v1, v2 = simplex
            p1 = complex.vertex_coords[v1]
            p2 = complex.vertex_coords[v2]

            mapper_fig.add_trace(
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
                )
            )

        mapper_fig.update_layout(
            title='Mapper Complex',
            scene=dict(
                aspectmode='cube',
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    center=dict(x=0, y=0, z=0),
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            width=400,
            height=400
        )

        # save plots to temporary HTML files and display them
        point_cloud_fig.write_html("temp_point_cloud.html")
        mapper_fig.write_html("temp_mapper.html")

        self.point_cloud_html.load_file("temp_point_cloud.html")
        self.mapper_html.load_file("temp_mapper.html")

    
def main():
    root = tk.Tk()
    # linter says it's unused... lier!
    app = DrawingApp(root=root)
    root.mainloop()
    
if __name__ == "__main__":
    main()