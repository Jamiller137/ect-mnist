# ECT-MNIST
### A final project for MATH 4840: Mathematics of Machine Learning.

The point of this project is to support the idea that data is actually sampled from a high-dimensional manifold. We do this by taking the MNIST dataset of hand drawn digits, map them into R^3 and use mapper to get an embedded simplicial complex and train on the Euler Characteristic Transform (ECT) of these embedded mapper complexes. This has yielded good (~ 97% accuracy) results and gives basically the same results as just using ECT of the point complex. 

### Warning: The code is messy! 

### From Scratch Usage: 
- Clone the repo
- Run the MNIST Loader script
- Run the Complex Processing Script
- Run the cnn_training script
   - make sure the paths line up and the number of directions/thresholds make sense for your machine
 
### Interactive Application Demo:
Just run the digit_recognition_app.py file (make sure you have necessary packages installed. 

- The application relies on the html files output during training
   - if you ran your own training from scratch be sure to include the visualization_digit_x.html files inside of the /app/models/exemplars folder. Also make sure to edit the number of directions and file paths to match what you did.
