# ECT-MNIST
### A final project for MATH 4840: Mathematics of Machine Learning.

The point of this project is to support the idea that data is actually sampled from a high-dimensional manifold. We do this by taking the MNIST dataset of hand drawn digits, map them into R^3 and use mapper to get an embedded simplicial complex and train on the Euler Characteristic Transform (ECT) of these embedded mapper complexes. This has yielded good (~ 97% accuracy) results and gives basically the same results as just using ECT of the point complex. 

### Warning: The code is messy! 

### From Scratch Usage: 
- Clone the repo
- Install dependencies
- Run the MNIST Loader script
- Run the Complex Processing Script
- Run the cnn_training script
   - make sure the paths line up and the number of directions/thresholds make sense for your machine
 
### Interactive Application Demo:
Just run the digit_recognition_app.py file (make sure you have necessary packages installed).
![image](https://github.com/user-attachments/assets/0bb77d22-ff6c-49b8-88a2-2008234e30ec)

Newly added: A slider to visualize the simplex at different threshhold values:

![image](https://github.com/user-attachments/assets/7047d99c-68d5-4ae5-b0ad-cdf8ce62f242)


The application will take your drawing and do some preprocessing to have a similar format to the MNIST dataset. It will then display the models predictions for each digit. The application will display the point complex, mapper complex, and ECT heatmap for your drawing. The dotted line on the ECT heatmap is the column with least difference from the 'exemplar' digits (the entry in the MNIST dataset closest to the mean of that particular label). It then displays the corresponding direction on the point/mapper complex plots and overlays a copy of your drawing the transparency scaled with the dot product.

- The application relies on the html files output during training
   - if you ran your own training from scratch be sure to include the visualization_digit_x.html files inside of the /app/models/exemplars folder. Also make sure to edit the number of directions and file paths to match what you did.
