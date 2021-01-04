**Dimensionality reduction and clustering**

This project focuses on unsupervised machine learning techniques.

The project has two parts:

2. Dimensionality reduction: Perform dimensionality reduction of face shape data using PCA.
3. Clustering: Implement the bag of visual words method for face images using various clustering techniques.


- Dimensionality reduction

In this task, I will use principal component analysis (PCA)
to embed face shape points in a low dimensional space, referred
to as the principal component space. I will use the principal
components to generate new faces by manipulating vectors in the
principal component space. I will also measure how much of the
information is lost when transforming existing face points to and
back from the principal component space.
I will work with the IMM Fontral Face Database created at
DTU. It contains 120 facial images of 12 participants. For each
image, a total of 73 facial landmarks have been manually annotated.

The dataset can be found on the IMM website (http://www2.imm.dtu.dk/pubdb/pubs/3943-full.html).

Task 1: Implementing PCA
The first task is to implement the PCA method as well as functions
for transforming to and from the space defined by the principal
components.

Task 2: Evaluating precision


Using PCA to transform a sample x to a principal component 
space and back again incurs some error. This error
is called the reconstruction error. In this task I will implement a
method for calculating this error and use it to test the effect of increasing 
or decreasing the number of principal components used to construct.

Calculate reconstruction error: 
Implement a function that calculates the reconstruction error given a dataset X,
principle components W, and a mean vector μ.

Plot reconstruction error: When constructing W one may use
a single principal component or all of them. Plot the reconstruction 
error of W for all possible numbers of principle components.

Calculate variance: Create functions that calculate the proportional and cumulative proportional variance.

Plot variance metrics: Plot both the proportional and cumulative proportional variance in a single plot.

Task 3: Generative PCA
PCA can be used for generative processes, where new samples similar to the training data X are desired. 
This requires choosing a instead of finding one by transforming a real
sample. By varying the components it is possible to explore
what information each principle component encodes. I provide
an interface for easy exploration of the effect modifying the new instead has on the resulting face shape.

Exploring the space of the principal components: Read the documentation for SliderPlot in util.py . It is a helper class
for setting up and updating a Matplotlib window that allows you to adjust the components of a vector in the principal component
space. I create an instance as described in the documentation and call plt.show() after instantiation to show the window.


- Clustering

In this part I use the bag-of-visual-words technique to
identify people. 

The technique is divided into the following operations:
1. Decide on format of visual words. In this task, I use
small and evenly spaced windows of image-patches.
2. Learn a set of visual words by applying a clustering algorithm to
all image patches of the training dataset.
3. For a new image, classify its image-patches according to the
identified visual words. The image is then characterised by its
distribution of visual words.
4. This new representation is a vector in the space of possible distributions. 
By examining the distance between images it is possible
to determine how similar they are. In this task, I will
use this to find similar images but it has also been used effectively in object recognition tasks.

Task 1: Clustering setup
Learn a set of visual words using clustering as mentioned.

Load data and generate windows: Load the face images
using the read_image_files function in dset.py . use the
scale setting to adjust the scale for faster computation. Use
image_windows from util.py to create a list of windows for
each image . Finally, merge all windows into a single data matrix.

Train clustering models: Implement code for fitting a Scikit-learn 
sklearn.cluster.KMeans model to the window data.

Plot image results: Use the function plot_image_windows
from util.py to plot the cluster centers found by the k-means
model. They are accessed as <model_obj>.cluster_centers_ .

Try other clustering methods: Try training on sklearn.cluster.MeanShift
and sklearn.cluster.AgglomerativeClustering as well and compare the output clusters.

Task 2: Calculating the distribution for an image
In this task, I will use visual words to find images that are similar 
to each other. First, I will use the visual word clusters found
in the last task to create a word histogram for each image. Similar
images can be found by finding histograms that are geometrically closest to the source.

Predict clusters for each image: For each image in the
dataset, use the trained model to predict ( <model_obj>.predict )
a cluster for each window. The result is an array of cluster in-
dices. Use this array to create a histogram for each image, 
representing the distribution of window clusters in that image.

Set up neighbor search: Scikit-learn provides the class
sklearn.neigbors.NearestNeighbors for finding nearest neigh-bors easily. 
Create an instance of the model and use its fit
method to initialise it to the list of histograms, i.e. call <model>.fit(<histograms>) .

Finding closest neighbors: Use the  NearestNeighbors
method kneighbors(x, k) to find the k nearest neighbors to the histogram x.

Experimentation and visualisation: Experiment with using
different images for x. Visualize the resulting neighbors and
their histograms. Compare results for the different clustering
methods and different window sizes and window strides.
