# IMPLEMENTATION-OF-K-MEANS-ALGORITHM-IN-A-DISTRIBUTED-SYSTEM

by Erica Brisigotti, Ekaterina Chueva, Sofia Pacheco Garcia, Nadillia Sahputra

This work is the final project of the Management and Analysis of Physics Dataset (mod. B) class from the Physics of Data Master's Degree, held at
the University of Padova during Academic Year 2022-2023 by Professor J. Pazzini. 

The **goals of the project** are to:
- build a Spark cluster of 3 Virtual Machines provided on the OpenStack-based cloud provided by the University of Padova (called [Cloudveneto](https://cloudveneto.it/))
- implement 3 variations of the K-means algorithm (standard – “naive” one, K-means++ and K-means||)
- compare the 3 variations based on the performance over the chosen dataset, to ultimately find the best algorithm. 

K-means is an unsupervised machine learning algorithm, which aims to partition $n$ observations to $k$ clusters.
Each observation belongs to the cluster with the nearest in terms of distance mean (called centroid). 
A Euclidean distance is employed for numerical attributes, while a discrete distance is used for non-numerical ones.

The dataset chosen for the project is [kdcup99 dataset](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset),
which is an artificial dataset representing “bad” and “good” internet connections (intrusions/attacks and normal connections). Each observation consists of a series of attributes, some of which are numeric and some aren't.

### Content of the project

#### 1) Building a Spark cluster

We firstly built a Spark cluster of 3 VMs, specifying one machine as a master and worker at the same time an two others as workers.
Then we made it possible to work using PySpark in a Jupyter Notebook.

#### 2) Importing and preprocessing the dataset

We took the data (consisting of 42 columns and around 500000 rows) and the corresponding targets (classes).
K-means algorithms perform best when provided with comparable size classes. Therefore, we kept only the classes with the most data (>10000 occurences). 

![alt text](https://github.com/EkaterinaChueva/IMPLEMENTATION-OF-K-MEANS-ALGORITHM-IN-A-DISTRIBUTED-SYSTEM/blob/main/class_distribution.png)

We are left with 3 classes – smurf, neptune and normal. After we also wanted to keep only numeric columns, since later we will use the distance between points to classify a point (datum) to a specific class.

#### 3) Algorithms

The following section focuses on implementing the k-means algorithms for k=3: k-means||, k-means++ and naive k-means.
Each algorithm consists of the initialization (different for each specific k-means type) and the Lloyd’s algorithm –
the algorithm for assigning each datum to the nearest centroid.

#### 3.1) K-means||

The idea of the K-means parallel is that in the initialization for the given number of iterations we calculate the possible centroids
and in the end we choose only k=3 centroids among them that have the largest weights.
After this initialization we ended up with k=3 centroids which already have a “good start” and in theory require less Lloyd’s iterations (we prove it in practice as well).
For k-means|| we also found the optimal parameters by analysing the trend of the cost per number of iteration.

#### 3.2) K-means++

The idea of the algorithm is that we choose the first centroid uniformly at random,
then we choose the rest of the centroids from the remaining data points with probability proportional to its squared distance.

#### 3.3) Naive k-means

No fancy initialization for this algorithm, we just chose k=3 centroids uniformly at a random.

#### 4) Comparison of the algorithms

The results are shown in the picture below:

![alt text](https://github.com/EkaterinaChueva/IMPLEMENTATION-OF-K-MEANS-ALGORITHM-IN-A-DISTRIBUTED-SYSTEM/blob/main/comparison.png)

We see that k-means++ is the worst algorithm in terms of minimization of the cost. We then can compare naive k-means and k-means||.
We observe that both algorithms reach minimum of the cost, but the k-means|| takes more time in the initialization.
Although it may look like naive k-means is the best, in reality we cannot rely on this algorithm,
since the result strongly depends on the fully random initialization.
On the contrary, although we pay the cost in time taken for initialization in k-means||, it is more reliable.

#### 5) Evaluating the performance based on the number of partitions


We experimented with the number of partitions and obtained the following results:

![alt text](https://github.com/EkaterinaChueva/IMPLEMENTATION-OF-K-MEANS-ALGORITHM-IN-A-DISTRIBUTED-SYSTEM/blob/main/partitions.png)

We concluded that for implementing the k-means parallel the best number of partitions is 12,
which equals number of workers multiplied by number of cores available for each worker.
