# IMPLEMENTATION-OF-K-MEANS-ALGORITHM-IN-A-DISTRIBUTED-SYSTEM

by Erica Brisigotti, Ekaterina Chueva, Sofia Pacheco Garcia, Nadillia Sahputra

This work is the final project of the Management and Analysis of Physics Dataset (mod. B) class from the Physics of Data Master's Degree, held at
the University of Padova during Academic Year 2022-2023 by Professor J. Pazzini. 

The **goals of the project** are to:
- build a Spark cluster of 3 Virtual Machines provided on the OpenStack-based cloud provided by the University of Padova ([Cloudveneto](https://cloudveneto.it/))
- implement 3 variations of the K-means algorithm (standard – “naive” one, K-means++ and K-means||)
- compare the 3 variations based on the performance over the chosen dataset, to ultimately find the best algorithm. 

K-means is an unsupervised machine learning algorithm, which aims to partition $n$ observations to $k$ clusters.
Each observation belongs to the cluster with the nearest in terms of distance mean (called centroid). 
A Euclidean distance is employed for numerical attributes, while a discrete distance is used for non-numerical ones.

The dataset chosen for the project is [kdcup99 dataset](https://scikit-learn.org/stable/datasets/real_world.html#kddcup-99-dataset),
which is an artificial dataset representing “bad” and “good” internet connections (intrusions/attacks and normal connections). 
Each observation consists of a series of attributes, which are numeric and non-numeric.

### Content of the project

#### 1) Building a Spark cluster

We first build a Spark cluster of 3 virtual machines: we specify that one machine is both a master and worker, and the remaining two are workers.
We then install all the components necessary to run Python, Jupyter Notebooks, and PySpark on said VMs.

#### 2) Importing and preprocessing the dataset

We import a subset of the data from the Python scikit-learn library: our dataset consists of 42 columns/attributes and around 500000 rows/observations, 
and the corresponding targets/classes. We represent take a look at the distribution of the targets/classes:

![Distribution of targets/classes](https://github.com/EkaterinaChueva/IMPLEMENTATION-OF-K-MEANS-ALGORITHM-IN-A-DISTRIBUTED-SYSTEM/blob/main/class_distribution.png)

K-means algorithms perform best when provided with comparable size classes. Therefore, we decide to keep only the classes with the most data (>10000 occurrences). 
We are left with 3 classes ($k=3$) – smurf, neptune, and normal.

We further modify the dataset by keeping only the numeric columns/attributes. If we had kept all columns/attributes, we would have used a mixed metric 
(defined as the sum of the Euclidean and discrete distances): in that case, the non-numeric columns/attributes would give larger contributions 
than the numeric ones, leading to an imbalanced estimate of the centroid and cluster.

#### 3) Algorithms

The following section focuses on the implementation of the 3 variations of the k-means algorithm: k-means||, k-means++, and naive k-means.
Each algorithm consists of an initialization (specific to that variation) and Lloyd’s algorithm, which assigns each datum to the nearest centroid.

#### 3.1) K-means||

The idea behind the K-means|| initialization is to find a good compromise between the random initialization of the naive approach and the k-means++ initialization, which can be thought as occurring at two ends of a spectrum. From the naive approach, we would like to select multiple points at a time and keep the number of iterations small. From the k++ algorithm, we want to take the non-uniform distribution from which the points are randomly extracted. 

The desired sweet spot is found by tuning the two main parameters of the distribution, which are:
- $l$ is a parameter related to the non-uniform distribution from which new centroids are extracted: the distribution is proportional to the square distance, so that we are more likely to be picking new centroids far away from the ones that we already have
- the number of iterations of the initialization

Because of the fixed number of iterations, this algorithm doesn’t accept the first k centroids that it finds, as the previous algorithms do. 
It accepts the $k$ most important centroids that are found, based on the number of observations that are in the corresponding cluster. 
For this reason, the algorithm is more likely (compared to other random initializations) to be going in the right direction from the beginning of Lloyd's algorithm. This also seems to be the reason why the k-means|| algorithm is more stable, requires fewer Lloyd’s iterations to converge, and is more likely to converge to the global minimum of the cost.

After this initialization we ended up with k=3 centroids which already have a “good start” and in theory require fewer Lloyd’s iterations (we prove it in practice as well).
For k-means|| we also found the optimal parameters by analyzing the trend of the cost per number of iterations.

#### 3.2) K-means++

The idea of the algorithm is that we choose the first centroid uniformly at random,
then we choose the rest of the centroids from the remaining data points with probability proportional to its squared distance.

#### 3.3) Naive k-means

This algorithm extracts $k=3$ centroids randomly from a uniform distribution (without repetition).

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
