# Clustering-on-Graph-Dataset

This project focuses on 2 primitive clustering tasks on a dataset containing movies and actors. In Task 1, I've applied kmeans to group similar actors with an additional two cost functions. These two cost functions helpout in determining the ideal ```k``` for our algorithm. 

Firstly, I've visualized the entire dataset with the help of the ```networkx``` and ```StellarGraph``` library, creating a bipartite graph. Consequently, I've created a random walker using [UniformRandomMetaPathWalk](https://stellargraph.readthedocs.io/en/stable/api.html?highlight=UniformRandomMetaPathWalk#stellargraph.data.UniformRandomMetaPathWalk),  that helps in training the W2V model, and moreover, calculating the ```node_ids``` and ```node_embeddings```. ```node_ids``` includes the ids of the movies and actors extracted fromm the walker. Both of these variables help in computing ```actor_nodes```, ```movie_nodes```, ```actor_embeddings```, ```movie_embeddings``` for each ```node_target```. Cost1 and Cost2 functions are calculated with the below formulae: 
![image](https://user-images.githubusercontent.com/64201589/133936606-a794b2b2-54c7-45ee-93a7-95cfc4b8323a.png)
 
Then I apply the kmeans algorithm and iterate it over a list of potential ks and compute the cost functions to find the ideal k. Moreover, I've visualized these newly found d-dimensional ```actor_embeddings``` with T-SNE. Same procedure has been followed for Task 2 but in this one, similar movies are grouped together. Further instructions have been provided in the documentation part of the notebook.

### Libraries needed
You need to install the following methods and libraries: 
```
import networkx as nx
from networkx.algorithms import bipartite
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import stellargraph
# you need to have tensorflow 1.x +
from stellargraph.data import UniformRandomMetaPathWalk
from stellargraph import StellarGraph 
```
**NOTE: Check your decorator version before running the cells and make sure it isnt't more than 5. Keeping the networkx version 2.3 worked fine for me**

#### Link to the course
https://www.appliedaicourse.com/course/11/Applied-Machine-learning-course 

