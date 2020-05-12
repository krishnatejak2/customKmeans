---
marp: true
pagination : true
theme: default
class : default
style: |
  section {
    background-color: #FFFF;
    
  }
---

# Constrained Clustering

Krishna Teja

---

# Problem Statement



* How to enforce known rules/domain knowledge into the unsupervised learning and convert it to a semi-supervised learning?

  - Say we know some conditions about data points, 
  - some information which could typically be used for the betterment of unsupervised clustering?

<!-- Image from output of the data -->

---

# Random Data Generation

1. Generate 2D Multivariate normal data
2. Simple start - 2 clusters, easily separable
3. Establish GroundTruth based on the desired conditions ( x>0 & y<0 )
4. Use Ground truth for constraint generation or evaluate Performance Metrics

![ExpectedOutput](./images/ExpectedOutput.png)

---

# Metrics 


* **NMI - Normalised Mutual Information**
* **Rand Score - proxy for Accuracy**

---

# NMI Explained

* **NMI tells about the reduction in the entropy of the class labels we get if we know the cluster labels**  

![NMI](./images/NMI_1.png)

---

# NMI Explained (contd.)
![NMI](./images/NMI_2.png)

## Python to the rescue

* Sklearn - **normalized_mutual_info_score**
* NMI - [0,1], higher the better

---

# Existing Literature

![Literature](./images/Literature_Algorithms.png)


<!-- Multiple papers suggesting the implementations being done in the past and fwe Dee plearning pulciations are present too-->

---

# Interesting Algorithms with references


1. Cop-Kmeans (Constrained KMeans) [[1]]('https://www.cs.cmu.edu/~./dgovinda/pdf/icml-2001.pdf')
2. PCKMeans (Pairwise Constrained KMeans) & Others [[2]]('https://www.researchgate.net/publication/286812684_Semi-supervised_clustering_with_pairwise_and_size_constraints?enrichId=rgreq-0d2b185b378046c7f808e88656061d70-XXX&enrichSource=Y292ZXJQYWdlOzI4NjgxMjY4NDtBUzozMjI3OTM4NTIzNDIyNzVAMTQ1Mzk3MTQ1OTkzNg%3D%3D&el=1_x_3&_esc=publicationCoverPdf')
3. PreIdentify and Kmeans



# Implemented Algorithms

1. Cop-Kmeans (Constrained KMeans)
2. PreIdentify and Kmeans

---
# Cop-KMeans 


![width:600px height:600px](./images/CopKMeans_Algo.png
)

![bg right width:500px height:500px](./images/Labels.png)

<!-- <img src="./CopKMeans_Algo.png" width="200" height = "400"></img> -->

---

# Cop-KMeans - Implemented Algorithms(contd.)

### Steps Involved : 
1. Generate all Must-Link and Cannot-Link constraints from Ground Truth of the data
2. ML - datapoints in one cluster
3. CL - datapoints from different cluster
4. When the whole GT is known, every combination can be achieved
5. Transitive closure should be oberved when making constraints
6. In reality, we might know less than 5% of the GT
7. So we take these available info from GT(1%) and make them into constraints
8. Run the algorithm as mentioned where constraints are not violated along with finding the closest cluster

---

# Data & Constraints Information

**Data :** 
- 1000 records
- 2 dimensions
- 3 Labels

**ML & CL** - 499500 

* 1 % - 4995
* 0.1 % - 499.5
* 0.01 % - 49.95


---

# PreIdentify-KMeans - Implemented Algorithms(contd.)

### Steps Involved : 
1. Pre-label all the data points with the desired cluster tag
2. Find the centroid of that specific cluster
3. Find the remaning centroids on all of the remaining data based on distance
4. Converge when the centroids do not move - as usual

--- 
# Results : Cop-Kmeans - 0.01 % constraints
<!--Left hand side -->

![COP_Output width:1200px height:400px](./images/COP-kmeans_Output_0.0001.png)
 * NMI ~ 0.84

---

 # Results : Cop-Kmeans - 0.1 % constraints
<!--Left hand side -->

![COP_Output width:1200px height:400px](./images/COP-kmeans_Output_0.001.png)
 * NMI ~ 0.8

---

# Results : Cop-Kmeans - 1 % constraints
<!--Left hand side -->

![COP_Output width:1200px height:400px](./images/COP-kmeans_Output_0.01.png)
 * NMI ~ 1.0

---
<!--Right hand side -->
# Results : PreIdentify-Kmeans

![COP_Output width:1200px height:400px](./images/KMeans_Predict_Output.png)
* NMI = 0.782
---
# Limitations

### Cop-Kmeans
1. Cop-KMeans is time consuming
2. Sometimes, does not converge after processing for a long time - due to impractical constraints - need for correct domain knowledge

### PreIdentify-Kmeans
1. Pre-Idenifying the data points with their labels is exactly not part of unsupervised/semi-supervised for that cluster
2. There is no clustering happening in the selected portion, just finding a centroid
3. Always need labels for the special cluster

--- 

# References:
* https://github.com/Behrouz-Babaki/COP-Kmeans
* https://en.wikipedia.org/wiki/Mutual_information
* [COP-Kmeans](https://www.cs.cmu.edu/~./dgovinda/pdf/icml-2001.pdf)
* [Constrined CLustering - PCKS - Size constraints](https://www.researchgate.net/publication/286812684_Semi-supervised_clustering_with_pairwise_and_size_constraints?enrichId=rgreq-0d2b185b378046c7f808e88656061d70-XXX&enrichSource=Y292ZXJQYWdlOzI4NjgxMjY4NDtBUzozMjI3OTM4NTIzNDIyNzVAMTQ1Mzk3MTQ1OTkzNg%3D%3D&el=1_x_3&_esc=publicationCoverPdf)


# Topics:
* KL-Divergence
* Jensen-Shannon Divergence
* Jensens Inequality
* ELBO
* Projection Gradient

---

# Thank you
https://github.com/krishnatejak2/customKmeans