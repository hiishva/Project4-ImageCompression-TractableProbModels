# Project4-ImageCompression-TractableProbModels
## K-Means Clustering on Images ##
You can use either Java or Python to implement this part.
In this problem, you will use K-means clustering for image compression. We have provided you with two images
  * Display the images after data compression using K-Means clustering for different values of K (2, 5, 10, 15, 20).
  * What are the compression ratios for different values of K? Note that you have to repeat the experiment multiple times with different initializations and report the average as well as the variance in the compression ratio
  * Is there a tradeoff between image quality and degree of compression? What would be a good value for K for each of the two images?

We have provided you with a Java template "KMeans.java" which implements various image input/output operations. You will have to implement the function kMeans in the template. If you want to use Python, you will have to replicate the code in KMeans.java in Python (will take roughly 10-15 minutes). Note that you cannot use scikit learn implementation of k-means for this part

## Tractable Probabilistic Models
In this part, you will implement the following three algorithms and test their performance on the 10 datasets available. Each dataset has three files: ".ts.data" (training data); ".valid.data" (validation data); and .test.data(test data).

* **Tree Bayesian Networks** Use the Chow-Liu algorithm to learn the structure and parameters of the Bayesian network. Use 1-Laplace soothing to ensure you don't have any zeros when computing the mutual information as well as zero probabilities in the model. See section 2 in [Melia and Jordan 2001]. An implementation of Chow-Liu Tree is provided in the code base for you
*  **Mixture of Tree Bayesian Networks using EM** The model is defined as follows. We have one latent variable having k values and each mixture component is a Tree Bayesian Network. Thus, the distribution over the observed variables, denoted by **X** (variables in the dataset) is given by:
  $$P(X=x) = \sum_{i=1}^k p_iT_i(X=x)$$
  where $p_i$ is the probability of the i-th mixture component and $T_i$ is the distribution represented  by the i-th Tree Bayesian network. Write code to learn the structure and parameters of the model using the EM algorithm(in the M-step each mixture component is learned using the Chow-Liu algorithm). Run the EM algorithm until the convergence of 50 iterations whichever is easier. See Section 3 in [Melia and Jordan, 2001]. Use the following values for $k\in \{2,5,10,20\}$. Test performance using the "test set". In the code provided, see MIXTURE.CLT.py, you have to write two functions "learn(...)" and "computeLL(...)"

* **Mixtures of Tree Bayesian networks using Random Forests** The model is defined as above. Learn the structure and parameters of the model using the following Random-Forests style approach. Given two hyper-parameters (k,r), generate k sets of bootstrap samples and learn the i-th Tree Bayesian networks using the i-th set of the Bootstrap samples by randomly setting exactly r mutual information scores to 0 (as before using the Chow-Liu algorithm, with r mutual information scores set to 0 to learn the structure and parameter of the Tree Bayesian network). Select k and r using the validation set and use 1-laplace smoothing. You can use either $p_i = 1/k$  for all $i$ or use any reasonable method (reasonable is extra credit). Describe your (reasonable) method precisely in your report. Does it improve over the baseline approach that uses $p_i = 1/k$

Report Test-set Log Likelihood (LL) score on the 10 datasets available. For EM and Random Forest, choose the hyper-parameters (k and r) using the validation set and then run the algorithms 5 times and report the average and standard deviation. Can you rank the algorithms in terms of accuracy (measured using test set LL) based on your experiments? Comment on why you think ranking makes sense
