# Machine Learning on Clinical waiting time

Machine learning aims to develop complex computer algorithms that learn from
data to solve particular tasks without being explicitly programmed (i.e. hardcoded).
These soft-coded algorithms are designed to emulate
human intelligence by adapting to changing environments through repetition
(i.e. experience) such that the prediction (result) becomes better and better at reaching a desired
goal. The concept of this adaption is referred to as training, in which samples of
input data are provided to the algorithm, along with the desired outcomes
for the algorithm to produce. During training, the algorithm self-optimizes in
such a way that it is not only able to produce the desired outcome from input
datasets but also able to generalize outcomes for new, previously-unseen data (sample to population).

- There are two benefits to a successful machine learning algorithm. 
  1. It can be used to substitute laborious and repetitive tasks. 
  2. It can potentially detect non-trivial patterns from large, noisy
or complex datasets better than the average human observer. 


- Techniques
	1. Regularization
		- Technique used to reduce overfitting
		- Adds a regularization term to prevent coefficients from fitting too perfectly
		1. L1 Regularization (Ridge):
			- The regularization term added is the sum of the absolute values of the weights (L1-norm)
			- Computationally inefficient on non-sparse cases
			- Sparse outputs (ex. out of 100 features, only 10 have non-zero coefficients), which naturally leads to feature selection
		1. L2 Regularization (Lasso):
			- The regularization term added is the sum of the squares of the weights (L2-norm)
			- Computationally efficient due to having analytical solutions
			- Non-sparse outputs so no feature selection
		1. Elastic Net:
			- Combines L1 and L2 regularization by adding to the objective function both l1-norm and l2-norm terms
			- Solves the limitations of both techniques
	1. Cross Validation
		- Technique used to reduce overfitting
		- Partition data into training and validation/test set, repeat training with different partitions
	1. Boosting
		- **Ensemble method** (multiple learning algorithms used to obtain better performance than could be obtained from any of the constituent learning algorithms alone) to reduce bias and variance.
		- Converts a set of **weak learners** (classifier only slightly correlated with the true classification (better than random guessing)) into a **single strong learner** (arbitrarily well-correlated with the true classification)
	1. AdaBoost (Adaptive Boosting)
		- Ensemble method
		- Outputs of weak learners are combined into a weighted sum as the final output of the boosted classifier
		- Adopts by adjusting the weights of incorrectly classified instances so the subsequent classifiers focus more on difficult cases
		- Sensitive to noisy data and outliers
	1. Bagging (Bootstrap aggregating)
		- For reducing variance
		- Base regressors are fitted on random subsets of the original dataset and the individual predictions are aggregated to form a final prediction
- Models (using sklearn):
	1. Linear Regression:
		- Fits a best-fit line to the dataset based on continuous random variables.
		- Simple and fast, but overly simplistic for most problems in this case apart from weight and height prediction.
	1. Decision Tree:
		- Divides input space by trying different split points and choosing the best split with the best (lowest) cost (MSE)
		- Best tree depth found to be 2, makes use of only the previous average arrival time (overfitting happens with large height).
		- Overfitting the training data leads to have poor performance on the testing set. We can specify the minimum number of training instances assigned to each leaf node. If a split results in a node of less than minimum count, then the split is rejected and the current node is taken as the final leaf node.
	1. Random Forest:
		- Collection of decision trees which decide on a classification/regression output in a vote fashion. By Strong Law of Large Numbers, it has the advantage of inheritent accuracy over single decision tree (less variance, noise). It also deals with the curse of high dimensionality. However, it is very difficult to interpret the intuition behind results.	
	1. Support Vector Machine:
		 - Classification: Constructs a hyper-plane in high dimensional space to achieve good separation that has the largest distance to the data points of any class.
		 - Regression: Constructs a hyper-plane that best fits the data to within some margin epsilon where data within e are ignored when optimizing the mean errors.
		 - Has the benefit of building hyperplanes in the smallest dimension possible using a kernel function (ie non-linear maps) to map data into a higher dimension.
	1. K-Means (K-Clusters):
		- Has the benefit of conducting unsupervised training
		- Picks k random points for each cluster to create a centroid  (Buitinck et al., 2013).
	
	1. Neural Network:
		- Sigmoid function: 1/(1+e^(-z)) where z = (w_i)(a_i) + bias
		- When weights are too large (positive or negative), z tends to be large as well, driving the output of the sigmoid to the far left (0) or far right (1). These are saturation regions where the gradient/derivative is too small, slowing down learning. Learning slows down when the gradient is small, because the weight upgrade of the network at each iteration is directly proportional to the gradient magnitude.
