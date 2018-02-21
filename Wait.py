import matplotlib.pyplot as plt
import numpy as np
import sys
from time import time
from sklearn import datasets, linear_model, preprocessing, tree, ensemble, svm, neighbors
from sklearn.model_selection import KFold, cross_val_score
from process_data import pickle_load, print_full

train_x = pickle_load('train_x')
train_y = pickle_load('train_y')
test_x = pickle_load('test_x')
test_y = pickle_load('test_y')

train_x = preprocessing.scale(train_x)
test_x = preprocessing.scale(test_x)

base_estimators = [linear_model.LinearRegression(), linear_model.Ridge(), linear_model.Lasso(), linear_model.ElasticNet(), tree.DecisionTreeRegressor()]

def frange(start, end, step):
    tmp = start
    while(tmp < end):
        yield tmp
        tmp += step  

def linear_regression():
	old_stdout = sys.stdout
	log_file = open("lin_reg_results.log", "w")
	sys.stdout = log_file

	lin_reg = linear_model.LinearRegression()
	cv_score = cross_val_score(lin_reg, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
	lin_reg.fit(train_x, train_y)
	estimate_y = lin_reg.predict(test_x)
	print "Linear regression: "
	print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
	print '\t Coefficients: ', lin_reg.coef_
	print '\t Variance score: %.2f' % lin_reg.score(test_x, test_y)		# R_2 Coefficient of Determination: 1 is perfect prediction
	print'\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
	print ""

	ridge = linear_model.Ridge()
	cv_score = cross_val_score(ridge, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
	ridge.fit(train_x, train_y)
	estimate_y = ridge.predict(test_x)
	print "Ridge regression: "
	print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
	print '\t Coefficients: ', ridge.coef_
	print '\t Variance score: %.2f' % ridge.score(test_x, test_y)		# R_2 Coefficient of Determination: 1 is perfect prediction
	print'\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
	print ""

	lasso = linear_model.Lasso()
	cv_score = cross_val_score(lasso, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
	lasso.fit(train_x, train_y)
	estimate_y = lasso.predict(test_x)
	print "Lasso regression: "
	print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
	print '\t Coefficients: ', lasso.coef_
	print '\t Variance score: %.2f' % lasso.score(test_x, test_y)		# R_2 Coefficient of Determination: 1 is perfect prediction
	print'\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
	print ""

	elastic = linear_model.ElasticNet()
	cv_score = cross_val_score(elastic, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
	elastic.fit(train_x, train_y)
	estimate_y = elastic.predict(test_x)
	print "Elastic Net regression: "
	print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
	print '\t Coefficients: ', elastic.coef_
	print '\t Variance score: %.2f' % elastic.score(test_x, test_y)		# R_2 Coefficient of Determination: 1 is perfect prediction
	print'\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
	print ""

	sys.stdout = old_stdout
	log_file.close()

def decision_tree():
	old_stdout = sys.stdout
	log_file = open("dec_tree_results.log", "w")
	sys.stdout = log_file

	best = 0
	for max_depth in xrange(1, 20):
		dec_tree = tree.DecisionTreeRegressor(max_depth = max_depth)
		cv_score = cross_val_score(dec_tree, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
		dec_tree.fit(train_x, train_y)
		estimate_y = dec_tree.predict(test_x)
		score = dec_tree.score(test_x, test_y)
		if score > 0:
			print "Decision Tree regression: "
			print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
			print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
			print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
			print '\t Tree depth: ', dec_tree.tree_.max_depth
			print ""
			tree.export_graphviz(dec_tree, out_file='dec_tree_' + str(max_depth) + '.dot')

			if score > best:
				best = score

	print time() - start_time
	print 'Best: ', best

	sys.stdout = old_stdout
	log_file.close()

def ensemble_methods():
	start_time = time()

	old_stdout = sys.stdout
	log_file = open("adaboost_results.log", "w")
	sys.stdout = log_file

	best = 0
	for base_estimator in base_estimators:
		for n_estimators in xrange(1, 20):
			for loss in ['linear', 'square', 'exponential']:
				adaboost = ensemble.AdaBoostRegressor(base_estimator = base_estimator, n_estimators = n_estimators, loss = loss)
				cv_score = cross_val_score(adaboost, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
				adaboost.fit(train_x, train_y)
				estimate_y = adaboost.predict(test_x)
				score = adaboost.score(test_x, test_y)
				if score > 0:
					print "AdaBoost regression: "
					print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
					print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
					print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
					print '\t Number of estimators: ', n_estimators
					print '\t Loss function: ', loss
					print '\t Base estimator: ', base_estimator
					print ""

					if score > best:
						best = score

	print time() - start_time
	print 'Best: ', best

	print time() - start_time
	start_time = time()

	log_file.close()
	log_file = open("bagging_results.log", "w")
	sys.stdout = log_file

	best = 0
	for base_estimator in base_estimators:
		for n_estimators in xrange(1, 20):
			bagging = ensemble.BaggingRegressor(base_estimator = base_estimator, n_estimators = n_estimators)
			cv_score = cross_val_score(bagging, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
			bagging.fit(train_x, train_y)
			estimate_y = bagging.predict(test_x)
			score = bagging.score(test_x, test_y)
			if score > 0:
				print "Bagging regression: "
				print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
				print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
				print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
				print '\t Number of estimators: ', n_estimators
				print '\t Base estimator: ', base_estimator
				print ""

				if score > best:
					best = score

	print time() - start_time
	print 'Best: ', best

	print time() - start_time
	start_time = time()

	log_file.close()
	log_file = open("extra_trees_results.log", "w")
	sys.stdout = log_file

	best = 0
	for bootstrap in [True, False]:
		for max_depth in xrange(1, 20):
			for n_estimators in xrange(1, 20):
				extra_trees = ensemble.ExtraTreesRegressor(n_estimators = n_estimators, max_depth = max_depth, bootstrap = bootstrap)
				cv_score = cross_val_score(extra_trees, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
				extra_trees.fit(train_x, train_y)
				estimate_y = extra_trees.predict(test_x)
				score = extra_trees.score(test_x, test_y)
				if score > 0:
					print "Extra Trees regression: "
					print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
					print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
					print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
					print '\t Boostrap?: ', bootstrap
					print '\t Max depth: ', max_depth
					print '\t Number of estimators: ', n_estimators
					print ""

					if score > best:
						best = score

	print time() - start_time
	print 'Best: ', best

	print time() - start_time
	start_time = time()

	log_file.close()
	log_file = open("gradient_boost_results.log", "w")
	sys.stdout = log_file

	best = 0
	for loss in ['ls', 'lad', 'huber', 'quantile']:
		for max_depth in xrange(1, 20):
			for n_estimators in xrange(50, 200):
				gradient_boost = ensemble.GradientBoostingRegressor(n_estimators = n_estimators, max_depth = max_depth, loss = loss)
				cv_score = cross_val_score(gradient_boost, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
				gradient_boost.fit(train_x, train_y)
				estimate_y = gradient_boost.predict(test_x)
				score = gradient_boost.score(test_x, test_y)
				if score > 0:
					print "Gradient Boosting regression: "
					print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
					print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
					print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
					print '\t Loss function: ', loss
					print '\t Max depth: ', max_depth
					print '\t Number of estimators: ', n_estimators
					print ""

					if score > best:
						best = score

	print time() - start_time
	print 'Best: ', best

	print time() - start_time
	start_time = time()

	log_file.close()
	log_file = open("random_forest_results.log", "w")
	sys.stdout = log_file

	best = 0
	for bootstrap in [True, False]:
		for max_depth in xrange(1, 20):
			for n_estimators in xrange(1, 24):
				random_forest = ensemble.RandomForestRegressor(n_estimators = n_estimators, bootstrap = bootstrap, max_depth = max_depth)
				cv_score = cross_val_score(random_forest, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
				random_forest.fit(train_x, train_y)
				estimate_y = random_forest.predict(test_x)
				score = random_forest.score(test_x, test_y)
				if score > 0:
					print "Random Forest regression: "
					print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
					print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
					print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
					print '\t Bootstrap?: ', bootstrap
					print '\t Max depth: ', max_depth
					print '\t Number of estimators: ', n_estimators
					print ""

					if score > best:
						best = score

	print time() - start_time
	print 'Best: ', best

	print time() - start_time
	start_time = time()

	sys.stdout = old_stdout
	log_file.close()

def svm_methods():
	start_time = time()

	old_stdout = sys.stdout
	log_file = open("svr_results.log", "w")
	sys.stdout = log_file
	best = 0
	for kernel in ['linear', 'poly', 'rbf']:
		print kernel
		svr = svm.SVR(kernel = kernel)
		cv_score = cross_val_score(svr, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
		svr.fit(train_x, train_y)
		estimate_y = svr.predict(test_x)
		score = svr.score(test_x, test_y)
		if score > 0:
			print "Support Vector regression: "
			print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
			print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
			print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
			print '\t Kernel: ', kernel
			print ""

			if score > best:
				best = score

	print 'Best: ', best
	print time() - start_time
	start_time = time()

	log_file.close()
	log_file = open("linear_svr_results.log", "w")
	sys.stdout = log_file

	best = 0
	for loss in ['epsilon_insensitive', 'squared_epsilon_insensitive']:
		lin_svr = svm.LinearSVR(loss = loss)
		cv_score = cross_val_score(lin_svr, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
		lin_svr.fit(train_x, train_y)
		estimate_y = lin_svr.predict(test_x)
		score = lin_svr.score(test_x, test_y)
		if score > 0:
			print "Linear Support Vector regression: "
			print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
			print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
			print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
			print '\t Loss function: ', loss
			print ""

			if score > best:
				best = score

	print time() - start_time
	print 'Best: ', best

	print time() - start_time
	start_time = time()

	log_file.close()
	log_file = open("nu_svr_results.log", "w")
	sys.stdout = log_file

	nu_svr = svm.NuSVR()
	cv_score = cross_val_score(nu_svr, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
	nu_svr.fit(train_x, train_y)
	estimate_y = nu_svr.predict(test_x)
	score = nu_svr.score(test_x, test_y)
	if score > 0:
		print "Nu Support Vector regression: "
		print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
		print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
		print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
		print ""

	print time() - start_time

	sys.stdout = old_stdout
	log_file.close()

def kNN():
	start_time = time()

	old_stdout = sys.stdout
	log_file = open("knn_results.log", "w")
	sys.stdout = log_file

	best = 0
	for n_neighbors in xrange(2, 15):
		for algorithm in ['ball_tree', 'kd_tree', 'brute']:
			knn = neighbors.KNeighborsRegressor(n_neighbors = n_neighbors, algorithm = algorithm)
			cv_score = cross_val_score(knn, train_x, train_y, cv=5, scoring='neg_mean_squared_error')
			knn.fit(train_x, train_y)
			estimate_y = knn.predict(test_x)
			score = knn.score(test_x, test_y)
			if score > 0:
				print "K-Nearest Neighbors regression: "
				print '\t Cross Validation mean squared errors: ', ", ".join(str(0 - x) for x in cv_score)
				print '\t Variance score: %.2f' % score		# R_2 Coefficient of Determination: 1 is perfect prediction
				print '\t Mean squared error on test set: %.2f' % np.mean((estimate_y - test_y)**2)
				print '\t Number of neighbours: ', n_neighbors
				print '\t Algorithm used: ', algorithm
				print ""

				if score > best:
					best = score

	print time() - start_time
	print 'Best: ', best

	sys.stdout = old_stdout
	log_file.close()
