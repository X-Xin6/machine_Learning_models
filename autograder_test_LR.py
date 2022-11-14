import numpy as np
import pandas as pd
# import scipy
# import matplotlib.pyplot as plt
# import sklearn
# import sklearn.metrics as metrics
# from sklearn.utils import shuffle


def load_data(path, header):
    df = pd.read_csv(path, header=header, delimiter="\t")
    return df

def sigmoidal(z):
		z = z.astype(float)
		return (1/(1+np.exp(-z)))


def predict(w, xtest, normal = True):
		
	pred = sigmoidal(w.T.dot(xtest.T))

	for i in range(pred.shape[0]):
		for j in range(pred.shape[1]):

			if pred.iloc[i][j]<0.5:
				pred.iloc[i][j] = 0
			else:
				pred.iloc[i][j] = 1 
	
	
	return pred
		

def calc_accuracy(ytest, ypred):
		

	return 100*np.mean(ytest == ypred)


if __name__ == "__main__":
    # load the data from the file

	val = load_data("validation_dataset.tsv",None)

	xval = np.array(val.iloc[:,:-1])
	bias = np.ones(xval.shape[0]).T
	bias = bias.reshape(len(bias),1)
	xval = np.hstack((bias, xval))
	yval = np.array(val.iloc[:,-1])
	
	w = load_data("weights.tsv", None)

	ypred = predict(w, xval)
	ypred = np.array(ypred)
	yval = yval.reshape(1, len(yval))
	accuracy = calc_accuracy(yval, ypred)
	
	
	print("Accuracy of the logistic regression model is:", accuracy)
	

