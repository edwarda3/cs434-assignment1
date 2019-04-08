import numpy
import math
import mpmath
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train',help='Training file')
parser.add_argument('test',help='Testing file')
parser.add_argument('learningrate',help='Learning Rate')
args = parser.parse_args()

class LogRegression:
	def __init__(self,train,test,learn):
		(self.trainingattrs,self.trainingres) = train
		(self.testingattrs,self.testingres) = test
		self.lr = learn
		self.weights = numpy.array([float(0)]*self.trainingattrs.shape[1])
		self.trainacc = []
		self.testacc = []

	def train(self):
		attrs = self.trainingattrs
		res = self.trainingres
		epsilon = 10000
		gradientnorm = epsilon+1

		while(gradientnorm > epsilon):
			gradient = numpy.array([0]*attrs.shape[1])
			for i in range(attrs.shape[0]): #For every statistic/instance
				instancedata = numpy.array(attrs[i,:])[0]
				#Use algorithm from page 17 of slides. Batch learning for logistic regression
				try:
					prediction = 1 / (1 + math.exp(-1 * numpy.dot(self.weights,instancedata)))
				except OverflowError:
					# The overflow happens because guess is typically a number around ~-800, so when exp(-1*guess) occurs, we are essentially doing exp(800), and this causes an integer overflow
					# We can use a large number math library (mpmath) to solve this
					prediction = float(1 / (1 + mpmath.exp(-1 * numpy.dot(self.weights,instancedata))))
				gradient = gradient + (prediction - res[i])*instancedata
			self.weights -= self.lr * gradient
			gradientnorm = numpy.linalg.norm(gradient)

			self.getacc()

	def getacc(self):
		#Get accuracy for training data
		traincorrect = 0
		for i in range(self.trainingattrs.shape[0]):
			instancedata = numpy.array(self.trainingattrs[i,:])[0]
			try:
				prediction = int(1 / (1 + math.exp(-1 * numpy.dot(self.weights,instancedata))))
			except OverflowError:
				prediction = int(1 / (1 + mpmath.exp(-1 * numpy.dot(self.weights,instancedata))))
			if(prediction == self.trainingres[i]):
				traincorrect+=1
		self.trainacc.append(traincorrect/self.trainingres.shape[0])

		#get accuracy for testing data
		testcorrect = 0
		for i in range(self.testingattrs.shape[0]):
			instancedata = numpy.array(self.testingattrs[i,:])[0]
			try:
				prediction = int(1 / (1 + math.exp(-1 * numpy.dot(self.weights,instancedata))))
			except OverflowError:
				prediction = int(1 / (1 + mpmath.exp(-1 * numpy.dot(self.weights,instancedata))))
			if(prediction == self.testingres[i]):
				testcorrect+=1
		self.testacc.append(testcorrect/self.testingres.shape[0])

def getdata(file):
	lines = None
	attrs = []
	res = []
	with open(file, 'r') as f:
		lines = f.readlines()
	for line in lines:
		data = [int(x) for x in line.strip().split(',')]
		attrs.append(data[:-1])
		res.append(data[-1])
	return numpy.matrix(attrs),numpy.array(res)

def plotlines(training,testing):
	xaxis = list(range(1,len(training)+1))
	plt.plot(xaxis,[t*100 for t in training],'r--o',xaxis,[t*100 for t in testing],'b--^')
	plt.title('Model accuracy over iterations of gradient descent')
	plt.xlabel('Number of iterations')
	plt.ylabel('Accuracy (Percentage correct)')
	plt.show()

if __name__ == "__main__":
	trainingattrs,trainingres = getdata(args.train)
	testingattrs,testingres = getdata(args.test)
	lr = float(args.learningrate)

	model = LogRegression((trainingattrs, trainingres), (testingattrs,testingres), lr)
	model.train()
	#print(model.weights)
	plotlines(model.trainacc,model.testacc)
	

