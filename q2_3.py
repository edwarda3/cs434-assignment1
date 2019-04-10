import sys
import numpy
import math
import mpmath
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train',help='Training file')
parser.add_argument('test',help='Testing file')
parser.add_argument('lambdas',help='A list of lambdas to test. input should be floats delineated by commas, like: 0.1,0.5,1,2 etc. Ensure that no spaces exist.')
args = parser.parse_args()

class LogRegression:
    def __init__(self,train,test,learn):
        print('Initializing model with lambda={}...'.format(learn),file=sys.stderr)
        (self.trainingattrs,self.trainingres) = train
        (self.testingattrs,self.testingres) = test
        print('{n} features, {trainnum} training points, {testnum} testing points'.format(n=self.trainingattrs.shape[1],trainnum=self.trainingattrs.shape[0],testnum=self.testingattrs.shape[0],file=sys.stderr))
        self.lambd = lambd
        self.lr = 0.1
        self.weights = numpy.array([float(1)]*self.trainingattrs.shape[1])

    def train(self):
        attrs = self.trainingattrs
        res = self.trainingres
        epsilon = 5000
        gradientnorm = epsilon+1
        print('Training until gradient < (epsilon = {})'.format(epsilon),file=sys.stderr)

        while(gradientnorm > epsilon):
            #Initialize an empty gradient
            gradient = numpy.array([0]*attrs.shape[1])
            for i in range(attrs.shape[0]): #For every statistic/instance
                instancedata = numpy.array(attrs[i,:])[0] #Retrieve a row (one data point)
                #Use algorithm from page 17 of slides. Batch learning for logistic regression
                try:
                    prediction = 1 / (1 + math.exp(-1 * numpy.dot(self.weights,instancedata)))
                except OverflowError:
                    # The overflow happens because guess is typically a number around ~-800, so when exp(-1*guess) occurs, we are essentially doing exp(800), and this causes an integer overflow
                    # We can use a large number math library (mpmath) to solve this
                    prediction = float(1 / (1 + mpmath.exp(-1 * numpy.dot(self.weights,instancedata))))
                # Our loss function, we add the loss vector to our gradient vector
                gradient = gradient + (prediction - res[i])*instancedata

            #Apply the loss to the weights by a factor of the learning rate 
            # w(t+1) = w(t) + lr * ( gradient of likelihood + l2penalty )
            l2penalty = numpy.array([math.log(-2*self.lambd*w) if w<0 else math.log(2*self.lambd*w) for w in self.weights])
            self.weights -= self.lr * (gradient + l2penalty)

            gradientnorm = numpy.linalg.norm(gradient)

    def getacc(self):
        #Get accuracy for training data
        traincorrect = 0
        for i in range(self.trainingattrs.shape[0]):
            instancedata = numpy.array(self.trainingattrs[i,:])[0]
            try:
                pred = 1 / (1 + math.exp(-1 * numpy.dot(self.weights,instancedata)))
                prediction = int(pred)
            except OverflowError:
                prediction = int(1 / (1 + mpmath.exp(-1 * numpy.dot(self.weights,instancedata))))
            if(prediction == self.trainingres[i]):
                traincorrect+=1
        trainacc = traincorrect/self.trainingres.shape[0]

        #get accuracy for testing data
        testcorrect = 0
        for i in range(self.testingattrs.shape[0]):
            instancedata = numpy.array(self.testingattrs[i,:])[0]
            try:
                pred = 1 / (1 + math.exp(-1 * numpy.dot(self.weights,instancedata)))
                prediction = int(pred)
            except OverflowError:
                prediction = int(1 / (1 + mpmath.exp(-1 * numpy.dot(self.weights,instancedata))))
            if(prediction == self.testingres[i]):
                testcorrect+=1
        testacc = testcorrect/self.testingres.shape[0]
        return trainacc,testacc

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

def plotlines(lambdas,training,testing):
    plt.plot(lambdas,[t*100 for t in training],'r--o',lambdas,[t*100 for t in testing],'b--^')
    plt.xscale('log')
    plt.title('Model accuracy over differing values of lambda when using L-2 Normalization')
    plt.xlabel('Lambda value for L-2 Normalization')
    plt.ylabel('Accuracy (Percentage correct)')
    plt.legend(['Training Acc','Testing Acc'])
    plt.show()

if __name__ == "__main__":
    trainingattrs,trainingres = getdata(args.train)
    testingattrs,testingres = getdata(args.test)
    lambdastrs = args.lambdas.split(',')
    lambdas = [float(i) for i in lambdastrs]

    trainacc = []
    testacc = []
    for lambd in lambdas:
        model = LogRegression((trainingattrs, trainingres), (testingattrs,testingres), lambd)
        model.train()
        trainaccuracy,testaccuracy = model.getacc()
        trainacc.append(trainaccuracy)
        testacc.append(testaccuracy)
        print("Model with lambda={}, train: {}%, test: {}%".format(lambd,trainaccuracy*100,testaccuracy*100))
        print()
    plotlines(lambdas,trainacc,testacc)
    

