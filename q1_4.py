import numpy
import argparse
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('train',help='Training file')
parser.add_argument('test',help='Testing file')
args = parser.parse_args()

integerindexes = [3]

# Read the data, and return the attribute array, result array, and total data points.
def getdata(file,d):
	lines = []
	attrs = []
	res = []
	with open(file) as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		# We take all the elements and make them floats (from strings), or ints if they are not numeric (defined globally)
		data = [int(attr) if i in integerindexes else float(attr) for i,attr in enumerate(line.split())]
		attr = data[:-1]
		for i in range(d):
			attr.append(numpy.random.standard_normal())
		# Insert dummy variable of 1 to the first column.
		#data.insert(0,1)
		# All but one element is an attribute (feature)
		attrs.append(attr)
		# Last element is the result
		res.append([data[-1]])
	return numpy.matrix(attrs), numpy.array(res), len(lines)

# Use the numpy functions to do all matrix operations.
# No reason to code this from scratch...
def weightvector(attrs,res):
	# inv(transpose(X) * X) * transpose(X) * Y
	return (attrs.T * attrs).I * attrs.T * res

def printweights(weights):
	print('Learned weight vector (Rounded to 4 decimal places):\n{}'.format([round(w[0,0],4) for w in weights]))

# Add all squared errors and then divide them with the count.
def getSSE(weights, attrs, res, count):
	resguess = attrs * weights
	sumsqerr = 0.0
	for i in range(count):
		sumsqerr = pow(res[i,0] - resguess[i,0], 2)
	return sumsqerr/count

def plotlines(data1,data2,xaxis):
	plt.plot(xaxis,data1,'r--',xaxis,data2,'b--')
	plt.legend(['Training ASE','TestingASE'])
	plt.xlabel('# of random variables')
	plt.ylabel('Average Squared Error')
	plt.title('ASE of Training & Testing when adding d random variables')
	plt.show()

if __name__ == "__main__":
	drandom = range(1,10)
	trainase = []
	testase = []
	for j,d in enumerate(drandom):
		print('Inserting {} random variables...'.format(d))
		trainingattrs, trainingres, trainingcount = getdata(args.train,d)
		testingattrs, testingres, testingcount = getdata(args.test,d)

		w = weightvector(trainingattrs,trainingres)
		printweights(w)

		trainase.append(getSSE(w,trainingattrs,trainingres,trainingcount))
		print('AvgSqErr (ASE) of training: {}'.format(trainase))
		testase.append(getSSE(w,testingattrs,testingres,testingcount))
		print('AvgSqErr (ASE) of testing: {}'.format(testase))

	plotlines(trainase,testase,drandom)
