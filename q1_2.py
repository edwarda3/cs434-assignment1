import numpy
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('train',help='Training file')
parser.add_argument('test',help='Testing file')
args = parser.parse_args()

integerindexes = [3]

# Read the data, and return the attribute array, result array, and total data points.
def getdata(file):
	lines = []
	attrs = []
	res = []
	with open(file) as f:
		lines = f.readlines()
	for line in lines:
		line = line.strip()
		# We take all the elements and make them floats (from strings), or ints if they are not numeric (defined globally)
		data = [int(attr) if i in integerindexes else float(attr) for i,attr in enumerate(line.split())]
		# Insert dummy variable of 1 to the first column.
		data.insert(0,1)
		# All but one element is an attribute (feature)
		attrs.append(data[:-1]) 
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

if __name__ == "__main__":
	trainingattrs, trainingres, trainingcount = getdata(args.train)
	testingattrs, testingres, testingcount = getdata(args.test)
	
	w = weightvector(trainingattrs,trainingres)
	printweights(w)

	trainase = getSSE(w,trainingattrs,trainingres,trainingcount)
	print('AvgSqErr (ASE) of training: {}'.format(trainase))
	testase = getSSE(w,testingattrs,testingres,testingcount)
	print('AvgSqErr (ASE) of testing: {}'.format(testase))
