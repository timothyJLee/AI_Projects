#!/usr/bin/python3
import pickle

#open pickle file
pickleFile = open( "testData.pickle", "rb" )

#using a for x in range() loop we could do the same right?
testData1 = pickle.load(pickleFile) #testData1 will never converege :(
testData2 = pickle.load(pickleFile)
testData3 = pickle.load(pickleFile) #testData3 will never converege :(
testData4 = pickle.load(pickleFile)


# a fancy print function 
def fancyPrint(threshold,learning_rate,weights,testData):
	print("#" * 96)
	
	#fancy printf
	print("\nstarting to teach on testData%d with threshold:%3.4f, learning_rate:%3.4f and weights:%s\n" 
		% (testData,threshold,learning_rate,str(weights)))

	print("#" * 96)


#Stolen wikipidea code with comments
def dot_product(values, weights):
    return sum(value * weight for value, weight in zip(values, weights))

# example training_set [((0,0,0),0),((0,0,1),1),((0,1,0),0),((0,1,1),1),((1,0,0),0),((1,0,1),1),((1,1,0),0),((1,1,1),1)]
def teach(threshold,learning_rate,weights,training_set):
	
	while True:

		##Print a dotted line
		print('-' * 60)

		error_count = 0

		#for (1,0,0) = input_vector 1 = desired_output
		for input_vector, desired_output in training_set:

			print(weights)

			#false = 0, true = 1
			result = dot_product(input_vector, weights) > threshold

			#Since our desired output is the second element in the
			#tuple it will be something like 1 - result
			error = desired_output - result

			if error != 0:
				error_count += 1

				#enumerate is saying give us a index and a element from input_vector 
				for index, value in enumerate(input_vector):
					weights[index] += learning_rate * error * value
		if error_count == 0:
			break



#First run
threshold = .001
learning_rate = .1
weights = [0,2,0]

#fancy print function
fancyPrint(threshold,learning_rate,weights,2)

#actual call to teach method
teach(threshold,learning_rate,weights,testData2)


print("")

#Second run
#playing with weights
threshold = .001
learning_rate = .1
weights = [9,9,9]

fancyPrint(threshold,learning_rate,weights,4)


teach(threshold,learning_rate,weights,testData4)

print("")

#Third run
#more playing
threshold = .9
learning_rate = .5
weights = [0,6,9]

fancyPrint(threshold,learning_rate,weights,2)


teach(threshold,learning_rate,weights,testData2)

print("")

#Fourth run
threshold = .001
learning_rate = 5
weights = [100,0,2]

fancyPrint(threshold,learning_rate,weights,4)

teach(threshold,learning_rate,weights,testData4)



