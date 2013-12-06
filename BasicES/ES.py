from random import *
import numpy as np
import matplotlib.pyplot as pyplot
import top_block as gnuradio

numgen = int(raw_input("Please enter the number of generations: "))
arity = int(raw_input("Please enter the arity of the constellation: "))
arity_x2 = arity*2

mean = 0
stddev = 1

u = 10
numtimesu = 5
recomb = u*2

def randominit(x,y):
	return x+y

# minimize this function
def fitnessfunction(individual):
	#summation = 0
	#for x in range(0, len(individual)):
	#	summation += individual[x]
	#return summation
	#print(len(individual))
	
	constellation = np.zeros(arity_x2/2, dtype=complex)
	for x in range(0, len(individual),2):
		constellation[x/2] = complex(individual[x], individual[x+1])

	tb = gnuradio.top_block(constellation)

	tb.set_noise_level(0.4)
	tb.run()
	#ber = beraw.BERAWGNSimu(5)
	#ber.run()
	#bandwidth = tb.get_bandw()
	error_rate = np.mean(tb.blocks_vector_sink_x_1.data())
	#print error_rate
	#print x
	return error_rate

def mutate(individual, mean, stddev):
	newindiv = np.arange(arity_x2+1,dtype=float)
	for x in range(0, arity_x2):
		newindiv[x] = individual[x] + gauss(mean, stddev)
	# Tack on fitness to the end of the new matrix
	newindiv[arity_x2] = fitnessfunction(newindiv[0:arity_x2])
	return newindiv

def recombine(individuals):
	newindiv = np.zeros(arity_x2+1,dtype=float)
	for i in range(0, arity_x2):
		randparent1 = np.random.randint(0,u)
		randparent2 = np.random.randint(0,u)
		newindiv[i] = (individuals[randparent1,i]+individuals[randparent2,i])/2
	newindiv[arity_x2] = fitnessfunction(newindiv[0:arity_x2])
	return newindiv

def plotbest(sortedindiv):
	bestindiv = sortedindiv[0:1,0:arity_x2]
#print(bestindiv)
	imag_values = bestindiv[0,range(0,arity_x2,2)]
	real_values = bestindiv[0,range(1,arity_x2,2)]
	#print(real_values)
	#print(imag_values)
	h = pyplot.plot(imag_values,real_values,'bo')
	#pyplot.figure()	
	#pyplot.show()
	#h[0].set_data(imag_values,real_values)
	pyplot.show(block=False)
	#pyplot.draw()
			

indiv = np.random.normal(mean,stddev,(u,arity_x2))
meanfitnesses = []
bestfitnesses = []
for x in range(0, numgen):
	# Create initial matrix of new individuals
	newindiv = np.zeros((u*numtimesu+recomb,arity_x2+1), dtype=float)

	# Mutation
	for i in range(0,u):	
		# Mutate each individual numtimesu times
		#print(indiv[i,:])
		for numtimes in range(0,numtimesu):
			newindiv[numtimesu*i+numtimes,:] = mutate(indiv[i,:],mean,stddev)

		# print "This is the new matrix"
	
	# Recombination
	for i in range(0,recomb):
		newindiv[u*numtimesu+i,:] = recombine(indiv)
		

	# Sort the matrix based on the fitness
	newindiv=newindiv[newindiv[:,arity_x2].argsort()]
	#print(newindiv)
	#print(newindiv[0:u,0:arity_x2])

	# Set the individuals to the u best of the new individuals
	indiv = newindiv[0:u,0:arity_x2]
	fitnesses = newindiv[0:u,arity_x2]
	# print newindiv
	meanfitnesses.append(np.mean(fitnesses))
	#indiv = newindiv
	#print(indiv)
	bestfitnesses.append(newindiv[0,arity_x2])
	#plotbest(indiv)

#print(newindiv)
plotbest(indiv)
# pyplot.show()
pyplot.figure()
pyplot.plot(meanfitnesses)
print meanfitnesses
pyplot.figure()
pyplot.plot(bestfitnesses)
print bestfitnesses
pyplot.show()
	
