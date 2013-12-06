from random import *
import numpy as np
import matplotlib.pyplot as pyplot
import channel_model as gnuradio
import os

numgen = int(raw_input("Please enter the number of generations: "))
arity = int(raw_input("Please enter the arity of the constellation: "))
noise_level = float(raw_input("Please enter the desired noise level: "))

arity_x2 = arity*2
#runname = "ES_" + str(raw_input("Please enter the name of run: "))
#rundir = runname + '/'

# Parameters for the random distribution used to create the initial population
mean = 0
stddev = 1

u = 20

# Number of times to do mutation on each individual
numtimesu = 5

# Number of recombinations to do
recomb = u*2

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

	tb = gnuradio.channel_model(constellation)

	tb.set_noise_level(noise_level)
	tb.run()
	#ber = beraw.BERAWGNSimu(5)
	#ber.run()
	#bandwidth = tb.get_bandw()
	error_rate = np.mean(tb.blocks_vector_sink_x_1.data())
	#print error_rate
	#print x
	return error_rate

def mutateStddev(sigma, gen):
	learningRate = 1.0/np.sqrt(gen+1)
	newstddev = sigma*np.exp(gauss(mean, learningRate))
	return newstddev

def mutate(individual, mean, stddev):
	newindiv = np.arange(arity_x2,dtype=float)
	for x in range(0, arity_x2):
		newindiv[x] = individual[x] + gauss(mean, stddev)
	
	return newindiv

def recombine(individuals):
	newindiv = np.zeros(arity_x2,dtype=float)
	for i in range(0, arity_x2):
		randparent1 = np.random.randint(0,u)
		randparent2 = np.random.randint(0,u)
		newindiv[i] = (individuals[randparent1,i]+individuals[randparent2,i])/2
	
	return newindiv

def recombineStddev(stddevs):
	randparent1 = np.random.randint(0,u)
	randparent2 = np.random.randint(0,u)
	stddev = (stddevs[randparent1]+stddevs[randparent2])/2
	
	return stddev

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

def writeToFile(indiv,meanfitnesses,bestfitnesses,bestindivs,seed):
	if not os.path.exists(rundir):
		os.makedirs(rundir)
	fo = open(rundir+ runname + "_info.txt", "w+")
	
	fo.write("Number of Generations = " + str(numgen) + "\n")
	fo.write("Arity = " + str(arity) + "\n")
	fo.write("\nu = ")
	fo.write(str(u))
	fo.write("\nNumTimesU = ")
	fo.write(str(numtimesu))
	fo.write("\nrecomb = ")
	fo.write(str(recomb) + "\n")
	fo.write(str(indiv))
	fo.write("\nseed = " + str(seed) + "\n")
	
	np.savetxt(rundir + runname + "_meanfit.txt", meanfitnesses, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ')
	np.savetxt(rundir + runname + "_bestfit.txt", bestfitnesses, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ')
	np.savetxt(rundir + runname + "_bestindivs.txt", bestindivs, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ')
	
def saveIndiv(indivs):
	if not os.path.exists(rundir):
		os.makedirs(rundir)
		
	with open(rundir + runname + '_allindivs.txt','a+') as f_handle:
		np.savetxt(f_handle,indivs, delimiter=', ')
	
# Generates a uniform random distribution within the unit circle. 
def randomUniformPolar(radius, outputsize):
	mags = np.random.uniform(0,radius, outputsize)
	deg = np.random.uniform(0,2*np.pi,outputsize)
	outputmatrix = mags*np.sin(deg)
	return outputmatrix
    
if __name__ == '__main__':
	
	runname = "ES_arity" +str(arity) + "_" + str(numgen) +"g" + str(raw_input("Please enter anything else you would like in the name: "))
	numruns = int(raw_input("Please enter the number of runs: "))
	
	for numrun in range(0, numruns):
		seed = numrun
		rundir = runname + '/r' +str(numrun) + '/'
		np.random.seed(seed)
		
		#indiv = np.random.normal(mean,stddev,(u,arity_x2))
		indiv = randomUniformPolar(1 , (u,arity_x2))
		print indiv
		stddevs = np.random.normal(stddev,stddev,u)
		
		# Arrays for recording mean, best, and best individual vs. generation
		meanfitnesses = np.empty(numgen,dtype='float')
		bestfitnesses = np.empty(numgen,dtype='float')
		bestindivs = np.empty((numgen,arity_x2),dtype='float')
		
		for x in range(0, numgen):
			# Create initial matrix of new individuals, another matrix for their mutation rates, and their fitnesses.
			# The reasoning behind this is to make the code expandable to individuals with varying chromosome lengths.
			newindiv = np.zeros((u*numtimesu+recomb,arity_x2), dtype=float)
			newstddevs = np.zeros((u*numtimesu+recomb,1), dtype=float)
			newfitness = np.zeros((u*numtimesu+recomb,1), dtype=float)

			# Mutation
			for i in range(0,u):	
				# Mutate each individual numtimesu times
				for numtimes in range(0,numtimesu):
					# Mutate the mutation rate for the individual
					newstddevs[numtimesu*i+numtimes,0] = mutateStddev(stddevs[i],x)
					# Create a new individual via mutation, and store the stddev
					newindiv[numtimesu*i+numtimes,:] = mutate(indiv[i,:],mean,newstddevs[numtimesu*i+numtimes,0])
				
					# Store the fitness value for this new individual
					newfitness[numtimesu*i+numtimes,0] = fitnessfunction(newindiv[numtimesu*i+numtimes,:])	
			

			# Recombination
			for i in range(0,recomb):
				newindiv[u*numtimesu+i,:] = recombine(indiv)
				newstddevs[u*numtimesu+i,:] = recombineStddev(stddevs)
				newfitness[u*numtimesu+i,:] = fitnessfunction(newindiv[u*numtimesu+i,:])		
				

			# Sort the matrix and the standard deviation vector based on the fitness
			newindiv=newindiv[newfitness[:,0].argsort()]
			newstddevs=newstddevs[newfitness[:,0].argsort()]

			#print(newindiv)
			#print(newindiv[0:u,0:arity_x2])

			# Set the population to the u best of the new individuals
			indiv = newindiv[0:u,0:arity_x2]
			stddevs = newstddevs[0:u,0]

			fitnesses = newfitness

			meanfitnesses[x]= np.mean(fitnesses)
			bestfitnesses[x] = fitnesses[0,0]
			bestindivs[x] = indiv[0:1,0:arity_x2]
			saveIndiv(indiv)

		# Record the data to file
		writeToFile(newindiv,meanfitnesses,bestfitnesses,bestindivs,seed)
		
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
	
