from random import *
import numpy as np
import matplotlib.pyplot as pyplot
import channel_model as gnuradio
import multiprocessing as multip
import ctypes
import os


############### Run Parameters ###########################
numgen = int(raw_input("Please enter the number of generations: "))
arity = int(raw_input("Please enter the arity of the constellation: "))
noise_level = float(raw_input("Please enter the desired noise level: "))

arity_x2 = arity*2

# Parameters for the random distribution used to create the initial population
mean = 0
stddev = 1

# Number of elites to keep
elites = 3

u = 20

# Number of times to do mutation on each individual
numtimesu = 5

# Number of recombinations to do
recomb = u*2

# Age Levels for ALPS
agelevels = np.array([4,8,16,32,64,numgen], dtype=int)

# Use this line to run a standard ES
# agelevels = np.array([numgen], dtype=int)

############################################################

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

def mutateStddev(sigma, age):
	learningRate = 1.0/np.sqrt(age+1)
	newstddev = sigma*np.exp(gauss(mean, learningRate))
	return newstddev

def mutate(individual, mean, stddev):
	newindiv = np.arange(arity_x2,dtype=float)
	for x in range(0, arity_x2):
		newindiv[x] = individual[x] + gauss(mean, stddev)
	
	return newindiv

# Recombines the individuals given, and returns the new individual and the age of its oldest parent
def recombine(level,numindiv):
	newindiv = np.zeros(arity_x2,dtype=float)
	individuals = level[0]
	ages = level[2]
	max_age = 0
	for i in range(0, arity_x2):
		randparent1 = np.random.randint(0,numindiv)
		randparent2 = np.random.randint(0,numindiv)
		oldest_parent_age = np.maximum(ages[randparent1],ages[randparent2])
		if oldest_parent_age > max_age:
			max_age = oldest_parent_age
		newindiv[i] = (individuals[randparent1,i]+individuals[randparent2,i])/2
	newindiv_and_age = np.empty((2),dtype='object')
	newindiv_and_age[0] = newindiv	
	newindiv_and_age[1] = max_age
	
	return newindiv_and_age

def recombineStddev(stddevs,numindiv):
	randparent1 = np.random.randint(0,numindiv)
	randparent2 = np.random.randint(0,numindiv)
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

	pyplot.show(block=False)
			
def run(level,level_num,u,numtimesu,recomb,arity_x2):
	indiv = level[0]
	stddevs = level[1]
	age = level[2]
	fitnesses = level[3]


	# Create initial matrix of new individuals, another matrix for their mutation rates, and their fitnesses.
	# The reasoning behind this is to make the code expandable to individuals with varying chromosome lengths.
	numindiv = len(indiv)
	numnewindiv = numindiv*numtimesu+recomb+np.minimum(elites,indiv.shape[0])
	newindiv = np.zeros((numindiv*numtimesu+recomb+elites,arity_x2), dtype=float)
	newstddevs = np.zeros((numnewindiv,1), dtype=float)
	newfitness = np.zeros((numnewindiv,1), dtype=float)
	newages = np.zeros((numnewindiv,1), dtype=float)
	
	
	# Mutation
	for i in range(0,numindiv):	
		# Mutate each individual numtimesu times
		for numtimes in range(0,numtimesu):
			# Mutate the mutation rate for the individual
			newstddevs[numtimesu*i+numtimes,0] = mutateStddev(stddevs[i],age[i])
			# Create a new individual via mutation, and store the stddev
			newindiv[numtimesu*i+numtimes,:] = mutate(indiv[i,:],mean,newstddevs[numtimesu*i+numtimes,0])
		
			# Store the fitness value for this new individual
			newfitness[numtimesu*i+numtimes,0] = fitnessfunction(newindiv[numtimesu*i+numtimes,:])	
			
			# Store the new age for the individual
			newages[numtimesu*i+numtimes,0] = age[i] 

	# Recombination
	for i in range(0,recomb):
		child_and_age = recombine(level,numindiv)
		newindiv[numindiv*numtimesu+i,:] = child_and_age[0]
		newstddevs[numindiv*numtimesu+i,:] = recombineStddev(stddevs,numindiv)
		newfitness[numindiv*numtimesu+i,:] = fitnessfunction(newindiv[numindiv*numtimesu+i,:])		
		newages[numindiv*numtimesu+i,:] = child_and_age[1] 

	# Save the elites from the previous level
	for i in range(0,np.minimum(elites,indiv.shape[0])):
		#print np.minimum(elites,indiv.shape[0])
		newindiv[numindiv*numtimesu+recomb+i,:] = indiv[i,:]
		newstddevs[numindiv*numtimesu+recomb+i,:] = stddevs[i]
		newfitness[numindiv*numtimesu+recomb+i,:] = fitnessfunction(indiv[i,:])	
		newages[numindiv*numtimesu+recomb+i,:] = age[i]

	# Sort the matrix and the standard deviation vector based on the fitness
	newindiv=newindiv[newfitness[:,0].argsort()]
	newstddevs=newstddevs[newfitness[:,0].argsort()]
	newages=newages[newfitness[:,0].argsort()] 
	newfitness=newfitness[newfitness[:,0].argsort()]
	#print(newindiv)
	#print(newindiv[0:u,0:arity_x2])

	# Set the population to the u best of the new individuals
	indiv = newindiv[0:u,0:arity_x2]
	stddevs = newstddevs[0:u,0]
	ages = newages[0:u,0]
	fitness = newfitness[0:u,0]
	# fitnesses = newfitness
	# print newindiv
	# meanfitnesses[l].append(np.mean(fitnesses))
	#indiv = newindiv
	#print(indiv)
	#bestfitnesses.append(fitnesses[0,0])
	#plotbest(indiv)
	
	newlevel=np.empty((4),dtype='object')
	newlevel[0] = indiv
	newlevel[1] = stddevs
	
	# Age all the individuals by 1
	newlevel[2] = ages + 1
	newlevel[3] = fitness

	#print age
	return newlevel

# Move individuals up a level if they are too old
def sortLevels(levels):
	for l in range(0,len(levels)-1):
		ages = levels[l][2]
		
		# Get the indexes of the old guys
		oldguys = np.where(ages > agelevels[l])[0]
		#print oldguys
		#print len(oldguys)
		
		# Copy old guys into next level
		for i in range(0,len(oldguys)):
			#print levels[l][0]
			#oldguycopy = np.empty((3),dtype='object')
			oldguycopy = np.array([levels[l][0][i,:]])
			if levels[l+1][0] == None:
				levels[l+1][0] = oldguycopy
				levels[l+1][1] = levels[l][1][i]
				levels[l+1][2] = levels[l][2][i]
				levels[l+1][3] = levels[l][3][i]
			else:
				levels[l+1][0] = np.append(levels[l+1][0],oldguycopy,axis=0)
				levels[l+1][1] = np.append(levels[l+1][1],levels[l][1][i])
				levels[l+1][2] = np.append(levels[l+1][2],levels[l][2][i])
				levels[l+1][3] = np.append(levels[l+1][3],levels[l][3][i])
		
		#print oldguys
		
		# Remove old guys from the level
		if len(oldguys) > 0:
			levels[l][0]=np.delete(levels[l][0],oldguys,axis=0)
			levels[l][1]=np.delete(levels[l][1],oldguys,axis=0)	
			levels[l][2]=np.delete(levels[l][2],oldguys,axis=0)
			levels[l][3]=np.delete(levels[l][3],oldguys,axis=0)	
		#print levels
	
	return levels

def run_process(level,level_num,u,numtimesu,recomb,arity_x2,new_level):
	new_level[level_num] = run(level,level_num,u,numtimesu,recomb,arity_x2)
	print new_level[level_num]
	return

class RunProcess(multip.Process):
	def __init__(self,level,level_num,u,numtimesu,recomb,arity_x2,queue):
		multip.Process.__init__(self)
		self.level = level
		self.level_num = level_num
		self.u = u
		self.numtimesu = numtimesu
		self.recomb = recomb
		self.arity_x2 = arity_x2
		self.newlevel = np.empty((4),dtype='object')
		self.queue = queue
	
	def run(self):
		self.newlevel = run(self.level,self.level_num,self.u,self.numtimesu,self.recomb,self.arity_x2)
		self.queue.put([self.level_num,self.newlevel])
		#print self.newlevel
	def getLevelNum(self):
		return self.level_num
	
            
def debugPrint(levels):
	for l in range(0,len(levels)):
		print "Level number"
		print l
		print "Individuals"
		print levels[l][1]
		print "Ages"			
		print levels[l][2]
		print "Fitnesses"
		print levels[l][3]

def saveIndiv(indivs, level):
	if not os.path.exists(rundir):
		os.makedirs(rundir)
		
	with open(rundir + runname + '_level'+ str(level) + '_allindivs.txt','a+') as f_handle:
		np.savetxt(f_handle,indivs, delimiter=', ')

def writeToFile(levels,meanfitnesses,bestfitnesses,seed):
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
	fo.write("Age Levels =" + str(agelevels) + "\n")
	fo.write(str(levels) + "\n")
	fo.write("sSeed = " + str(seed) + "\n")
	
	np.savetxt(rundir + runname + "_meanfit.txt", meanfitnesses, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ')
	np.savetxt(rundir + runname + "_bestfit.txt", bestfitnesses, fmt='%.18e', delimiter=',', newline='\n', header='', footer='', comments='# ')
	
def randomUniformPolar(radius, outputsize):
	mags = np.random.uniform(0,radius, outputsize)
	deg = np.random.uniform(0,2*np.pi,outputsize)
	outputmatrix = mags*np.sin(deg)
	return outputmatrix	

if __name__ == '__main__':
	
	runname = "ALPS_arity" +str(arity) + "_" + str(numgen) +"g" + str(raw_input("Please enter anything else you would like in the name: "))
	numruns = int(raw_input("Please enter the number of runs: "))
	replace_freq = int(raw_input("How many generations should the bottom be replaced? Min = " + str(agelevels[0]) + ": "))
	
	for numrun in range(0, numruns):
		rundir = runname + '/r' +str(numrun) + '/'
		seed = numrun
		np.random.seed(seed)
		#indiv = 2*np.random.random_sample((u,arity_x2))
		indiv = randomUniformPolar(1, (u,arity_x2))
		stddevs = np.random.normal(mean,stddev,u)
		ages = 1*np.ones(u)
		meanfitnesses = np.empty((len(agelevels),numgen),dtype='object')
		meanfitnesses[:] = np.NAN
		bestfitnesses = np.empty((len(agelevels),numgen),dtype='object')
		bestfitnesses[:] = np.NAN
		
		# Initialize vector of levels
		levels = np.empty((len(agelevels)),dtype='object')

		for l in range(0,len(agelevels)):
			levels[l] = np.empty((4),dtype='object')
			
		levels[0][0] = indiv
		levels[0][1] = stddevs
		levels[0][2] = ages
		for gen in range(0,numgen):
			process_array = [None]*len(levels)
			new_levels = multip.Array(ctypes.py_object,len(levels))
			queue = multip.Queue()
			
			# Create process for each level
			for l in range(0,len(levels)):
				if levels[l][0] != None and len(levels[l][0]) != 0:
					process = RunProcess(levels[l],l,u,numtimesu,recomb,arity_x2,queue)
					process_array[l] = process 
					#print process_array
					process.start()
			
			# Stop all the processes
			for l in range(0,len(levels)):
				#print levels[l][0]
				if process_array[l] != None:
					#print "Stopping thread"
					process_array[l].join()
			
			# Empty Multiprocessing communication queue
			while queue.empty()==False:
				msg = queue.get()
				levels[msg[0]] = msg[1] 
					
			levels = sortLevels(levels)
			#debugPrint(levels)
			
			# Save mean and best fitness value for each level
			for l in range(0,len(agelevels)):
				if levels[l][3] != None and len(levels[l][3]) >0:
					meanfitnesses[l,gen] = np.mean(levels[l][3])
					bestfitnesses[l,gen] = levels[l][3][0]
					saveIndiv(levels[l][0],l)
			
			if gen%replace_freq == 0 and len(agelevels) > 1:
				indiv = randomUniformPolar(1,(u,arity_x2))
				stddevs = np.random.normal(mean,stddev,u)
				ages = 1*np.ones(u)
				levels[0][0] = indiv
				levels[0][1] = stddevs
				levels[0][2] = ages

		writeToFile(levels,meanfitnesses,bestfitnesses,seed)
		print meanfitnesses
	

	#plotbest(levels[len(levels)-2][0])
	
