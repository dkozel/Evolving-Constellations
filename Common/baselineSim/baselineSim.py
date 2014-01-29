from random import *
import numpy as np
import matplotlib.pyplot as pyplot
import channel_model as gnuradio
#import channel_model_nomulti as gnuradio
import os

#numgen = int(raw_input("Please enter the number of generations: "))
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
	

	tb = gnuradio.channel_model(individual)

	tb.set_noise_level(noise_level)
	tb.run()
	#ber = beraw.BERAWGNSimu(5)
	#ber.run()tb.blocks_vector_sink_x_1.data()
	#bandwidth = tb.get_bandw()
	error_rate = np.mean(tb.blocks_vector_sink_x_1.data())
	#print error_rate
	#print x
	return error_rate

def convertToReals(individual):
	newindiv = np.arange(arity_x2,dtype=float)
	for i in range(0,arity):
		newindiv[2*i] = individual[i].real
		newindiv[2*i +1] = individual[i].imag
	return newindiv

def plotindiv(indiv):
#print(bestindiv)
	imag_values = indiv[range(1,arity_x2,2)]
	real_values = indiv[range(0,arity_x2,2)]
	normfactor = 0
	for i in range(0,len(imag_values)):
		normfactor += np.linalg.norm([imag_values[i],real_values[i]])
	normfactor = normfactor/len(imag_values)
	imag_values = imag_values/normfactor
	real_values = real_values/normfactor
		#print np.linalg.norm([imag_values[i,1],real_values[i,1]])
		#print np.linalg.norm([0.707,0.707])
	print imag_values
	
	xb = real_values
	yb = imag_values
	fig = pyplot.figure()
	if len(xb) >= 8:
		ax = pyplot.axes(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5))
	else:
		ax = pyplot.axes(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5))
	ax.plot(xb,yb, 'bo')
	pyplot.show()
		

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
	individual = np.arange(arity_x2,dtype=float)
	individual[0] = 0.707
	individual[1] = 0.707
	individual[2] = -0.707
	individual[3] = 0.707
	individual[4] = 0.707
	individual[5] = -0.707
	individual[6] = -0.707
	individual[7] = -0.707
	#print fitnessfunction(individual)
	if arity == 4:
		individual =  np.asarray(gnuradio.digital.constellation_qpsk().points())
	elif arity == 8:
		individual =  np.asarray(gnuradio.digital.constellation_8psk().points())
	elif arity == 16:
		individual =  np.asarray(gnuradio.digital.qam.qam_constellation().points())
	else:
		individual =  np.asarray(gnuradio.digital.qam.qam_constellation(arity).points())
	fitness = 0
	
	for i in range(0,10):
		fitness_new = fitnessfunction(individual)
		print fitness_new
		fitness += fitness_new
	fitness = fitness/10
	print individual
	print fitness
	#print(newindiv)
	plotindiv(convertToReals(individual))
	# pyplot.show()
	
