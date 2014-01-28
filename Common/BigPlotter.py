import matplotlib.pyplot as pyplot
import numpy as np
import sys

fitnessesES = np.genfromtxt("bestavgES.txt",delimiter = ',',dtype=float)
fitnessesALPS = np.genfromtxt("bestavgALPS.txt",delimiter = ',',dtype=float)
fitnessesIsland = np.genfromtxt("bestavgisland.txt",delimiter = ',',dtype=float)

#print fitnessesES.T
#print fitnessesALPS.T.shape


fitnessesALPS =  np.min(fitnessesALPS, axis = 0)
fitnessesIsland = np.min(fitnessesIsland, axis = 0)
print fitnessesALPS
fitnesses = np.vstack((fitnessesES.T, fitnessesALPS))
fitnesses = np.vstack((fitnesses, fitnessesIsland))

pyplot.figure()
pyplot.plot(fitnesses.T)
#pyplot.axis([1,150,0.6,1])
pyplot.xlabel("Generation")
pyplot.ylabel("Fitness")
#pyplot.yscale("log")
pyplot.legend(["Basic ES", "ALPS ES", "Island ES"])
#pyplot.xscale("log")
pyplot.title("Best Fitness vs. Gen")
pyplot.show()
