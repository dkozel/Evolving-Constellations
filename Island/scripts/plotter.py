import matplotlib.pyplot as pyplot
import numpy as np
import sys

numislands = 4
if len(sys.argv) == 1:
    print "You can also give filename as a command line argument"
    filename = raw_input("Enter Filename: ")
else:
    filename = sys.argv[1]
    
fitnesses = np.genfromtxt(filename,delimiter = ',',dtype=float)
fitnesses = fitnesses[:,0:len(fitnesses)-1]
print len(fitnesses)
print fitnesses
fitnesses = np.reshape(fitnesses, (len(fitnesses)/numislands, numislands))
fitnesses = fitnesses.T
print fitnesses
pyplot.figure()
pyplot.plot(fitnesses.T)
pyplot.xlabel("Generation")
pyplot.ylabel("Fitness")
pyplot.title("Best Fitness vs. Gen")
pyplot.show()
