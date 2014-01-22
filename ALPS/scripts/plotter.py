import matplotlib.pyplot as pyplot
import numpy as np
import sys

if len(sys.argv) == 1:
    print "You can also give filename as a command line argument"
    filename = raw_input("Enter Filename: ")
else:
    filename = sys.argv[1]
    
fitnesses = np.genfromtxt(filename,delimiter = ',',dtype=float)
print fitnesses
pyplot.figure()
pyplot.plot(fitnesses.T)
pyplot.xlabel("Generation")
pyplot.ylabel("Fitness")
pyplot.title("Best Fitness vs. Gen")
pyplot.show()
