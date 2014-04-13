import matplotlib.pyplot as pyplot
import numpy as np
import sys
import Tkinter, tkFileDialog 
import os.path 

if len(sys.argv) == 1:
    print "You can also give filename as a command line argument."
    print "Double-click on the directory of the run you want to analyze"
    root = Tkinter.Tk()
    root.withdraw()
    full_path = tkFileDialog.askdirectory(initialdir="../")
    print full_path
    runname = os.path.split(full_path)[1]
    rundir = os.path.split(full_path)[0]
    print runname
    #runname = raw_input("Enter Run Name: ")
    isMean = int(raw_input("Is it mean or best? 1 for mean and 0 for best: "))
else:
    runname = sys.argv[1]
    isMean = sys.argv[2]

numruns = 10

if isMean:
	filename = rundir +"/" + runname + "/" + "r0" + "/" + runname + "_meanfit.txt"
else:
	filename = rundir +"/"+ runname + "/" + "r0" + "/" + runname + "_bestfit.txt"
	
fitnesses = np.genfromtxt(filename,delimiter = ',',dtype=float)
print np.nanmin(fitnesses)
globalMinFitness = np.nanmin(fitnesses)
globalMinRun = 0
for i in range(1,numruns):
	if isMean:
		filename = rundir +"/" + runname + "/" + "r" + str(i) + "/" + runname + "_meanfit.txt"
	else:
		filename = rundir +"/"+ runname + "/" + "r" + str(i) + "/" + runname + "_bestfit.txt"
		
	new_fitnesses = np.genfromtxt(filename,delimiter = ',',dtype=float) 
	minfitness = np.nanmin(new_fitnesses)
	if minfitness < globalMinFitness:
		globalMinFitness = minfitness
		globalMinRun = i
	fitnesses += new_fitnesses
	print minfitness
	
print "The best run is " + str(globalMinRun)
fitnesses = fitnesses/numruns

if isMean:
	np.savetxt("meanavgES.txt", fitnesses, delimiter = ",")
else:
	np.savetxt("bestavgES.txt", fitnesses, delimiter = ",")
#print fitnesses
pyplot.figure()
totalPlot = pyplot.plot(fitnesses.T)
pyplot.xlim(0,len(fitnesses))
pyplot.xlabel("Generation")
pyplot.ylabel("Fitness")
numberPlots = len(totalPlot)
legends = []
for i in range (0, numberPlots):
	 legends.append('Level ' + str(i +1))
	
if len(legends) > 1:
	pyplot.legend(totalPlot, legends, prop={'size':12})

if isMean:
	pyplot.title("Mean Fitness vs. Gen")
else:
	pyplot.title("Best Fitness vs. Gen")
pyplot.show()

