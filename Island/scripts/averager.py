import matplotlib.pyplot as pyplot
import numpy as np
import sys
import Tkinter, tkFileDialog

numislands = 5
if len(sys.argv) == 1:
	print "You can also give filename as a command line argument"
	root = Tkinter.Tk()
	root.withdraw()
	runname = tkFileDialog.askdirectory()
	print runname
	#runname = raw_input("Enter Run Name: ")
	isMean = int(raw_input("Is it mean or best? 1 for mean and 0 for best: "))
	#root = Tkinter.Tk()

    
else:
    runname = sys.argv[1]
    isMean = sys.argv[2]

numruns = 10

if isMean:
	filename = runname + "/" + "r0" + "/" + "1" + "_mean.txt"
else:
	filename = runname + "/" + "r0" + "/" + "1" + "_bestfit.txt"
	
fitnesses = np.genfromtxt(filename,delimiter = ',',dtype=float)
fitnesses = fitnesses[0:(len(fitnesses)-1)]
print np.nanmin(fitnesses)
for i in range(1,numruns):
	if isMean:
		filename = runname + "/" + "r" + str(i) + "/1" + "_mean.txt"
	else:
		filename = runname + "/" + "r" + str(i) + "/1" + "_bestfit.txt"
		
	fitnesses2 = np.genfromtxt(filename,delimiter = ',',dtype=float)
	fitnesses2 = fitnesses2[0:(len(fitnesses2)-1)]
	print np.nanmin(fitnesses2)
	fitnesses += fitnesses2
	
fitnesses = fitnesses/numruns
print len(fitnesses)
print fitnesses
fitnesses = np.reshape(fitnesses, (len(fitnesses)/numislands, numislands))
fitnesses = fitnesses.T

if isMean:
	np.savetxt("meanavgisland.txt", fitnesses, delimiter = ",")
else:
	np.savetxt("bestavgisland.txt", fitnesses, delimiter = ",")
pyplot.figure()
pyplot.plot(fitnesses.T)
pyplot.xlim(0,len(fitnesses[0]))
pyplot.xlabel("Generation")
pyplot.ylabel("Fitness")

if isMean:
	pyplot.title("Mean Fitness vs. Gen")
else:
	pyplot.title("Best Fitness vs. Gen")
pyplot.show()

