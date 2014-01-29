from random import *
import numpy as np
import matplotlib.pyplot as pyplot
import channel_model as gnuradio
import os

class IslandEvolution(object):
    def __init__(self, islandCount, logPath):
      self.islandCount = islandCount
      self.runCount = 0
      self.runArity = 0
      self.runGenMax = 0
      self.runPopSize = 0
      self.runElites = 0
      self.epochNum = 0
      self.logPath = logPath
      self.logFile = ""
      self.bestFile = ""
      self.bestFitFile = ""
      self.meanFile = ""
      self.islandFiles = []
      self.migrantCount = 0

    # Computes a single run with a given population size and an epoch of a given number of generations
    def evolve(self, popSize, epochNum, genMax, arity, migrantCount, elites):
      self.runCount += 1
      self.runArity = arity
      self.runGenMax = genMax
      self.runPopSize = popSize
      self.runEpochNum = epochNum
      self.migrantCount = migrantCount
      self.runElites = elites
	
      self.logInfo()

      islands = [self.stdPopInit(popSize, 0, 1, arity) for _ in range(0, self.islandCount)]

      for gen in range(1, genMax+1):
        print "Run %d Generation %d" % (self.runCount, gen)

        if gen%epochNum == 0:
          islands = self.stdMigration(islands, migrantCount)
          print "Migrating"

        for index, island in enumerate(islands):


          newpop = self.stdMutation(island, 0, gen, elites)
          newpop = self.stdCrossover(newpop, 0.5)

          self.stdFitness(newpop)
          islands[index] = self.stdSelection(newpop, popSize)

        self.logData(islands)

      self.logFile.close()
      self.meanFile.close()
      for iFile in self.islandFiles:
        iFile.close()
      self.bestFitFile.close()
      self.bestFile.close()

    # Takes a list of all the populations and migrates a number of individuals from each island to the next
    def stdMigration(self, islands, migrantCount):
      migrants = []
      for index, island in enumerate(islands):
        migrants.append([])
        for _ in range(0, migrantCount):
          migrants[index].append(island.pop(np.random.randint(0,len(island))))

      for index, individuals in enumerate(migrants):
        islands[index-1].extend(individuals)

      return islands

    # Initializes a population using a normal distribution
    def stdPopInit(self, popSize, mean, mutationRate, arity):
      pop = []
      for i in range(0, popSize):
        #indiv = Individual(np.random.normal(mean, mutationRate, (arity*2)), arity, mutationRate)
        indiv = Individual(self.randomUniformPolar(1, arity*2), arity, mutationRate)
        pop.append(indiv)

      return pop

    def randomUniformPolar(self, radius, outputsize):
      mags = np.random.uniform(0,radius, outputsize)
      deg = np.random.uniform(0,2*np.pi,outputsize)
      outputmatrix = mags*np.sin(deg)
      return outputmatrix

    # Performs uniform crossover on the population
    def stdCrossover(self, population, xoverRate):
      children = []
      for index in range(0, self.runPopSize*2):
        parent1 = np.random.randint(0,len(population))
        parent2 = np.random.randint(0,len(population))
        childMutationRate = (population[parent1].mutationRate + population[parent2].mutationRate)/2

        childGenotype = []
        for geneIndex in range(0, len(population[0].genotype)):
          childGenotype.append((population[parent1].genotype[geneIndex] + population[parent2].genotype[geneIndex])/2)


        child = Individual(childGenotype, self.runArity, childMutationRate)
        children.append(child)

      population.extend(children)
      return population


    # Performs mutation on the population
    def stdMutation(self, population, mean, gen, elites):
      children = []
      learningRate = 1.0/np.sqrt(gen+1)
      children.extend(sorted(population, key=lambda x: x.fitness)[:elites])
      for index in range(0, self.runPopSize*5-elites):
        parentIndex = np.random.randint(0,len(population))
        parent = population[parentIndex]
        
        newindiv = Individual([], self.runArity, parent.mutationRate + np.exp(gauss(mean, learningRate)))
        for gene in parent.genotype:
          newindiv.genotype.append(gene + gauss(0, newindiv.mutationRate))

        children.append(newindiv)

      #population.extend(children) # This is disabled to replace the parents before crossover
      return children


    # Selects the popSize best individuals from the population
    def stdSelection(self, population, popSize):
      popSorted = sorted(population, key=lambda x: x.fitness)
      return popSorted[0:popSize]


    def stdFitness(self, population):
      for index, indiv in enumerate(population):

        constellation = np.zeros(indiv.arity, dtype=complex)

        for x in range(0, len(indiv.genotype), 2):
          constellation[x/2] = complex(indiv.genotype[x], indiv.genotype[x+1])

        tb = gnuradio.channel_model(constellation)
        tb.set_noise_level(0.4)
        tb.run()

        error_rate = np.mean(tb.blocks_vector_sink_x_1.data())
        indiv.fitness = error_rate


    def logInfo(self):
      if not os.path.exists(self.logPath):
        os.makedirs(self.logPath)

      self.logFile  = open(self.logPath + str(self.runCount) + "_info.txt", "w+")
      self.bestFile = open(self.logPath + str(self.runCount) + "_best.txt", "w+")
      self.bestFitFile = open(self.logPath + str(self.runCount) + "_bestfit.txt", "w+")
      self.meanFile = open(self.logPath + str(self.runCount) + "_mean.txt", "w+")
      for index in range(0, self.islandCount):
       self.islandFiles.append(open(self.logPath + str(self.runCount) + "_island" + str(index+1) + ".txt", "w+"))

      self.logFile.write("Number of Generations = %s\n" % self.runGenMax)
      self.logFile.write("Arity = %s\n" % self.runArity)
      self.logFile.write("Population Size = %s\n" % self.runPopSize)
      self.logFile.write("Number of Islands = %s\n" % self.islandCount)
      self.logFile.write("Length of each epoch = %s\n" % self.epochNum)
      self.logFile.write("Number of migrants = %s\n" % self.migrantCount)
      self.logFile.write("Number of elites = %s\n" % self.runElites)
    
    def logData(self, islands):
      meanFitness = 0
      bestFitness = 1
      for index, island in enumerate(islands):
        for indivIndex, individual in enumerate(island):
          if individual.fitness < bestFitness:
            bestFitness = indivIndex

          meanFitness += individual.fitness
          self.islandFiles[index].write(str(individual.genotype)+"\n")

        meanFitness /= len(island)

        self.meanFile.write("%f,"% meanFitness)
        self.bestFile.write(str(island[index].genotype)+"\n")
        self.bestFitFile.write("%f," % island[index].fitness)


class Individual(object):
  def __init__(self, genotype, arity, mutationRate):
    self.arity = arity
    self.genotype = genotype
    self.mutationRate = mutationRate
    self.fitness = 0

  def __str__(self):
    return "Arity: %d, Mutation Rate: %f, Fitness: %f, Genotype: %s" % (self.arity, self.mutationRate, self.fitness, self.genotype)
  
  def printGenotype(self):
    string = ""
    num = 0
    print self.genotype
    for gene in self.genotype:
      if num == 0:
        string = str(gene)
      else:
        string = string + "," + str(gene)

    return string + "\n"

if __name__ == '__main__':
  
  popSize = int(raw_input("Please enter the population size: "))
  numgen = int(raw_input("Please enter the number of generations: "))
  islandCount = int(raw_input("Please enter the number of islands: "))
  epochNum = int(raw_input("Please enter the length of each epoch: "))
  migrantCount = int(raw_input("Please enter the number of migrants for each epoch: "))
  elites = int(raw_input("Please enter the number of elites: "))
  arity  = int(raw_input("Please enter the arity of the constellation: "))
  runs = int(raw_input("Please enter the number of runs: "))

  runname = "ESIsland_" + str(raw_input("Please enter the name of run: "))
  logPath = runname + '/'
  #evolution = IslandEvolution(islandCount, logPath)

  for gen in range(0, runs):
	logPath2 = logPath + "r" + str(gen) + "/"
	evolution = IslandEvolution(islandCount, logPath2)
	evolution.evolve(popSize, epochNum, numgen, arity, migrantCount, elites)
