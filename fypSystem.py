# -*- coding: utf-8 -*-
# coding: utf-8
import os 
import random
import numpy as np
from scipy.stats.stats import pearsonr
from deap import base, creator, tools
# from bokeh.tests.test_driving import offset
from bokeh import *   # conda install bokeh
from bokeh.plotting import figure, show


try:
    from Tkinter import *
    import Tkinter as tk
    import ttk
    import tkFileDialog as filedialog
except:
    from tkinter import *
    import tkinter as tk
    from tkinter import ttk
    from tkinter import filedialog

import datetime
import csv
import shutil

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import matplotlib.animation as animation
from matplotlib import style

from matplotlib import pyplot as plt



doLogging = True
minInt = float("-inf")
#set default value
# maxYear - minYear = size of arrays
minYear = 1970
maxYear = 2016
populationNumber = 50
#mininum number of indicators in a chromosome
minIndicators = 1
headers = []
compare_headers = []
data_matrix = []
compare_data_matrix = []


# CXPB  is the probability with which two individuals
#       are crossed
#
# MUTPB is the probability for mutating an individual
CXPB, MUTPB = 0.5, 0.2

#set some font style, later for tkinter
LARGE_FONT= ("Verdana", 12)
NORM_FONT = ("Helvetica", 10)
SMALL_FONT = ("Helvetica", 8)
style.use("ggplot")

#plot 3 graph 
f = Figure(figsize=(10,6), dpi=100)
a = f.add_subplot(311)
b = f.add_subplot(312)
c = f.add_subplot(313)

#plot single graph
fig = Figure(figsize=(12,5.5), dpi=100)
ax = fig.add_subplot(111)

figure_graph2 = Figure(figsize=(12,5.5), dpi=100)
ax2 = figure_graph2.add_subplot(111)

figure_graph3 = Figure(figsize=(12,5.5), dpi=100)
ax3 = figure_graph3.add_subplot(111)


prev_path = "/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/"
if not os.path.exists(prev_path):
    os.makedirs(prev_path)
    open(prev_path+"ALL_previous.csv", 'w+').close()
    open(prev_path+"evolution1_previous.csv", 'w+').close()
    open(prev_path+"evolution2_previous.csv", 'w+').close()
    open(prev_path+"evolution3_previous.csv", 'w+').close()


def popupmsg(msg):
    popup = tk.Tk()
    popup.wm_title("!")
    label = ttk.Label(popup, text=msg, font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)
    B1 = ttk.Button(popup, text="Okay", command = popup.destroy)
    B1.pack()
    popup.mainloop()


def animate_graph1(i):
    xar = []
    yar = []
    zar = []
    
    file_size = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv").st_size

    if file_size != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv", "r") as f:
                reader = csv.reader(f, skipinitialspace=TRUE)
                #skip the header
                next(reader)
                for row in reader:
                    xar.append(int(row[0]))
                    yar.append(float(row[1]))
                    zar.append(float(row[5]))
        except IOError:
            print("File not found!")

    # Plot line graphs
    ax.clear()
    ax.plot(xar, yar, color="purple", marker="o" , label = "Average Fitness")
    ax.plot(xar, zar, 'bo-', label = "Max Fitness")
    ax.legend()
    ax.set_title('Graph 1')
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitnesses')


def animate_graph2(i):
    xar2 = []
    yar2 = []
    zar2 = []

    file_size2 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv").st_size

    if file_size2 != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv", "r") as f2:
                reader2 = csv.reader(f2, skipinitialspace=TRUE)
                #skip the header
                next(reader2)
                for row in reader2:
                    xar2.append(int(row[0]))
                    yar2.append(float(row[1]))
                    zar2.append(float(row[5]))
        except IOError:
            print("File not found!")

    # Plot line graphs
    ax2.clear()
    ax2.plot(xar2, yar2, color="purple", marker="o" , label = "Average Fitness")
    ax2.plot(xar2, zar2, 'bo-', label = "Max Fitness")
    ax2.legend()
    ax2.set_title('Graph 2')
    ax2.set_xlabel('Generation')
    ax2.set_ylabel('Fitnesses')


def animate_graph3(i):
    xar3 = []
    yar3 = []
    zar3 = []

    file_size3 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv").st_size

    if file_size3 != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv", "r") as f3:
                reader3 = csv.reader(f3, skipinitialspace=TRUE)
                #skip the header
                next(reader3)
                for row in reader3:
                    xar3.append(int(row[0]))
                    yar3.append(float(row[1]))
                    zar3.append(float(row[5]))
        except IOError:
            print("File not found!")

    # Plot line graphs   
    ax3.clear()
    ax3.plot(xar3, yar3, color="purple", marker="o" , label = "Average Fitness")
    ax3.plot(xar3, zar3, 'bo-', label = "Max Fitness")
    ax3.legend()
    ax3.set_title('Graph 3')
    ax3.set_xlabel('Generation')
    ax3.set_ylabel('Fitnesses')


class AutoScrollbar(Scrollbar):
    # a scrollbar that hides itself if it's not needed.  only
    # works if you use the grid geometry manager.
    def set(self, lo, hi): 
        if float(lo) <= 0.0 and float(hi) >= 1.0:
            # grid_remove is currently missing from Tkinter!
            self.tk.call("grid", "remove", self)
        else:
            self.grid()
        Scrollbar.set(self, lo, hi)
    def pack(self, **kw):
        raise (TclError,"pack cannot be used with this widget") 
    def place(self, **kw):
        raise (TclError, "place cannot be used  with this widget")


def animate(i):
    xar = []
    yar = []
    zar = []

    file_size = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv").st_size

    if file_size != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv", "r") as f:
                reader = csv.reader(f, skipinitialspace=TRUE)
                #skip the header
                next(reader)
                for row in reader:
                    xar.append(int(row[0]))
                    yar.append(float(row[1]))
                    zar.append(float(row[5]))
        except IOError:
            print("File not found!")

    a.clear()  
    a.plot(xar,yar, color="purple", marker="o", label = "Average Fitness")
    a.plot(xar,zar, 'bo-', label = "Max Fitness")
    a.legend()
    a.set_ylabel('Fitnesses')

#def animate1(i):
    xar2 = []
    yar2 = []
    zar2 = []

    file_size2 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv").st_size

    if file_size2 != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv", "r") as f2:
                reader2 = csv.reader(f2, skipinitialspace=TRUE)
                #skip the header
                next(reader2)
                for row in reader2:
                    xar2.append(int(row[0]))
                    yar2.append(float(row[1]))
                    zar2.append(float(row[5]))
        except IOError:
            print("File not found!") 

    b.clear()
    b.plot(xar2,yar2, color="orange", marker="o", label = "Average Fitness")
    b.plot(xar2,zar2, 'bo-', label = "Max Fitness")
    b.legend()
    b.set_ylabel('Fitnesses')

#def animate2(i):
    xar3 = []
    yar3 = []
    zar3 = []

    file_size3 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv").st_size

    if file_size3 != 0:
        try:
            with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv", "r") as f3:
                reader3 = csv.reader(f3, skipinitialspace=TRUE)
                #skip the header
                next(reader3)
                for row in reader3:
                    xar3.append(int(row[0]))
                    yar3.append(float(row[1]))
                    zar3.append(float(row[5]))
        except IOError:
            print("File not found!")

    c.clear()
    c.plot(xar3,yar3, color="green", marker="o", label = "Average Fitness")
    c.plot(xar3,zar3, 'bo-', label = "Max Fitness")
    c.legend()
    c.set_xlabel('Generation')
    c.set_ylabel('Fitnesses')


def doStuff():
    if not directory1Select.folder_path:
        popupmsg("Please select input path for Select Input Folder ! ")
    else:
        path = "%s/"%(directory1Select.folder_path)
    
    if not directory2Select.folder_path:
        popupmsg("Please select compare path for Select Compare Folder ! ")
    else:
        compare_path = "%s/"%(directory2Select.folder_path)

    if not directory3Select.folder_path:
        popupmsg("Please select output path for Select Output Folder ! ")
    else:
        stats_path = "%s/"%(directory3Select.folder_path)

    global inputRowHeader
    inputRowHeader = str(v.get())

    global compareRowHeader
    compareRowHeader = str(v1.get())

    global populationNumber

    global CXPB

    global MUTPB

    
    if(doLogging):
        logFile = open(stats_path + "logs.txt", "w")
        simpleLog = open(stats_path +"logs_simplified.txt","w")
    
    
    getFileExtension(path)
    getFileExtension(compare_path)
    #Validate row header inside file
    getHeadersbyYear(path,inputRowHeader)
    getHeadersbyYear(compare_path,inputRowHeader)

    # creates a giant 2D array to lookup operands == data in time series
    # there's no year in the first column
    headers = getHeaders(path, inputRowHeader)
    
    files = os.listdir(compare_path) 
    
    # num of runs controls how many times we are going to run ga on
    # the same compare indicator so that we can collect some stats
    # in collectiveStats file
    num_of_runs = 3
    collectiveStats_headers = "mean, standard deviation, minimum CC, indicators, maximum CC, indicators\n"
    
    for filename in files:    #loop each of the files

        #collectiveStats records the best individual from individual ga runs
        #for each compare_indicators
        collectiveStats = open(stats_path + "ALL_"+filename, "w")
        collectiveStats.write(collectiveStats_headers)
        for i in range(num_of_runs):
            compare_data_matrix = []    
    
            statsFile = open(stats_path + "evolution"+str(i+1) + "_" + filename, "w")
            #writes the first line in stats file for column headers
            statsFile.write("Generation, Average, Std, Min_Fitness, Min_Individual, Max_Fitness, Max_Individual\n")
            
            #row by column matrix
            data_matrix =  [[0 for x in range(len(headers))] for y in range(maxYear - minYear + 1)]
            
            if(str(inputRowHeader) == "indicatorList"):
                parseDataFile(path, data_matrix, len(headers), False)
            elif(str(inputRowHeader) == "yearList"):
                parseDataFilebyYear(path, data_matrix, inputRowHeader, len(headers), False)

            #gets the comparison data
            compare_headers = getHeadersFromFile(compare_path, filename, compareRowHeader)
            compare_data_matrix = [[0 for x in range(len(compare_headers))] for y in range(maxYear - minYear + 1)]

            if(str(inputRowHeader) == "indicatorList"):
                parseSingleDataFile(compare_path, filename, compare_data_matrix, len(compare_headers), True)    
            elif(str(inputRowHeader) == "yearList"):
                parseSingleDataFilebyYear(compare_path, filename, compare_data_matrix, compareRowHeader, len(compare_headers), True)

            #generate random chromosomes
            line = startGA(headers, compare_headers, data_matrix, compare_data_matrix, maxYear - minYear + 1, len(headers), logFile, simpleLog, statsFile)
            collectiveStats.write(line)
        collectiveStats.close()
    if(doLogging):
        logFile.close()
        simpleLog.close()
    source = r"%s" % stats_path
    destination = r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/"
    consolidate(source,destination)
    print("All done!")

def resolve_path(filename, destination_dir, fileExt):
    string1 = str(filename)
    substring1 = "ALL_"
    substring2= "evolution1_"
    substring3= "evolution2_"
    substring4= "evolution3_"
    newName = ""
    if string1.find(substring1) == 0:
        newName = substring1 + "previous" + fileExt

    elif string1.find(substring2) == 0:
        newName = substring2 + "previous" + fileExt

    elif string1.find(substring3) == 0:
        newName = substring3 + "previous" + fileExt

    elif string1.find(substring4) == 0:
        newName = substring4 + "previous" + fileExt

    else:
        newName = filename

    dest = os.path.join(destination_dir, newName)
    return dest

def consolidate(source, destination):
    if not os.path.exists(destination):
        os.makedirs(destination)
    for root, dirs, files in os.walk(source):
        for f in files:
            #if f.lower().endswith(extension):
            source_path = os.path.join(root, f)
            fileExtension = os.path.splitext(source_path)[1]
            destination_path = resolve_path(f, destination, fileExtension)
            shutil.copyfile(source_path, destination_path)

def getFileExtension(folder):
    for root, dirs, files in os.walk(folder):
        for filename in files:
            source_path = os.path.join(root, filename)
            fileExtension = os.path.splitext(source_path)[1]
            if(str(fileExtension) != ".csv"):
                popupmsg("%s is not a type of csv file! " % source_path)

def startGA(headers, compare_headers, matrix, compare_matrix, rowNum, colNum, logFile, simpleLog, statsFile):
    line_to_write = ""
    # To assure reproductibility, the RNG seed is set prior to the items
    # dict initialization. It is also seeded in main().
    random.seed(64)
    
    #weights=(1.0,) indicates that we only seek to maximize the only 1 objective function
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    
    toolbox = base.Toolbox()
    
    #max number of possible indicator combinations being colNum/2
    toolbox.register("chromosome", initIndividual, creator.Individual, colNum, minIndicators, int(colNum/2))    
    toolbox.register("population", tools.initRepeat, list, toolbox.chromosome)  
    
    #all of the evaluation functions and operators
    toolbox.register("evaluate", evaluateInd)

    # register the crossover operator
    toolbox.register("mate", tools.cxTwoPoint)
    # a crossover may potentially introduce duplicate indicator, so
    # we need to remove it from an individual
    toolbox.register("mate_correction", mateCorrectionFunc)

    # register a mutation operator with a probability to
    # flip each attribute/gene of 0.05
    #toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("mutate", mutationFunc, colNum)

    # operator for selecting individuals for breeding the next
    # generation: each individual of the current generation
    # is replaced by the 'fittest' (best) of three individuals
    # drawn randomly from the current generation.
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Variable keeping track of the number of generations
    g = 0

    #run GA
    pop = toolbox.population(n=populationNumber)
    
    print("Start of evolution")
    # Evaluate the entire population
    #fitnesses = list(map(toolbox.evaluate, pop))
    fitnesses = [toolbox.evaluate(ind, matrix, compare_matrix, headers, compare_headers,logFile,g) for ind in pop]

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    print("  Evaluated %i individuals" % len(pop))
    logFile.write("  Evaluated %i individuals \n" % len(pop))

    # Extracting all the fitnesses of 
    fits = [ind.fitness.values[0] for ind in pop]
   
    length = len(pop)
    mean = sum(fits) / length
    sum2 = sum(x*x for x in fits)
    std = abs(sum2 / length - mean**2)**0.5
    index_min = np.argmin(fits)
    index_max = np.argmax(fits)
   
    logFile.write("min fitness: " + str(min(fits)) + "     max fitness: " + str(max(fits)) + "     mean fitness: " + str(mean) + "\n")
    if(doLogging):
        logResult(simpleLog, pop, g, min(fits), max(fits), mean)

    line_to_write = "%s, %s, %s, %s,\"%s\", %s,\"%s\"\n" % (g, mean, std, min(fits), pop[index_min], max(fits), pop[index_max])
    statsFile.write(line_to_write)


    # Begin the evolution
    while max(fits) < 100 and g < populationNumber:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                if(len(child1) > 1 and len(child2) > 1):
                    toolbox.mate(child1, child2)
                    child1 = toolbox.mate_correction(child1)
                    child2 = toolbox.mate_correction(child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
        
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]

        fitnesses = [toolbox.evaluate(ind, matrix, compare_matrix, headers, compare_headers,logFile,g) for ind in pop]
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        print("  Evaluated %i individuals" % len(invalid_ind))
        logFile.write("  Evaluated %i individuals" % len(invalid_ind))
        #The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        index_min = np.argmin(fits)
        index_max = np.argmax(fits)
        
        #outputs min max individuals to the stats file
        line_to_write = "%s, %s, %s, %s,\"%s\", %s,\"%s\"\n" % (g, mean, std, min(fits), pop[index_min], max(fits), pop[index_max])
        statsFile.write(line_to_write)

        logFile.write("\n min fitness: " + str(min(fits)) + "     max fitness: " + str(max(fits)) + "     mean fitness: " + str(mean) + "\n")
        if(doLogging):
            logResult(simpleLog, pop, g, min(fits), max(fits), mean)

        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
    
    print("-- End of (successful) evolution --")
    logFile.write("-- End of (successful) evolution --")
    simpleLog.write("-- End of (successful) evolution --")

    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
    logFile.write("Best individual is %s, %s \n" % (best_ind, best_ind.fitness.values))
    simpleLog.write("Best individual is %s, %s \n" % (best_ind, best_ind.fitness.values))
    
    statsFile.close()
    line_to_write = "%s, %s, %s,\"%s\", %s,\"%s\"\n" % (mean, std, min(fits), convertToNames(headers, pop[index_min]), max(fits),convertToNames(headers, pop[index_max]))
    return line_to_write

def convertToNames(headers, offset_list):
    result = ""
    for offset in offset_list:
        result += headers[offset] + ";"
    return result

# a crossover may potentially introduce duplicate indicator, so
# we need to remove it from an individual
def mateCorrectionFunc(individual):
    # the list cannot be created directly since it has the fitness attribute
    # so we will create an individual then assign fitness to it
    fitness = individual.fitness
    individual = creator.Individual(list(set(individual)))
    individual.fitness = fitness
    return individual

def mutationFunc(maxRange, individual):
    offset = random.randint(0, len(individual)-1)
    val = random.randint(0, maxRange-1)
    while(any(val == elem for elem in individual)):
        val = random.randint(0, maxRange-1)
    individual[offset] = val    
    return 

def evaluateInd(individual, l_matrix, compare_data_matrix,headers,compare_headers,logFile,generationNumber):
    total = 0.0
    total_two_tail = 0.0
    compareDataMatrixColNum = len(compare_data_matrix[0])
    individualAverage = 0
    PvalueAverage = 0
   
    logFile.write("---- Generation: " + str(generationNumber) + " ----\n")
    logFile.write("---- Individual: " + str(individual) + " ----\n")
    
    #iterate through the genes in an individual
    for columnOffset in individual:
        # gets the list from matrix and find the start and end year where data isn't 0 as start and end
        start_row_index, end_row_index = findStartAndEndIndex(l_matrix, columnOffset)
        
        seriesData = getSubseriesFromMatrix(l_matrix, start_row_index, end_row_index, columnOffset)
        currentIndicatorAverage = 0
        total = 0
        total_two_tail = 0
        for i in range(compareDataMatrixColNum):
            subseries = getSubseriesFromMatrix(compare_data_matrix, start_row_index, end_row_index, i)
            #use the start and end year to compute pearson CC on all series in compare_data
            cc, two_tail = pearsonr(seriesData, subseries)
            #NOTE: Debug use only
            #print(seriesData)
            #print(subseries)
            #if(two_tail == 1.0):
            #    print(str(start_row_index) + " " + str(end_row_index) + " " + str(columnOffset))
            #    print(seriesData)
            #    print(subseries)

            if(doLogging):
                maxYeardiff = len(l_matrix) - end_row_index
                logPopulation(logFile, headers, compare_headers, individual, columnOffset, (i), seriesData, subseries, start_row_index, maxYeardiff, cc, two_tail)

            total += cc
            total_two_tail += two_tail
        currentIndicatorAverage = total / (compareDataMatrixColNum)
        individualAverage += currentIndicatorAverage

        currentPvalue = total_two_tail / (compareDataMatrixColNum)
        PvalueAverage += currentPvalue
        
    #get the average
    individualAverage = individualAverage/len(individual)
    logFile.write("Average fitness for current population: " + str(individualAverage) + "\nAverage p_value: " + str(PvalueAverage) + "\n")
    return individualAverage, #the comma is VERY IMPORTANT!!!


def getSubseriesFromMatrix(l_matrix, startRIndex, endRIndex, columnOffset):
    l_list = []
    for i in range(startRIndex,endRIndex+1):
        l_list.append(l_matrix[i][columnOffset])
    return l_list


def findStartAndEndIndex(l_matrix, columnOffset):
    startIndex = 0
    endIndex = len(l_matrix) - 1
    for i in range(len(l_matrix)):
        if(l_matrix[i][columnOffset] != minInt):
            startIndex = i
            break
    for i in range(len(l_matrix)-1,0,-1):
        if(l_matrix[i][columnOffset] != minInt):
            endIndex = i
            break
    return startIndex, endIndex


def initIndividual(icls, totalOffsets, minLength, maxLength):
    ind = creator.Individual(np.random.choice(totalOffsets,random.randint(minLength,maxLength), replace = False))
    return ind    


def parseSingleDataFile(folder, file, l_matrix, col_num, ignoreZero):
    #col_num needs to be +1 since the getHeaders() ignores the year column
    total_col_num = col_num + 1
    col = 0
    col_increment = 0
    
    count = 0
    handle = open(folder+file,"r")
    line = handle.readline().replace("\n","")
    try:
        while line:
            if(len(line) == 0):
                break
            #ignore sthe first row as it names the columns (which has been read
            # by getHeaders()))
            if(count != 0):
                #ignores the first column since it's year, but we use it as index into
                #matrix just in case the year for this file doesn't start with minYear        
                array = line.split(",")
                col_increment = len(array)
                for idx,item in enumerate(array):
                    if(idx == 0): #year
                        index = int(array[idx]) - minYear                   
                    elif(idx < total_col_num):
                        if(array[idx] != None and len(array[idx]) > 0):
                            l_matrix[index][(idx-1)] = float(array[idx])
                        else:
                            if(ignoreZero):
                                l_matrix[index][(idx-1)] = 0
                            else:
                                l_matrix[index][(idx-1)] = minInt
            count = count + 1        
            line = handle.readline().replace("\n","")
        col = col + col_increment
        handle.close()
    except:
            popupmsg("Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
          
        
def parseDataFile(folder, l_matrix, col_num, ignoreZero):
    #col_num needs to be +1 since the getHeaders() ignores the year column
    total_col_num = col_num + 1
    col = 0
    col_increment = 0
    #reads all .csv files, ignores the first column since it's YEAR
    files = os.listdir(folder)

    for file in files:
        count = 0
        handle = open(folder+file,"r")
        line = handle.readline().replace("\n","")
        #replace() string.replace(old, new, count)
        try:
            while line:
                if(len(line) == 0):
                    break
                #ignore sthe first row as it names the columns (which has been read
                # by getHeaders()))
                if(count != 0):
                    #ignores the first column since it's year, but we use it as index into
                    #matrix just in case the year for this file doesn't start with minYear        
                    array = line.split(",")
                    col_increment = len(array)
                    for idx,item in enumerate(array):
                        if(idx == 0): #year
                            index = int(array[idx]) - minYear                     
                        elif(idx < total_col_num):
                            if(array[idx] != None and len(array[idx]) > 0):
                                l_matrix[index][(idx-1)] = float(array[idx])
                            else:
                                if(ignoreZero):
                                    l_matrix[index][(idx-1)] = 0
                                else:
                                    l_matrix[index][(idx-1)] = minInt
                count = count + 1        
                line = handle.readline().replace("\n","")
            col = col + col_increment
            handle.close()
        except:
            popupmsg("Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
            


def parseSingleDataFilebyYear(folder, file, compare_matrix, rowHeader, col_num, ignoreZero):
    total_col_num = col_num + 1
    handle = open(folder+file,"r")
    year = []
    count = 0
    with handle as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            #skip the first column as it is Year in str
            #the rest of the column will be year in int
            year = row[1:]
            break
        for row2 in reader:
            array = row2[1:]
            index = int(year[count]) - minYear

            try:
                for idx,item in enumerate(array):
                    if(idx < total_col_num):
                        if(array[idx] != None and len(array[idx]) > 0):
                            compare_matrix[idx][index] = float(array[idx])
                        else:
                            if(ignoreZero):
                                compare_matrix[idx][index] = 0
                            else:
                                compare_matrix[idx][index] = minInt
                count += 1
            except:
                popupmsg("Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
          
    handle.close()


def parseDataFilebyYear(folder, l_matrix, rowHeader, col_num, ignoreZero):
    total_col_num = col_num + 1
    files = os.listdir(folder)
    for file in files:
        handle = open(folder+file,"r")
        year = []
        count = 0
        with handle as f:
            reader = csv.reader(f, skipinitialspace=True)
            for row in reader:
                #skip the first column as it is Year in str
                #the rest of the column will be year in int
                year = row[1:]
                break
            for row2 in reader:
                array = row2[1:]
                index = int(year[count]) - minYear

                try:
                    for idx,item in enumerate(array):
                        if(idx < total_col_num):
                            if(array[idx] != None and len(array[idx]) > 0):
                                l_matrix[idx][index] = float(array[idx])
                            else:
                                if(ignoreZero):
                                    l_matrix[idx][index] = 0
                                else:
                                    l_matrix[idx][index] = minInt
                    count += 1
                except:
                    popupmsg("Data format is incorrect!\nPlease ensure only number format is allowed!\n*Please remove comma within number")
        handle.close()


def getHeadersFromFile(folder, file, rowHeader):
    l_headers = []
    handle=open(folder + file, "r")
    if(str(rowHeader) == "indicatorList"):
        #ignores the first column since it's year
        array = handle.readline().split(",")[1:]
        for item in array:
            l_headers.append(item.replace("\n",""))
    if(str(rowHeader) == "yearList"):
        with handle as f:
            reader = csv.reader(f, skipinitialspace=True)
            #skip the header
            next(reader)
            for row in reader:
                l_headers.append(row[0])
    handle.close()
    return l_headers        

def getHeaders(folder,rowHeader):
    l_headers = []
    files = os.listdir(folder)
    for file in files:
        handle = open(folder + file, "r")
        if(str(rowHeader) == "indicatorList"):
            #ignores the first column since it's year
            array = handle.readline().split(",")[1:]
            for item in array:
                l_headers.append(item.replace("\n",""))
        
        if(str(rowHeader) == "yearList"):
            with handle as f:
                reader = csv.reader(f, skipinitialspace=True)
                #skip the header
                next(reader)
                for row in reader:
                    l_headers.append(row[0])
        handle.close()
    return l_headers

def getHeadersbyYear(folder,rowHeader):
    global minYear
    global maxYear
    files = os.listdir(folder)
    for file in files:
        handle = open(folder + file, "r")
        l_headers_year = []
        if(str(rowHeader) == "indicatorList"):
            with handle as f:
                
                reader = csv.reader(f, skipinitialspace=True)
                #skip the header
                next(reader)
                for row in reader:
                    l_headers_year.append(row[0])
        if(str(rowHeader) == "yearList"):
            #ignores the first column since it's year
            array = handle.readline().split(",")[1:]
            for item in array:
                l_headers_year.append(item.replace("\n",""))

        try:
            minYear = int(l_headers_year[0])       
            maxYear = int(l_headers_year[-1])

        except:
            popupmsg("Please check your row header! ")

        """
        if(int(l_headers_year[0]) != minYear):
            popupmsg("This is not Min Year! ")
        if(int(l_headers_year[-1]) != maxYear):
            popupmsg("This is not Max Year! ")
        """
        handle.close()
    return l_headers_year


def logPopulation(logFile, headers, compare_headers, ind, offset, compareidx, input_matrix, compare_matrix, start, end, cc, pvalue):
        logFile.write("input subset: ['" + str(headers[offset]) + "']     compare subset: ['" + str(compare_headers[compareidx]) + "']     min year: ['" + str(minYear+start) + "']     max year: ['" + str(maxYear-end+1) + "']     cc: ['" + str(cc) + "']     p_value: ['" + str(pvalue) + "']\n")
    
def logResult(simpleLog, pop, generation,minf,maxf,meanf):
    simpleLog.write("---- Generation: " + str(generation) + " ----\n")
    for ind in pop:
        simpleLog.write(str(ind) + " Fitness:"+ str(ind.fitness) + "\n")
    simpleLog.write("Min fitness: " + str(minf) + "\nMax fitness: " + str(maxf) + "\nMean fitness: " + str(meanf) + "\n")
     

def tutorial():

    def configp():
        tut.destroy()
        tut2 = tk.Tk()

        tut2.wm_title("Configuration Page!")
        
        label = ttk.Label(tut2, text="Configuration Page", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)

        label1 = ttk.Label(tut2, text="Before proceeding, Please ensure the following: \n -Only CSV file is included in the INPUT and COMPARE folder \n -Indicator's name within CSV file is not consisted of ',' \n\n -->Select the INPUT, COMPARE and OUTPUT folder \n using the 'Browse Folder' button \n\n -->Row header is based on the first row within the CSV file \n\n --> If the population number and probability are not selected, \n the default value will be used: \n Population Number = 50 \n CrossOver Probability = 0.5 \n Mutation Probability = 0.2", font = SMALL_FONT)
        label1.pack()

        button1 = ttk.Button(tut2, text="Done!", command= tut2.destroy)
        button1.pack()

        tut2.mainloop()

    def graphp():
        tut.destroy()
        tut3 = tk.Tk()

        tut3.wm_title("Graph Page!")
        
        label = ttk.Label(tut3, text="Graph Page", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)

        label1 = ttk.Label(tut3, text="* All the graphs are depending on previous computed result, \n once computed again,  then only it will refreshed \n\n -->Graph Page using result from three evolution cycle \n\n -->Detail Graph 1 using result from 'evolution1_previous.csv' \n\n -->Detail Graph 2 using result from 'evolution2_previous.csv' \n\n -->Detail Graph 3 using result from 'evolution3_previous.csv' \n\n", font=SMALL_FONT)
        label1.pack()

        button1 = ttk.Button(tut3, text="Done!", command= tut3.destroy)
        button1.pack()

        tut3.mainloop()
    
    def prevtable():
        tut.destroy()
        tut4 = tk.Tk()

        tut4.wm_title("Previous Table!")
        
        label = ttk.Label(tut4, text="Previous Table", font=NORM_FONT)
        label.pack(side="top", fill="x", pady=10)

        label1 = ttk.Label(tut4, text="* Previous Table only able to take previous result, will be updated later to live \n\n -->Previous Table 1 using result from first evolution in 'ALL_previous.csv' and 'evolution1_previous' \n\n -->Previous Table 2 using result from second evolution in 'ALL_previous.csv' and 'evolution2_previous' \n\n -->Previous Table 3 using result from third evolution in 'ALL_previous.csv' and 'evolution3_previous' ", font=SMALL_FONT)
        label1.pack()

        button1 = ttk.Button(tut4, text="Done!", command= tut4.destroy)
        button1.pack()

        tut4.mainloop()

    tut = tk.Tk()
    tut.wm_title("Tutorial")
    label = ttk.Label(tut, text="What do you need help with?", font=NORM_FONT)
    label.pack(side="top", fill="x", pady=10)

    B1 = ttk.Button(tut, text = "Configuration Page", command=configp)
    B1.pack()

    button2 = ttk.Button(tut, text="Graph Page", command= graphp)
    button2.pack()

    button3 = ttk.Button(tut, text="Previous Table Page", command= prevtable)
    button3.pack()

    tut.mainloop()


class FinancialDataSeries(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        tk.Tk.wm_geometry(self,"1200x700")
        # tk.Tk.iconbitmap(self,default="Financial.ico")
        tk.Tk.wm_title(self, "Extracting Hidden Trends and Patterns from Financial Data Series")

        container = tk.Frame(self)

        container.pack(side="top", fill="both", expand = True)

        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        menubar = tk.Menu(container)
        filemenu = tk.Menu(menubar)
        filemenu.add_command(label="Save settings", command = lambda: popupmsg("Not supported just yet!"))
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=quit)
        menubar.add_cascade(label="File", menu=filemenu)

        configure = tk.Menu(menubar)
        configure.add_command(label="Configure Page", command=lambda: self.show_frame(ConfigPage))
        menubar.add_cascade(label="Configuration", menu=configure)
        
        graph = tk.Menu(menubar)
        graph.add_command(label="Graph Page", command=lambda: self.show_frame(ThreeGraph))
        graph.add_command(label="Detail Graph 1", command=lambda: self.show_frame(Detail_Graph1))
        graph.add_command(label="Detail Graph 2", command=lambda: self.show_frame(Detail_Graph2))
        graph.add_command(label="Detail Graph 3", command=lambda: self.show_frame(Detail_Graph3))
        menubar.add_cascade(label="Graph View", menu=graph)
        
        configureT = tk.Menu(menubar, tearoff = 1)
        configureT.add_command(label="Previous Table 1", command=lambda: self.show_frame(TablePage1))
        configureT.add_command(label="Previous Table 2", command=lambda: self.show_frame(TablePage2))
        configureT.add_command(label="Previous Table 3", command=lambda: self.show_frame(TablePage3))
        menubar.add_cascade(label="Table Form", menu=configureT)

        helpmenu = tk.Menu(menubar)
        helpmenu.add_command(label="Tutorial", command=tutorial)
        menubar.add_cascade(label="Help", menu=helpmenu)


        tk.Tk.config(self, menu=menubar)

        self.frames = {}

        #multiple frame
        self.pages= (StartPage, ConfigPage, ThreeGraph, Detail_Graph1, Detail_Graph2, Detail_Graph3, TablePage1, TablePage2, TablePage3)
        for F in self.pages:

            frame = F(container, self, self.pages)

            self.frames[F] = frame

            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller,pages):
        tk.Frame.__init__(self,parent)
        label = tk.Label(self, text=("""This analysis application
        use at your own risk. There is no promise
        of warranty. """), font=LARGE_FONT)
        label.pack(pady = 10, padx = 10)

        label1 = tk.Label(self, text=("** ONLY CSV files are supported in this application **"), font=NORM_FONT)
        label1.pack(pady = 10, padx = 10)

        button1 = ttk.Button(self, text="Agree",
                            command=lambda: controller.show_frame(ConfigPage))
        button1.pack()

        button2 = ttk.Button(self, text="Disagree",
                            command=sys.exit)
        button2.pack()


class ConfigPage(tk.Frame):

    def __init__(self, parent, controller,pages):
        tk.Frame.__init__(self, parent)
        label = tk.Label(self, text="Configuration Page", font=LARGE_FONT)
        label.grid(row = 0, column = 1)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1, column = 1)

        button2 = tk.Button(self, text="Graph Page",
                            command=lambda: controller.show_frame(ThreeGraph))
        button2.grid(row = 2, column = 1)
   
        global directory1Select
        global directory2Select
        global directory3Select
        global population
        global crossover
        global mutation
        global minyear
        global maxyear
        global v
        global v1

        directory1Select = self.FolderSelect(self,"Select Input Folder")

        directory1Select.grid(row = 4, columnspan = 3)
        directory2Select = self.FolderSelect(self,"Select Compare Folder")
        directory2Select.grid(row = 5, columnspan = 3)

        directory3Select = self.FolderSelect(self,"Select Output Folder")
        directory3Select.grid(row = 6, columnspan = 3)

        v = StringVar()
        v.set("indicatorList")
        v1 = StringVar()
        v1.set("indicatorList")

        lbl = Label(self, text="Choose Row Header for Input Folder", width=30, anchor="w")
        lbl.grid(row = 7, column = 0)

        radiobtn = Radiobutton(self, text = "Indicator List", variable = v, value = "indicatorList")
        radiobtn.grid(row = 7, column = 1)

        radiobtn = Radiobutton(self, text = "Year List", variable = v, value = "yearList")
        radiobtn.grid(row = 7, column = 2)

        lbl1 = Label(self, text="Choose Row Header for Compare Folder", width=30, anchor="w")
        lbl1.grid(row = 8, column = 0)

        radiobtn1 = Radiobutton(self, text = "Indicator List", variable = v1, value = "indicatorList")
        radiobtn1.grid(row = 8, column = 1)

        radiobtn1 = Radiobutton(self, text = "Year List", variable = v1, value = "yearList")
        radiobtn1.grid(row = 8, column = 2)

        label = Label(self, text="Population Number", width=30, anchor="w")
        label.grid(row = 9, column = 0)

        population = Scale(self,  from_=0, to=100, resolution = 1, orient=HORIZONTAL, length=250)
        population.grid(row = 9, column = 1, columnspan = 2)

        label1 = Label(self, text="Cross-over Probability", width=30, anchor="w")
        label1.grid(row = 10, column = 0)

        crossover = Scale(self,  from_=0, to=0.9, resolution = 0.1, orient=HORIZONTAL, length=250)
        crossover.grid(row = 10, column = 1, columnspan = 2)

        label2 = Label(self, text="Mutation Probability", width=30, anchor="w")
        label2.grid(row = 12, column = 0)

        mutation= Scale(self,  from_=0, to=0.9, resolution = 0.1, orient=HORIZONTAL, length=250)
        mutation.grid(row = 12, column = 1, columnspan = 2)

        c = ttk.Button(self, text="find", command=lambda: [doStuff(),controller.show_frame(ThreeGraph),popupmsg("Done!")])
        c.grid(row = 17, column = 1)

    class FolderSelect(Frame):
        def __init__(self,parent=None,folderDescription="",**kw):
            Frame.__init__(self,master=parent,**kw) #highlightbackground="red", highlightcolor="red", highlightthickness=1,bd=0,
            
            self.folderPath = StringVar()
            self.lblName = Label(self, text=folderDescription, width=20, anchor="w")
            self.lblName.grid(row=0,column=0)
            self.entPath = Entry(self, textvariable=self.folderPath, width=30)
            self.entPath.config(state='readonly')
            self.entPath.grid(row=0,column=1)
            self.btnFind = ttk.Button(self, text="Browse Folder",command=self.setFolderPath)
            self.btnFind.grid(row=0,column=2)
        def setFolderPath(self):
            folder_selected = filedialog.askdirectory()
            self.folderPath.set(folder_selected)
        @property
        def folder_path(self):
            return self.folderPath.get()


class ThreeGraph(tk.Frame):

    def __init__(self, parent, controller,pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Graph Page", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1)

        button2 = tk.Button(self, text="Go to Detail Graph 1",
                            command=lambda: controller.show_frame(Detail_Graph1))
        button2.grid(row = 2)

        hbar=AutoScrollbar(self,orient=HORIZONTAL)
        vbar=AutoScrollbar(self,orient=VERTICAL)       

        canvas=FigureCanvasTkAgg(f,self)
        canvas.get_tk_widget().config(bg='#FFFFFF',scrollregion=(0,0,1200,600))
        canvas.get_tk_widget().config(width=1150,height=550)
        canvas.get_tk_widget().config(xscrollcommand=hbar.set, yscrollcommand=vbar.set)       
        canvas.get_tk_widget().grid(row = 3, sticky=W+E+N+S)       

        toolbarFrame = tk.Frame(self)
        toolbarFrame.grid(row=22, sticky=W+E+N+S)
        toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)
        toolbar.update()
         
        hbar.grid(row = 4, column = 0, sticky = W+E)
        hbar.config(command=canvas.get_tk_widget().xview)
        vbar.grid(row = 3, column = 1, sticky = N+S)
        vbar.config(command=canvas.get_tk_widget().yview)


class Detail_Graph1(tk.Frame):

    def __init__(self, parent, controller, pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Detail Graph 1", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1)

        button2 = tk.Button(self, text="Go to Detail Graph 2",
                            command=lambda: controller.show_frame(Detail_Graph2))
        button2.grid(row = 2)
        
        global canvas1
        canvas1=FigureCanvasTkAgg(fig, self)
        canvas1.get_tk_widget().grid(row = 3, sticky = "nsew")

        toolbarFrame1 = tk.Frame(self)
        toolbarFrame1.grid(row=22, sticky=W+E+N+S)
        toolbar1 = NavigationToolbar2Tk(canvas1, toolbarFrame1)
        toolbar1.update()
        

class Detail_Graph2(tk.Frame):

    def __init__(self, parent, controller, pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Detail Graph 2", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1)

        button2 = tk.Button(self, text="Go to Detail Graph 3",
                            command=lambda: controller.show_frame(Detail_Graph3))
        button2.grid(row = 2)
        
        
        global canvas2
        canvas2=FigureCanvasTkAgg(figure_graph2, self)
        canvas2.get_tk_widget().grid(row = 3, sticky = "nsew")

        toolbarFrame2 = tk.Frame(self)
        toolbarFrame2.grid(row=22, sticky=W+E+N+S)
        toolbar2 = NavigationToolbar2Tk(canvas2, toolbarFrame2)
        toolbar2.update()
        

class Detail_Graph3(tk.Frame):

    def __init__(self, parent, controller, pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Detail Graph 3", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1)

        button2 = tk.Button(self, text="Go to Graph Page",
                            command=lambda: controller.show_frame(ThreeGraph))
        button2.grid(row = 2)
        
        
        global canvas3
        canvas3=FigureCanvasTkAgg(figure_graph3, self)
        canvas3.get_tk_widget().grid(row = 3, sticky = "nsew")

        toolbarFrame3 = tk.Frame(self)
        toolbarFrame3.grid(row=22, sticky=W+E+N+S)
        toolbar3 = NavigationToolbar2Tk(canvas3, toolbarFrame3)
        toolbar3.update()


class TablePage1(tk.Frame):

    def __init__(self, parent, controller, pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Previous Table for First Evolution", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1,)

        button2 = tk.Button(self, text="Go to Previous Table for Second Evolution",
                            command=lambda: controller.show_frame(TablePage2))
        button2.grid(row = 2,)  

        lst = []

        file_size = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv").st_size

        if file_size != 0:
            try:
                with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv", "r") as f:
                    reader = csv.reader(f, skipinitialspace=TRUE)
                    #skip the header
                    next(reader)
                    for row in reader:
                        lst = row
                        break
            except IOError:
                print("File not found!") 
 
        name=["Mean","Standard Deviation","Minimum Fitness","Minimum Individual","Maximum Fitness","Maximum Individual"]
        
        frame = Frame(self)
        frame.grid(row = 3, columnspan = 6)

        j=0
        for item in name:
            e = Entry(frame, width=30, font=('Arial',8,'bold'))
            e.grid(row = 0, column = j, sticky = E+W)    
            e.insert(END, name[j])
            j+=1

        i=0
        for item in lst:
            e = Entry(frame, width=30)
            e.grid(row = 1, column = i, sticky = E+W)    
            e.insert(END, lst[i])
            i+=1

        list1 = []

        file_size2 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv").st_size

        if file_size2 != 0:
            try:
                with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution1_previous.csv", "r") as f2:
                    reader2 = csv.reader(f2, skipinitialspace=TRUE)
                    #skip the header
                    next(reader2)
                    for row in reader2:
                        list1.append(row)
            except IOError:
                print("File not found!")

        self.tree = ttk.Treeview(self, columns = (1,2,3,4,5,6,7), height = 25, show = "headings")
        self.tree.grid(row = 5, columnspan = 6)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(5, weight=1)
        self.tree.heading(1, text="Generation")
        self.tree.heading(2, text="Average Fitness")
        self.tree.heading(3, text="Std")
        self.tree.heading(4, text="Min Fitness")
        self.tree.heading(5, text="Min Individual")
        self.tree.heading(6, text="Max Fitness")
        self.tree.heading(7, text="Max Individual")

        self.tree.column(1, width = 160)
        self.tree.column(2, width = 160)
        self.tree.column(3, width = 160)
        self.tree.column(4, width = 160)
        self.tree.column(5, width = 160)
        self.tree.column(6, width = 160)
        self.tree.column(7, width = 160)
        
        scroll = ttk.Scrollbar(self, orient="vertical", command=self.tree.yview)
        scroll.grid(row = 5, column = 1, sticky = N+S)

        self.tree.configure(yscrollcommand=scroll.set)

        for val in list1:
            self.tree.insert('', 'end', values = (val[0], val[1], val[2], val[3], val[4], val[5], val[6]) )
        

class TablePage2(tk.Frame):

    def __init__(self, parent, controller, pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Previous Table for Second Evolution", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1)

        button2 = tk.Button(self, text="Go to Previous Table for Third Evolution",
                            command=lambda: controller.show_frame(TablePage3))
        button2.grid(row = 2)

        lst = []

        file_size = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv").st_size

        if file_size != 0:
            try:
                with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv", "r") as f:
                    reader = csv.reader(f, skipinitialspace=TRUE)
                    #skip the header
                    next(reader)
                    next(reader)
                    for row in reader:
                        lst = row
                        break
            except IOError:
                print("File not found!") 

        name=["Mean","Standard Deviation","Minimum Fitness","Minimum Individual","Maximum Fitness","Maximum Individual"]
        
        frame = Frame(self)
        frame.grid(row = 3, columnspan = 6)

        j=0
        for item in name:
            e = Entry(frame, width=30, font=('Arial',8,'bold'))
            e.grid(row = 0, column = j, sticky = E+W)    
            e.insert(END, name[j])
            j+=1

        i=0
        for item in lst:
            e = Entry(frame, width=30)
            e.grid(row = 1, column = i, sticky = E+W)    
            e.insert(END, lst[i])
            i+=1

        list1 = []

        file_size2 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv").st_size

        if file_size2 != 0:
            try:
                with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution2_previous.csv", "r") as f2:
                    reader2 = csv.reader(f2, skipinitialspace=TRUE)
                    #skip the header
                    next(reader2)
                    next(reader2)
                    for row in reader2:
                        list1.append(row)
            except IOError:
                print("File not found!")

        tree = ttk.Treeview(self, columns = (1,2,3,4,5,6,7), height = 25, show = "headings")
        tree.grid(row = 5, columnspan = 6)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(5, weight=1)
        tree.heading(1, text="Generation")
        tree.heading(2, text="Average Fitness")
        tree.heading(3, text="Std")
        tree.heading(4, text="Min Fitness")
        tree.heading(5, text="Min Individual")
        tree.heading(6, text="Max Fitness")
        tree.heading(7, text="Max Individual")

        tree.column(1, width = 160)
        tree.column(2, width = 160)
        tree.column(3, width = 160)
        tree.column(4, width = 160)
        tree.column(5, width = 160)
        tree.column(6, width = 160)
        tree.column(7, width = 160)
        
        scroll = ttk.Scrollbar(self, orient="vertical", command=tree.yview)
        scroll.grid(row = 5, column = 1, sticky = N+S)

        tree.configure(yscrollcommand=scroll.set)

        for val in list1:
            tree.insert('', 'end', values = (val[0], val[1], val[2], val[3], val[4], val[5], val[6]) )


class TablePage3(tk.Frame):

    def __init__(self, parent, controller, pages):
        tk.Frame.__init__(self, parent)
   
        label = tk.Label(self, text="Previous Table for Third Evolution", font=LARGE_FONT)
        label.grid(row = 0, pady = 10, padx = 10)

        button1 = tk.Button(self, text="Back to Home",
                            command=lambda: controller.show_frame(StartPage))
        button1.grid(row = 1)

        lst = []

        file_size = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv").st_size

        if file_size != 0:
            try:
                with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/ALL_previous.csv", "r") as f:
                    reader = csv.reader(f, skipinitialspace=TRUE)
                    #skip the header, till third evolution cycle
                    next(reader)
                    next(reader)
                    next(reader)
                    for row in reader:
                        lst = row
                        break
            except IOError:
                print("File not found!") 


        name=["Mean","Standard Deviation","Minimum Fitness","Minimum Individual","Maximum Fitness","Maximum Individual"]
        
        frame = Frame(self)
        frame.grid(row = 3, columnspan = 6)

        j=0
        for item in name:
            e = Entry(frame, width=30, font=('Arial',8,'bold'))
            e.grid(row = 0, column = j, sticky = E+W)    
            e.insert(END, name[j])
            j+=1

        i=0
        for item in lst:
            e = Entry(frame, width=30)
            e.grid(row = 1, column = i, sticky = E+W)    
            e.insert(END, lst[i])
            i+=1

        list1 = []
        
        file_size2 = os.stat(r"/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv").st_size

        if file_size2 != 0:
            try:
                with open("/Users/zrhun/FYP Topic - Data Science (Analysis M'sia Finance)/FYPsystem/previous/evolution3_previous.csv", "r") as f2:
                    reader2 = csv.reader(f2, skipinitialspace=TRUE)
                    #skip the header
                    next(reader2)
                    next(reader2)
                    next(reader2)
                    for row in reader2:
                        list1.append(row)
            except IOError:
                print("File not found!")

        tree = ttk.Treeview(self, columns = (1,2,3,4,5,6,7), height = 25, show = "headings")
        tree.grid(row = 5, columnspan = 6)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(3, weight=1)
        self.grid_rowconfigure(5, weight=1)
        tree.heading(1, text="Generation")
        tree.heading(2, text="Average Fitness")
        tree.heading(3, text="Std")
        tree.heading(4, text="Min Fitness")
        tree.heading(5, text="Min Individual")
        tree.heading(6, text="Max Fitness")
        tree.heading(7, text="Max Individual")

        tree.column(1, width = 160)
        tree.column(2, width = 160)
        tree.column(3, width = 160)
        tree.column(4, width = 160)
        tree.column(5, width = 160)
        tree.column(6, width = 160)
        tree.column(7, width = 160)
        
        scroll = ttk.Scrollbar(self, orient="vertical", command=tree.yview)
        scroll.grid(row = 5, column = 1, sticky = N+S)

        tree.configure(yscrollcommand=scroll.set)

        for val in list1:
            tree.insert('', 'end', values = (val[0], val[1], val[2], val[3], val[4], val[5], val[6]) )

    

####Tkinter Application
app = FinancialDataSeries()

# Gets the requested values of the height and widht.
windowWidth = app.winfo_reqwidth()
windowHeight = app.winfo_reqheight()
 
# Gets both half the screen width/height and window width/height
positionRight = int(app.winfo_screenwidth()/5 - windowWidth/2)
positionDown = int(app.winfo_screenheight()/6 - windowHeight/2)
 
# Positions the window in the center of the page.
app.geometry("+{}+{}".format(positionRight, positionDown))

ani = animation.FuncAnimation(f, animate, interval=3000)
ani1 = animation.FuncAnimation(fig, animate_graph1, interval=3000)
ani2 = animation.FuncAnimation(figure_graph2, animate_graph2, interval=3000)
ani3 = animation.FuncAnimation(figure_graph3, animate_graph3, interval=3000)
app.mainloop()