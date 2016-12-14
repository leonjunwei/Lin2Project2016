import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as anim

from pickle import dump, load

from Cell2D import Cell2D, Cell2DViewer


###### Problem Spaces ######


def generatePairs(x,y):
    for i in xrange(x):
        for j in xrange(y):
            yield i,j

class gradientArray(object):
    def __init__(self, array=None):
        self.array = array

class ProblemSpace(object):
    def __init__(self,minX, maxX, minY, maxY, numX, numY, resolution):
        """Generates a scalar field bounded by (minX,maxX), (minY,maxY) with numX, numY functions."""
        self.funcList = {}
        for i in range(10):
            self.funcList[i] = (lambda x,k=i: (k+1)**2*x*np.sin((k+1)*x),'sin', i)
            self.funcList[i+10] = (lambda x,k=i: (k+1)**2*x*np.cos(k+1*x),'cos', i)
        self.minX, self.maxX = minX, maxX
        self.minY, self.maxY = minY, maxY
        self.xInterval = self.maxX-self.minX
        self.yInterval = self.maxY-self.minY
        # self.xFuncs = random.sample(self.funcList,numX)
        # self.yFuncs = random.sample(self.funcList,numY)
        self.xFuncs = [8]
        self.yFuncs = [18]
        self.agents = set()
        self.array = None
        self.resolution = resolution
    def getValue(self,x,y):
        # returns value of scalar field at position x,y
        return sum([self.funcList[k][0](x) for k in self.xFuncs] + [self.funcList[h][0](y) for h in self.yFuncs])
        # return (x**2+y**2)

    def differentiate(self,funcTuple):
        """Takes in a (lambda function, sin/cos, multiplier) tuple and spits out a derivative function"""
        j = funcTuple[2] #multiplier
        if funcTuple[1] == 'sin':
            return lambda x,k=j: (k+1)**2*(np.sin((k+1)*x) + (k+1)*x*np.cos((k+1)*x))
        else: #if funcTuple[1] == 'cos':
            return lambda x,k=j: (k+1)**2*(np.cos((k+1)*x) - (k+1)*x*np.sin((k+1)*x))

    def getGradient(self,x,y):
        # returns an tuple of the scalar field's gradient at position x,y
        return sum([self.differentiate(self.funcList[k])(x) for k in self.xFuncs]), sum([self.differentiate(self.funcList[k])(y) for k in self.yFuncs])
        # return (2*x,2*y)


    def simulate(self, steps):
        for i in range(steps):
            self.step()

    def step(self):
        for agent in self.agents:
            agent.step()

    def generateValues(self):
        self.array = np.zeros((self.resolution-1,self.resolution-1))
        for x,y in generatePairs(self.resolution-1,self.resolution-1):
            self.array[x,y] = self.getValue(self.minX+(self.xInterval)*x*1.0/(self.resolution-1),self.minY+(self.yInterval)*y*1.0/(self.resolution-1))

    def generateGradient(self):
        resultX = np.zeros((self.resolution-1,self.resolution-1))  
        resultY =  np.zeros((self.resolution-1,self.resolution-1))     
        for x,y in generatePairs(self.resolution-1,self.resolution-1):
            resultX[x,y] = self.getGradient(self.minX+(self.xInterval)*x*1.0/(self.resolution-1),self.minY+(self.yInterval)*y*1.0/(self.resolution-1))[0]
            resultY[x,y] = self.getGradient(self.minX+(self.xInterval)*x*1.0/(self.resolution-1),self.minY+(self.yInterval)*y*1.0/(self.resolution-1))[1]
        return resultX, resultY

class SimulatedAnnealingSpace(ProblemSpace):
    def __init__(self,minX, maxX, minY, maxY, numX, numY, resolution, tempFactor):
        ProblemSpace.__init__(self,minX, maxX, minY, maxY, numX, numY, resolution)
        self.temp = 1
        self.tempFactor = tempFactor
        self.minTemp = 0.000001
    def step(self):
        for agent in self.agents:
            agent.step(self.temp)
            if self.temp>self.minTemp:
                self.temp = self.temp * self.tempFactor

class BeeSwarm(ProblemSpace):
    def __init__(self,minX, maxX, minY, maxY, numX, numY, resolution, k):
        ProblemSpace.__init__(self,minX, maxX, minY, maxY, numX, numY, resolution)
        self.numBees = 10
        self.numScouts = 3
        self.bees = []
        self.count = 10
        self.k = k #the learning factor of the gradient descent bees
        self.scouts = []
        self.makeScouts(self.numScouts)

    def makeScouts(self,num):
        #populates self.scouts with num scouts in random locations between minX,maxX and minY, maxY 
        for i in range(num):
            x, y = random.uniform(self.minX,self.maxX), random.uniform(self.minY,self.maxY)
            scout = GradientDescent(x,y,self,self.k)
            self.scouts.append(scout)

    def makeBees(self,number, x, y):
        #returns a list of 'number' bees randomly distributed around (x,y)
        result = []
        for i in range(number):
            newX, newY = random.uniform(self.xInterval*0.04,self.xInterval*-0.04), random.uniform(self.yInterval*0.04,self.yInterval*-0.04)
            bee = GradientDescent(x+newX,y+newY,self,self.k)
            result.append(bee)
        return result

    def releaseBees(self):
        sortedScoutList = sorted([(scout.getValue,scout) for scout in self.scouts])
        for i in range(len(sortedScoutList)):
            if i == 0:
                sortedScoutList[i][1].family = self.makeBees(6, sortedScoutList[i][1].x, sortedScoutList[i][1].y)
            if i == 1:
                sortedScoutList[i][1].family = self.makeBees(3, sortedScoutList[i][1].x, sortedScoutList[i][1].y)
            if i == 2:
                sortedScoutList[i][1].family = self.makeBees(1, sortedScoutList[i][1].x, sortedScoutList[i][1].y)

    def decideScouts(self):
        #each scout swaps places with the best bee in its family
        for scout in self.scouts:
            scoutValue = scout.getValue()
            if scout.family:
                for bee in scout.family:
                    if bee.getValue() < scoutValue:
                        scout.x,scout.y = bee.x,bee.y


    def step(self):
        if self.count < 10:
            for scout in self.scouts:
                scout.step()
                if scout.family:
                    for bee in scout.family:
                        bee.step()

            self.count +=1
        elif self.count == 10:
            self.decideScouts()
            self.releaseBees()
            self.count = 0








###### Agents ######


def rescale(value, inputMin, inputMax, outputMin, outputMax):
    return outputMin + float(value-inputMin)*(outputMax-outputMin)/(inputMax-inputMin)

def distance(tup1,tup2):
    return np.sqrt((tup1[0]-tup2[0])**2 + (tup1[1]-tup2[1])**2)

class Agent(object):
    def __init__(self,x,y,env):
        self.x = x
        self.y = y
        self.env = env
        self.history = []
    def getValue(self): #For convenience's sake. Adds a little overhead though.
        return self.env.getValue(self.x,self.y)
    def getGradient(self):
        return self.env.getGradient(self.x,self.y)
    def pos(self):
        return self.x,self.y
    def plot(self,axis):
        # xScale = float(self.env.resolution-1)/(self.env.xInterval) #if resolution-1 = 100 and interval = 4 (-2 to 2), 0,0 should correspond to 50,50 still. 1,1 should correspond to 75,75.
        # yScale = float(self.env.resolution-1)/(self.env.yInterval)
        x = [rescale(m[0],self.env.minX,self.env.maxX,0,self.env.resolution-1) for m in self.history]
        y = [rescale(n[1],self.env.minY,self.env.maxY,0,self.env.resolution-1) for n in self.history]
        axis.plot(x,y)
    def largestStep(self):
        store = [distance(self.history[i],self.history[i+1]) for i in range(len(self.history)-1)]
        return max(store)
    def checkBoundary(self):
        if self.x > self.env.maxX:
            self.x = self.env.maxX
        if self.x < self.env.minX:
            self.x = self.env.minX
        if self.y > self.env.maxY:
            self.y = self.env.maxY
        if self.y < self.env.minY:
            self.y = self.env.minY



class GradientDescent(Agent):
    def __init__(self,x,y,env,k):
        #k for GradientDescent is the factor by which we multiply gradient.
        Agent.__init__(self,x,y,env)
        self.k = k
        self.family = []
    def step(self):
        self.history.append((self.x,self.y))
        xMove = -1*self.getGradient()[0]*self.k
        yMove = -1*self.getGradient()[1]*self.k
        if np.fabs(xMove) > self.env.xInterval/100.0:
            xMove = np.sign(xMove)*self.env.xInterval/100.0 #we limit speed so the agent doesn't move too fast.
        if np.fabs(yMove) > self.env.yInterval/100.0:
            yMove = np.sign(yMove)*self.env.yInterval/100.0
        self.x += xMove
        self.y += yMove
        self.checkBoundary()

class SimulatedAnnealing(Agent):
    def __init__(self,x,y,env,k):
        Agent.__init__(self,x,y,env)
        self.k = k
    def checkValue(self,i,j):
        return self.env.getValue(i,j)
    def step(self,temp):
        self.history.append((self.x,self.y))
        moveX, moveY = random.uniform(self.env.xInterval*0.2*temp,self.env.xInterval*-0.2*temp), random.uniform(self.env.yInterval*0.2*temp,self.env.yInterval*-0.2*temp)
        newX,newY = self.x+moveX, self.y+moveY
        if self.checkValue(newX,newY) < self.getValue():
            self.x = newX
            self.y = newY
            self.checkBoundary()           
        else:
            # probability_required = ((self.checkValue(newX,newY)-float(self.getValue()))/self.getValue())**2 * temp
            probability_required = np.exp((float(self.getValue())-self.checkValue(newX,newY))/temp)
            if random.random()<probability_required:
                self.x = newX
                self.y = newY
                self.checkBoundary()   










def runExperiment1(size, resolution, numXFuncs, numYFuncs, steps, numAgents, experimentType, k):
    #k is sort of a flex value for stuff. 
        #In gradient descent it's the learning factor by which we multiply the gradient. k = 0.007 is not bad
        #In simulated annealing it's the factor by which the global temperature decreases. k = 0.98 is not bad
    agentDict = {"gradient_descent": lambda x,y,env,k:GradientDescent(x,y,env,k), "simulated_annealing":lambda x,y,env,k:SimulatedAnnealing(x,y,env,k)}
    problemDict = {"gradient_descent": lambda a,b,c,d,e: ProblemSpace(-1*a,a,-1*a,a,b,c, resolution = d), "simulated_annealing": lambda a,b,c,d, e: SimulatedAnnealingSpace(-1*a,a,-1*a,a,b,c, resolution = d, tempFactor = e)}
    a = problemDict[experimentType](size,numXFuncs,numYFuncs,resolution,k)
    for i in range(numAgents):
        c = agentDict[experimentType](random.uniform(-1*size, size), random.uniform(-1*size, size), a, k)
        a.agents.add(c)
   

    a.generateValues()
    viewValues = Cell2DViewer(a)
    b = a.generateGradient()
    fig = plt.figure()


    def update(i):
    # for i in range(steps):
        a.step()
        ax1=fig.add_subplot(1,2,1)
        ax1.clear()
        for c in a.agents:
            c.plot(ax1)
        viewValues.draw_array()

        ax2=fig.add_subplot(1,2,2)
        ax2.clear()
        for c in a.agents:
            c.plot(ax2)

        plt.quiver(b[0],b[1])

        # plt.show()
        # plt.pause(0.1)
    animation = anim.FuncAnimation(fig, update, frames=steps, repeat=False)
    plt.show()
    for c in a.agents:
        print str([a.getValue(*k) for k in c.history]) + "length: %d" %(len(c.history)) + "\n"
        # print str(c.history) + "length: %d" %(len(c.history)) + "\n"

runExperiment1(size=10,resolution=101,numXFuncs=1,numYFuncs=1,steps=500,numAgents=1,experimentType="simulated_annealing",k=0.98)


def runExperimentBees(size, resolution, numXFuncs, numYFuncs, steps, k):
    a = BeeSwarm(-size,size,-size,size,numXFuncs,numYFuncs,resolution,k)
    a.generateValues()
    viewValues = Cell2DViewer(a)
    b = a.generateGradient()
    fig = plt.figure()
    def update(i):
        a.step()
        ax1=fig.add_subplot(1,2,1)
        ax1.clear()
        for scout in a.scouts:
            scout.plot(ax1)
            if scout.family:
                for bee in scout.family:
                    bee.plot(ax1)
        viewValues.draw_array()

        ax2=fig.add_subplot(1,2,2)
        ax2.clear()
        for scout in a.scouts:
            scout.plot(ax2)
            if scout.family:
                for bee in scout.family:
                    bee.plot(ax2)
        plt.quiver(b[0],b[1])
    animation = anim.FuncAnimation(fig, update, frames=steps, repeat=False)
    plt.show()
    for c in a.scouts:
        print str([a.getValue(*k) for k in c.history]) + "length: %d" %(len(c.history)) + "\n"

runExperimentBees(size=10,resolution=101,numXFuncs=1,numYFuncs=1,steps=500,k=0.007)


# (self,minX, maxX, minY, maxY, numX, numY, resolution, k):

# a = ProblemSpace(-10,10,-10,10,1,1,101)
# print a.getValue(3.32719597083727, -1.6281008963831702)

### Unused Code ###


