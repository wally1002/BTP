from re import T
from types import DynamicClassAttribute
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

#np.seterr(divide='ignore', invalid='ignore')
random.seed(1)
np.random.seed(1)

class Rectangle:

    def __init__(self, x, y, n):
        self.x = x
        self.y = y
        self.length = x
        self.width = y
        self.m = 48 
        self.numIter = 2000
        self.n = n
        self.d = self._generate_distance_matrix()
        self.d = np.around(self.d, decimals=2)
        np.savetxt("distance_matrix.txt", self.d, fmt='%1.2f')

    def plot(self):
        plt.plot(self.x, self.y, 'ro')
        plt.show()

    def random_point(self):
        # return a random point inside the rectangle
        x = np.random.uniform(0, self.x)
        y = np.random.uniform(0, self.y)
        return x, y
    
    def generate_points(self, n):
        """
            Generates n Random points in a given rectangle. 
        """
        tx = []
        ty = []
        for i in range(n):
            a, b = self.random_point()
            tx.append(a)
            ty.append(b)
        self.tx = np.array(tx)
        self.ty = np.array(ty)
        return tx, ty

    def plot_random_points(self):
        tx, ty = self.generate_points(10)
        plt.plot(tx, ty, 'ro')
        plt.show()

    
    def _generate_distance_matrix(self):
        """
            Generates a distance matrix of n points in a given rectangle. 
        """
        tx, ty = self.generate_points(self.n)
        d = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                d[i][j] = np.sqrt((tx[i] - tx[j])**2 + (ty[i] - ty[j])**2)
        return d

    def generate_distance_matrix(self, n):
        """
            Generates a distance matrix of n points in a given rectangle. 
        """
        tx, ty = self.generate_points(n)
        d = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                d[i][j] = np.sqrt((tx[i] - tx[j])**2 + (ty[i] - ty[j])**2)
        return d    
    
    def plot_distance_matrix(self, n):
        d = self.generate_distance_matrix(n)
        G = nx.from_numpy_matrix(np.matrix(d), create_using=nx.DiGraph)
        layout = nx.spring_layout(G)
        labels = nx.get_edge_attributes(G, "weight")
        nx.draw_networkx_edges(G, pos=layout)
        nx.draw_networkx_nodes(G,pos=layout)
        plt.show()

    def calcDistance(self, cities, startIdx=0):
        #given the array of cities in order, this function calculates the total distance for rountrip
        #from startIdx
        dist=0
        k=len(cities)
        for i in range(startIdx,startIdx+k):
            dist+= self.d[cities[i%k]][cities[(i+1)%k]]
        return -dist

    def generateYt(self, S):
        #Generates yt (or cost array) of S
        return np.array([self.calcDistance(S[i]) for i in range(len(S))])

    def generateUt(self, y):
        #generates non negative scores of yt
        return y-min(y)

    def generatePt(self, u):
        #generated fitness scores of ut
        sum_ut=sum(u)
        return u/sum_ut

    def generateChild(self, parent):
        #Generates child by randomly swapping two indices
        idx = range(len(parent))
        i1, i2 = random.sample(idx, 2)
        child=np.copy(parent)
        child[i1],child[i2] = child[i2],child[i1]
        return child

    def choose_best_m(self, s1,s2,y1,y2,m):
        #chooses m values with least cost from s1 and s2
        #(or) best of s1 U s2
        topList=[]
        #each element of topList is stored in format [cost, position, index]
        #if the cost comes from y1, then position is 1, if y2, position is 2
        for i in range(len(y1)):
            topList.append([y1[i], 1, i])
        for i in range(len(y2)):
            topList.append([y2[i], 2, i])
        
        topList.sort(reverse=True) #we now sort topList by 0th element
        s=[] #our final list containing top m paths
        y=[] #cost array of s
        for i in range(m):
            cost,pos,idx=topList[i] #pos is required to know whether the path comes from s1 or s2
            y.append(cost)
            if pos==1:
                s.append(list(s1[idx]))
            if pos==2:
                s.append(list(s2[idx]))
        return np.array(s),np.array(y)
        

    def init_tsp(self):
        #Generating S0 and y0 randomly (random initialization)
        S0=[]
        y0=[]
        self.cities = [i for i in range(len(self.d))]
        for i in range(self.m):
            np.random.shuffle(self.cities)
            y0.append(self.calcDistance(self.cities))
            S0.append(list(self.cities))
        S0=np.array(S0)
        y0=np.array(y0)
        return S0, y0

    def solve_tsp(self):
        S, y = self.init_tsp()
        y_opt=y[0] #we will check for optimum solution at each and every step
        path_opt=S[0]
        for i in range(self.numIter):
            u=self.generateUt(y) #generate as u_t
            p=self.generatePt(u) #generate as p_t
            p=np.around(p*self.m) #rounding off p x m
            p=np.int64(p) #converting float to int for indexing purposes
            S_new=[] #same as S_t+1
            #generating children by fitness score and storing in S_new
            #we generate K children for S[i] if p[i]=K
            for j in range(len(S)):
                for k in range(p[j]):
                    S_new.append(list(self.generateChild(S[j])))
                    
            S_new=np.array(S_new) #converting to numpy array for easier operations
            y_new=self.generateYt(S_new)
            S,y=self.choose_best_m(S,S_new,y,y_new,self.m) #we choose best m out of S and S_new
            
            #checking for optimum solution
            for i in range(self.m):
                if y[i]>y_opt:
                    y_opt=y[i]
                    path_opt=S[i]
        y_opt=-y_opt 
        print(y_opt)
        print("\n")
        print(path_opt)
        return path_opt

    def plot_tsp(self):
        path = self.solve_tsp() 
        px = []
        py = []
        for i in path:
            px.append(self.tx[i])
            py.append(self.ty[i])
        plt.plot(px[0], py[0], 'ro')
        plt.plot(px[-1], py[-1], 'go')
        plt.plot(px, py, )
        plt.show()



    def divide(self, agents=4):
        # divide an area into n rectangles given two corners
        """
        p - corners of rectangle 
        n - no of agents
        """
        # calculate factors of a number
        factors = []
        print(agents)
        min_diff = 10000000
        for i in range(1, agents//2+1):
            if agents % i == 0:
                factors.append([i, agents//i])
                if abs(i - agents//i) < min_diff:
                    min_diff = abs(i - agents//i)
                    min_factors = [i, agents//i]
        val = 0
        if(self.length >= self.width):
            val = 0
        else:
            val = 1
        min_factors.sort(reverse=True)
        print(min_factors)
        # divide the area into n rectangles
        dx = self.length/min_factors[val]
        dy = self.width/min_factors[1-val]
        print(dx, dy)
        rectangles = []
        x_co = np.linspace(0, self.x, min_factors[val]+1, endpoint=True)
        y_co = np.linspace(0, self.y, min_factors[1-val]+1, endpoint=True)
        print(x_co, y_co)
        rectangles = []
        for x in x_co.tolist()[:-1]:
            for y in y_co.tolist()[:-1]:
                x1 = x
                x2 = x + dx
                y1 = y
                y2 = y + dy
                rectangles.append([(x1, y1), (x1, y2), (x2, y2), (x2, y1)])
        print(rectangles)
        rectangles = np.array(rectangles)
        for i in rectangles:
            plt.plot(i.T[0], i.T[1], 'ro')
        plt.show()
        return rectangles

if __name__ == '__main__':
    fence = Rectangle(10, 10, 30)
    divide_area = fence.divide(agents=10)
    print(divide_area.shape)
    fence.plot_tsp()
