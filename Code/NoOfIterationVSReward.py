#!/usr/bin/env python
# coding: utf-8
# @author Avijit Roy


# In[ ]:

# Importing the required module
import pandas as pd
import matplotlib.image as image
import matplotlib.cbook as cbook
from matplotlib import style
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import random
import pickle
import time
import os
import math
import numpy as np
import datetime
from datetime import timedelta
import glob
from PIL import Image

from pymongo import MongoClient
import json
from pprint import pprint
from bson.objectid import ObjectId



from Utilities import Utilities as ut
import matplotlib
# matplotlib.use("Agg")


# In[ ]:

class GridWorld:

    def __init__(self, epsilon=0.2, alpha=0.3, gamma=0.9):
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.Q = {}
        self.last_grid = None
        self.q_last = 0.0
        self.state_action_last = None

    def setEpsilon(self, epsilon):
        self.epsilon = epsilon

    def game_begin(self):
        self.last_grid = None
        self.q_last = 0.0
        self.state_action_last = None

    def epsilon_greedy(self, state, possible_moves):
        self.last_grid = tuple(state)
        if(random.random() < self.epsilon):
            move = random.choice(possible_moves)
            self.state_action_last = (self.last_grid, move)
            return move
        else:
            Q_list = []
            for action in possible_moves:
                Q_list.append(self.getQ(self.last_grid, action))
            maxQ = max(Q_list)

            if Q_list.count(maxQ) > 1:
                best_options = [i for i in range(
                    len(possible_moves)) if Q_list[i] == maxQ]
                i = random.choice(best_options)
            else:
                i = Q_list.index(maxQ)
            self.state_action_last = (self.last_grid, possible_moves[i])
            self.q_last = self.getQ(self.last_grid, possible_moves[i])
            return possible_moves[i]

    def getQ(self, state, action):
        if(self.Q.get((state, action))) is None:
            self.Q[(state, action)] = 1.0
        return self.Q.get((state, action))

    def updateQ(self, reward, state, possible_moves):
        q_list = []
        for moves in possible_moves:
            q_list.append(self.getQ(tuple(state), moves))
        if q_list:
            max_q_next = max(q_list)
        else:
            max_q_next = 0.0
        self.Q[self.state_action_last] = self.q_last + self.alpha * \
            ((reward + self.gamma*max_q_next) - self.q_last)

    def saveQtable(self, file_name):
        with open(file_name, 'wb') as handle:
            pickle.dump(self.Q, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def loadQtable(self, file_name):
        with open(file_name, 'rb') as handle:
            self.Q = pickle.load(handle)


# In[ ]:


class RunAgents:
    # width and height are the width and height of the gridworld
    def __init__(self, zone, crimes, longitude=10, latitude=10, num_agents=2, training=False, beta=-5, beta_3 = -1):
        self.width = longitude
        self.height = latitude
        self.num_agents = num_agents
        self.beta = beta
        self.zone = zone;
        self.beta_3 = beta_3;
        self.total_reward = 0;

        self.grid = -1 * np.ones(shape=(self.height, self.width), dtype=int)

        self.done = False
        self.agents = np.empty([self.num_agents], dtype=GridWorld)
        #print(crimes.head())
        (self.all_reward_states, self.hist_lat, self.hist_long)  = ut.lat_long_to_grid(crimes["lat"], crimes["lng"], self.width, self.height);
        self.reward_states = {}
        self.max_iter = 500



        x = datetime.datetime.now()
        now = str(x)[0:10]
        self.now = now;
        self.time = datetime.datetime.strptime(now, '%Y-%m-%d')

        self.visited_states = []
        self.k_coverage = 50 * self.num_agents  # last k steps to calculate coverage
        self.all_states = self.createAllPossibleIndex(self.height, self.width)
        self.reward_frequncy = 0.005 * self.num_agents
        self.reward_parameter = 11


        self.route = {}



    def createAllPossibleIndex(self, x, y):     
        result = []
        for i in range(x):
            for j in range(y):
                result.append([i, j])
        return result



    def reset(self):
        if(self.training):
            self.grid = -1 * \
                np.ones(shape=(self.height, self.width), dtype=int)
            self.reward_states = {}
            self.total_reward = 0;
            indices = random.sample(self.all_states, self.num_agents)
            for i in range(self.num_agents):
                x = indices[i][0]
                y = indices[i][1]
                self.grid[x][y] = i
                self.visited_states.append([x, y])
            return



    def evaluate(self, ch):
        proximity_reward = 0
        location = self.find_location(ch)
        for x in range(self.num_agents):
            distance = self.proximity(ch, x)
            # - 1 for own distance exp(0)
            proximity_reward += (math.exp(self.beta * distance) - 1)

        instant_reward = 0
        for y in self.reward_states.keys():
            distance = self.find_reward_state_distance(np.array(y), ch)
            if y[0] == location[0] and y[1] == location[1]:
                # * math.exp(-self.beta * distance)
                instant_reward += 50 * self.reward_states[y]
            # print("Distance = %2.2f"%(distance))
            # print(self.reward_states[y]  * math.exp(-self.beta * distance) )

        # print("Proximity Reward = %2.2f Instant Reward = %2.2f"%(proximity_reward,instant_reward))
        return proximity_reward + instant_reward, False





    def find_location(self, ch):
        temp = np.where(self.grid == ch)
        location = [temp[0][0], temp[1][0]]

        return np.array(location)

    def proximity(self, ch1, ch2):
        location_ch_1 = self.find_location(ch1)
        location_ch_2 = self.find_location(ch2)

        return np.linalg.norm(location_ch_1 - location_ch_2)

    def find_reward_state_distance(self, location_ch_1, ch2):
        location_ch_2 = self.find_location(ch2)

        return np.linalg.norm(location_ch_1 - location_ch_2)

    def possible_moves(self, ch):
        location = self.find_location(ch)
        remove = None
        return_value = ['d', 'l', 'u', 'r', 's']
        if (location[0] == (self.height - 1)) or (self.grid[location[0]+1][location[1]] != -1):
            return_value.remove('d')
        if (location[1] == 0) or (self.grid[location[0]][location[1]-1] != -1):
            return_value.remove('l')
        if (location[0] == 0) or (self.grid[location[0]-1][location[1]] != -1):
            return_value.remove('u')
        if (location[1] == (self.width - 1)) or (self.grid[location[0]][location[1]+1] != -1):
            return_value.remove('r')
        return return_value

    def step(self, agent, move):
        location = self.find_location(agent)
        new_location_x = location[0]
        new_location_y = location[1]

        step_penalty = 0;

        if move != 's':
            self.grid[location[0]][location[1]] = -1
            step_penalty = self.beta_3;
        if move == 'd':
            new_location_x += 1
        if move == 'l':
            new_location_y -= 1
        if move == 'u':
            new_location_x -= 1
        if move == 'r':
            new_location_y += 1

        self.grid[new_location_x][new_location_y] = agent
        self.visited_states.append([new_location_x, new_location_y])
        reward, done = self.evaluate(agent)

        return reward + step_penalty, done

    def startTraining(self, agents):
        self.training = True
        for i in range(self.num_agents):
            if(isinstance(agents[i], GridWorld)):
                self.agents[i] = agents[i]


    def train(self, iterations):
        if(self.training):
            for i in range(iterations):
                if ((i % iterations) == (iterations - 1)):
                    for agnt in self.agents:
                        agnt.setEpsilon(0.0);

                print("Iteration %d/%d" % (i+1, iterations))
                for j in range(self.num_agents):
                    self.agents[j].game_begin()

                self.reset()
                done = False
                agent = 0
                episode_length = 0
                timeSlot = self.time - timedelta(hours=1) + timedelta(days=1) - timedelta(hours=5) - timedelta(minutes=30) #for indian timezone -5:30 is added
                agentLocation = [[] for i in range(self.num_agents)]
                while (not done) and episode_length < self.max_iter:
                    episode_length += 1
                    if(random.random() < self.reward_frequncy):
                        temp_reward_state = random.sample(
                            self.all_reward_states, 1)[0]
                        self.reward_states[tuple(
                            temp_reward_state)] = self.reward_parameter

                    move = self.agents[agent].epsilon_greedy(
                        self.find_location(agent), self.possible_moves(agent))
                    reward, done = self.step(agent, move)
                    self.total_reward += reward;
                    self.agents[agent].updateQ(reward, self.find_location(
                        agent), self.possible_moves(agent))
                    agent = (agent + 1) % self.num_agents
                    #if ((i % iterations) == (iterations - 1)):
                        #print(episode_length)
                        #self.showGrid(self.grid, episode_length, reward)
                    # time.sleep(1)
                    '''print(self.find_location(agent))
                        print(self.hist_lat)
                        print(self.hist_long)'''
                    if ((i % iterations) == (iterations - 1)):
                        if episode_length > (self.max_iter - self.num_agents * 24):
                            currentLoc = (self.find_location(agent))
                            longitude = random.uniform(
                                self.hist_long[currentLoc[1]], self.hist_long[currentLoc[1]+1])
                            latitude = random.uniform(
                                self.hist_lat[currentLoc[0]], self.hist_lat[currentLoc[0]+1])
                            agentLocation[agent].append(
                                [agent, timeSlot, latitude, longitude])
                            if((episode_length - (self.max_iter - self.num_agents * 24))% self.num_agents == 1):
                                timeSlot = timeSlot + timedelta(hours=1)
                        

                    for reward_state in self.reward_states.keys():
                        self.reward_states[reward_state] -= (1/self.num_agents)

                    delete = [
                        key for key in self.reward_states if self.reward_states[key] <= 0]

                    for key in delete:
                        del self.reward_states[key]

                    # print(self.reward_states)

            frames = []
            for agentStep in range(self.num_agents):
                df = pd.DataFrame(agentLocation[agentStep], columns=[
                    "AgentId", "TimeSlot", "Latitude", "Longitude"])
                frames.append(df)
            df = pd.concat(frames)
            #print(df.head())
            df.to_excel("../Results/trajectory_" + str(self.beta_3) + "_" + self.zone +".xlsx")
            coverage = self.calculate_coverage(self.k_coverage)
            return self.total_reward

    def showGrid(self, grid_main, iteration, reward):
        grid = grid_main.copy()

        fig, ax = plt.subplots()
        '''img = plt.imread("../Figure/Ayodha.png")
        ax.imshow(img, origin='lower',
                  extent=[-1, self.width, -1, self.height])'''
        ax.title.set_text("Reward: %4.4f" % (reward))
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        self.mat = ax.matshow(grid, alpha=0.3, origin='lower')
        plt.colorbar(self.mat)
        for (i, j), z in np.ndenumerate(grid):
            for reward_state in self.reward_states.keys():
                if reward_state[0] == i and reward_state[1] == j:
                    ax.text(j, i, ' ', ha='center', va='center', color='g', bbox=dict(
                        boxstyle='round', facecolor='white', edgecolor='0.4'), size=self.reward_states[reward_state]*0.5)
            if z != -1:
                ax.text(j, i, '', ha='center', va='center', color='g', bbox=dict(
                    boxstyle='round', facecolor='white', edgecolor='0.4'), size=5)

        if not os.path.exists("../Figure/%s/%d/%4.4f" % (self.now, self.num_agents, self.beta)):
            os.makedirs("../Figure/%s/%d/%4.4f" %
                        (self.now, self.num_agents, self.beta))
        '''print('Grid')
        print(grid)
        print('reward states')
        print(self.reward_states)'''
        # plt.show()
        plt.savefig("../Figure/%s/%d/%4.4f/%.4d.png" %
                    (self.now, self.num_agents, self.beta, iteration))
        plt.close()

    def calculate_coverage(self, k):
        last_k = np.array(self.visited_states[-k:])
        last_k_T = last_k.T
        # from 2d array to 1d array of tuple
        last_k = list(zip(last_k_T[0], last_k_T[1]))
        # print(k,last_k)
        coverage = len(set(last_k))
        return coverage/(self.width*self.height)

    def saveStates(self):
        for i in range(self.num_agents):
            self.agents[i].saveQtable("./Agent_States/agent%dstates" % (i))

        # save Qtables

    def loadStates(self):
        for i in range(self.num_agents):
            self.agents[i].loadQtable("./Agent_States/agent%dstates" % (i))



def plotDiagrams(coverage_array_over_multiple_runs,now,no_of_iter):


    #coverage_array_over_multiple_runs = np.array(coverage_array_over_multiple_runs)   
    #np.save("NoOfIteration_vs_Reward",coverage_array_over_multiple_runs)  



    coverage_array_for_alpha_over_multiple_runs = np.load("./NoOfIteration_vs_Reward.npy")


    mean_coverage_array = np.mean(coverage_array_for_alpha_over_multiple_runs, axis = 0)
    std_coverage_array = np.std(coverage_array_for_alpha_over_multiple_runs,axis = 0)
    error = ((std_coverage_array.T[1] * 1.96) / math.sqrt(no_of_iter));
  
    print(std_coverage_array.T[1])

    plt.rc('xtick',labelsize=10)
    plt.rc('ytick',labelsize=10)
    #Plotting coverage vs number of agents
    fig, ax = plt.subplots()
    ax.plot(mean_coverage_array.T[0],mean_coverage_array.T[1], c='r', linewidth=6)

    ax.errorbar(mean_coverage_array.T[0],mean_coverage_array.T[1], yerr=error, fmt='.k');
    #ax.fill_between(mean_coverage_array.T[0],mean_coverage_array.T[1] - std_coverage_array.T[1],mean_coverage_array.T[1] + std_coverage_array.T[1],alpha = 0.1)
    #plt.title("Number of Iterations vs Reward Accumulated", fontsize=16, fontweight='bold')
    plt.xlabel("Number of Iterations", fontsize=14, fontweight='bold')
    plt.ylabel("Reward Accumulated", fontsize=14, fontweight='bold')
    if not os.path.exists("./Figure/%s/constant_initialization/"%(now)):
        os.makedirs("./Figure/%s/constant_initialization/"%(now))
    plt.savefig("./Figure/%s/constant_initialization/NoOfIteration_vs_Reward.png"%(now))
    plt.close() 




def runSingleAgent(zone,crimes, noOfLngGrid, noOfLatGrid):
    x = datetime.datetime.now();
    now = str(x)[0:10];


    number_of_runs = 25;
    epsilon = 0.4;
    coverage_array_over_multiple_runs = [];
    '''for run in range(number_of_runs):

        reward_array = []

        print("Run No. %d" % (run))
        num_agents_array = np.arange(5, 6)  # Number of agents in the grid
        
        for num_agents in num_agents_array:
            beta_array = [0.01]  # np.linspace(-20,20,num=50)            
            beta_3_array = [-1];
            for beta_3 in beta_3_array:
                iter_array = np.arange(1,52,5);
                for itr in iter_array:

                    print("Run = %d, Beta: %2.2f, Num Agents: %2.2d " %
                          (run, beta_3, num_agents))
                    agents = np.empty([num_agents], dtype=GridWorld)
                    game = RunAgents(zone, crimes, noOfLngGrid, noOfLatGrid, num_agents, True, 0.01, beta_3);
                    

                    for i in range(num_agents):
                        agents[i] = GridWorld(epsilon=epsilon)
                    game.startTraining(agents)
                    reward = game.train(iterations=itr)
                    reward_array.append([itr-1, reward])
                    print("Reward",itr,reward)


        coverage_array_over_multiple_runs.append(reward_array);'''
    
    plotDiagrams(coverage_array_over_multiple_runs,now,number_of_runs);



def getZones(db):
    zones = db.zones.find({});
    df = pd.DataFrame(zones);
    return df;

def getCrimes(db,zone, leftLat,leftLng,rightLat,rightLng):
    crimes = db.crimes.find({'zone':zone, 'lat' : {'$lte': leftLat, '$gte': rightLat}, 'lng' : {'$lte': rightLng, '$gte': leftLng}});
    df = pd.DataFrame(crimes);
    return df;


def saveRoutes(db,zone):


    beta_3_array = [0,-1,-2.5,-10,-20];   
    db_name = ["","1","2","3","4"]           


    for beta_3,name in zip(beta_3_array,db_name):

        df = pd.read_excel("../Results/trajectory_" + str(beta_3) + "_" + zone+".xlsx");

        #print(df["TimeSlot"][0])
        x = datetime.datetime.now()
        for step in df.itertuples():
            #print(step[2],zone);
            loc = db.nearestlocations.find({
                "zone" : zone,
                "location": {
                 "$near": {
                  "$geometry": {
                   "type": "Point",
                   "coordinates": [step[4], step[5]]
                  },
                  "$maxDistance": 1000
                 }
                }
               });
            df = pd.DataFrame(loc);
            
            lat = 0;
            lng = 0;
            if df.size>0:
                print(df.iloc[0, :].location["coordinates"][0]);
                lat = df.iloc[0, :].location["coordinates"][0];
                lng = df.iloc[0, :].location["coordinates"][1];
            else:
                lat = step[4];
                lng = step[5];

            loc = "locations" + name;


            db.vehicles.update_one({"vehicleId" : str(step[2]),"zone":zone},{"$push": \
                {loc: {"_id":ObjectId(),"createdAt":x,"updatedAt":x,"timeSlot":step[3],"lat":lat,"lng":lng}}});

if __name__ == "__main__": 
    client = MongoClient(port=27017);    
    db=client.conFusion;
    zones = getZones(db);

    for zone in zones.itertuples():
        if zone[10] == "ALD":
            continue;
        crimes = getCrimes(db, zone[10], zone[4], zone[5], zone[7], zone[8]);
        latDiff = abs(zone.rightLat - zone.leftLat);
        lngDiff = abs(zone.rightLng - zone.leftLng);
        latLongRatio = latDiff/lngDiff;
        noOfLngGrid = 10;
        noOfLatGrid = (int)(latLongRatio * noOfLngGrid);
        print(zone.zone, noOfLngGrid, noOfLatGrid);
        runSingleAgent(zone[10],crimes,noOfLngGrid,noOfLatGrid);
        #saveRoutes(db,zone[10]);

        

