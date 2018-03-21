import math
import numpy as np


class MafiaEnv():


    def __init__(self):

        self.playercount = 5
        self.badplayercount = 2
        self.badplayers = np.random.permutation(self.playercount)[:self.badplayercount]
        self.bpb = [False] * 5
        for i in self.badplayers:
            self.bpb[i] = True

        self.dpb = [False] * 5

        self.liveplayers = list(range(self.playercount))
        self.livebadplayers = self.badplayers

        # i guess you can vote for yourself?
        # maybe put some punishment for invalid actions to help learning
        self.action_space1 = self.liveplayers
        self.action_space2 = self.liveplayers

        # if game isn't finished by then everyone loses
        self.maxturns = self.playercount
        self.turn = 0

        # unpadded, add 0's and stuff to get unrepresented turns
        # 0/1 array each entry is player * player matrix 1 if a voted for b
        self.dayvotes = []
        self.nightvotes = []

        # vector of 0 if no deaths
        self.daydeaths = []
        self.nightdeaths = []

        # assemble state from votes + deaths
        # teams have different agents trained

        # inputs should also include role and agent index


    #day
    def step1(self, actionset):
        self.turn += 1

        #should votes from dead ppl be counted?
        votehistory = [[0] * self.playercount] * self.playercount

        votes = [0] * self.playercount
        for i in self.liveplayers:
            votes[actionset[i]] += 1
            votehistory[i][actionset[i]] = 1

        self.dayvotes.append(votehistory)

        maxvote = 0
        maxperson = -1
        for x in range(self.playercount):
            if votes[x] > maxvote:
                maxperson = x
                maxvote = votes[x]
            elif votes[x] == maxvote:
                maxperson = -1

        alreadydead = False
        if maxperson != -1:
            if self.dpb[maxperson]:
                alreadydead = True
                self.daydeaths.append([0] * self.playercount)
            else:
                dt = [0] * self.playercount
                dt[maxperson] = 1
                self.daydeaths.append(dt)

            self.dpb[maxperson] = True
            try:
                self.liveplayers.remove(maxperson)
                self.livebadplayers.remove(maxperson)
            except ValueError:
                pass
        else:
            self.daydeaths.append([0] * self.playercount)

        # array is array of rewards for agents

        #good people win
        if self.livebadplayers == 0:
            return maxperson, [1 - 2 * i for i in self.bpb], True

        #bad people win
        elif 2 * len(self.livebadplayers) >= len(self.liveplayers):
            return maxperson, [2 * i - 1 for i in self.bpb], True
        #game continues
        else:
            if self.turn > self.maxturns:
                return None, [-1] * self.playercount, True

            if alreadydead:
                return None, [0] * self.playercount, False
            else:
                return maxperson, [0] * self.playercount, False



        # return np.array(self.state), reward, done, {}

    #night
    def step2(self, actionset):
        votes = [0] * self.playercount
        votehistory = [[0] * self.playercount] * self.playercount
        for i in self.livebadplayers:
            votes[actionset[i]] += 1
            votehistory[i][actionset[i]] = 1

        self.nightvotes.append(votehistory)

        maxvote = 0
        maxperson = -1
        for x in range(self.playercount):
            if votes[x] > maxvote:
                maxperson = x
                maxvote = votes[x]
            elif votes[x] == maxvote:
                maxperson = -1

        alreadydead = False
        if maxperson != -1:
            if self.dpb[maxperson]:
                alreadydead = True
                self.nightdeaths.append([0] * self.playercount)
            else:
                dt = [0] * self.playercount
                dt[maxperson] = 1
                self.nightdeaths.append(dt)

            self.dpb[maxperson] = True
            try:
                self.liveplayers.remove(maxperson)
                self.livebadplayers.remove(maxperson)
            except ValueError:
                pass
        else:
            self.nightdeaths.append([0] * self.playercount)

        #good people win
        if self.livebadplayers == 0:
            return maxperson, [1 - 2 * i for i in self.bpb], True

        #bad people win
        elif 2 * len(self.livebadplayers) >= len(self.liveplayers):
            return maxperson, [2 * i - 1 for i in self.bpb], True
        #game continues
        else:
            if alreadydead:
                return None, [0] * self.playercount, False
            else:
                return maxperson, [0] * self.playercount, False

        # returns person who died, return vector for players,
        # and bool on if it is terminal state
        # return np.array(self.state), reward, done, {}

    def getstate(self, agent):
        # bad
        if self.bpb[agent]:

        #good
        else:
            result = np.zeros(shape=(self.maxturns,self.playercount,self.playercount))

    def reset(self):
        self.badplayers = np.random.permutation(self.playercount)[:self.badplayercount]
        self.bpb = [False] * 5
        for i in self.badplayers:
            self.bpb[i] = True

        self.dpb = [False] * 5

        self.liveplayers = list(range(self.playercount))
        self.livebadplayers = self.badplayers

        self.dayvotes = []
        self.nightvotes = []
        self.daydeaths = []
        self.nightdeaths = []
        self.turn = 0
