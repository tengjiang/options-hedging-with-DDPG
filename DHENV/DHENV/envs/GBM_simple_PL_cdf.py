import math
import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import describe

import gym
from gym import spaces
from gym.utils import seeding


class GBM_simple_PL_cdf(gym.Env):
    
    def __init__(self, std = 0.3, mean = 0.2, T = 100, S = 10, strike = 8, riskfree = 0.01, dividen = 0, deltat = 0.001, transac = 0.01):
        self.std = std
        self.mean = mean
        self.maturity = T * deltat
        self.maturity_const = T * deltat
        self.strike = strike
        self.riskfree = riskfree
        self.dividen = dividen
        self.deltat = deltat
        self.transac = transac
        self.count = 0

        self.S = S
        self.S_const = S
        self.prices = [S]
        
        self.reward = 0
        self.rewards = []

        self.stock_number = 0
        self.stock_numbers = [0]  
       

        c = self.bscall()
        p = self.bsput()

        self.callprices = [c]
        self.putprices = [p]
        
        self.saving = c[0]
        self.savings = [c[0]]  

        self.Account = - c[0] + self.stock_number * self.S + self.saving
        self.Accounts = [self.Account]

        self.seednumber = self.seed()

        self.actions = []

        self.action_space = spaces.Box(low = -1, high = 1, shape = (1,), dtype=np.float32)
        self.observation_space = spaces.Box(low = -np.inf, high = np.inf, shape = (4,),dtype=np.float32)#asset: stock, bank, stockprice, maturity
    
    def bscall(self):
        '''
        bsCall <- function(s, K, sigma, t, r=0, d=0){
        d1 <- (log(s/K) + (r - d)*t)/(sigma*sqrt(t)) + sigma*sqrt(t)/2
        d2 <- d1 - sigma*sqrt(t)
    
        c <- s*exp(-d*t)*pnorm(d1) - K*exp(-r*t)*pnorm(d2)
        delta <- exp(-d*t)*pnorm(d1)
        Gam <- dnorm(d1)/s/sigma/sqrt(t)
    
        data.frame(c, delta, Gam)
        }
        '''
        d1 = ( math.log(self.S / self.strike) + (self.riskfree - self.dividen) * self.maturity ) / self.std / math.sqrt(self.maturity) + self.std * math.sqrt(self.maturity) / 2
        d2 = d1 - self.std * math.sqrt(self.maturity)
        c = self.S * math.exp(-self.dividen*self.maturity) * stats.norm.cdf(d1) - self.strike * math.exp(-self.riskfree*self.maturity) * stats.norm.cdf(d2)
        delta = math.exp(-self.dividen*self.maturity) * stats.norm.cdf(d1)
        Gam = stats.norm.pdf(d1)/self.S/self.std/math.sqrt(self.maturity)
        return c, delta, Gam, d1, d2


    def bsput(self):
        d1 = ( math.log(self.S / self.strike) + (self.riskfree - self.dividen) * self.maturity )  / self.std / math.sqrt(self.maturity) + self.std * math.sqrt(self.maturity) / 2
        d2 = d1 - self.std * math.sqrt(self.maturity)
        p = self.strike * math.exp(-self.riskfree*self.maturity) * stats.norm.cdf(-d2) - self.S * math.exp(-self.dividen*self.maturity) * stats.norm.cdf(-d1) 
        delta = -math.exp(-self.dividen*self.maturity) * stats.norm.cdf(-d1)
        Gam = stats.norm.pdf(d1)/self.S/self.std/math.sqrt(self.maturity)
        return p, delta, Gam


    def GBMmove(self):
        dW = np.random.randn(1)[0] * math.sqrt(self.deltat)
        self.S = self.S + self.S * (dW * self.std + self.mean * self.deltat)


    def step(self, action):
        stock_add = action
        #stock_add = max(-1e5, min(stock_add, 1e5))
        self.actions.append(action)
        stock_money = stock_add * self.S
        self.saving -= stock_money
        self.saving -= abs(stock_money) * self.transac
        self.stock_number += stock_add
        self.stock_numbers.append(self.stock_number)

        self.GBMmove()
        self.maturity -= self.deltat
        self.maturity = max(self.maturity, 1e-15)
        saving_reward = self.saving * (math.exp(self.riskfree * self.deltat) - 1)
        self.saving += saving_reward 
        self.savings.append(self.saving)
        self.prices.append(self.S)
        self.callprices.append(self.bscall())
        self.putprices.append(self.bsput())

        self.Account = -self.callprices[-1][0] + self.savings[-1] + self.prices[-1] * self.stock_numbers[-1]
        self.Accounts.append(self.Account)

        self.reward = ( -self.callprices[-1][0] - (-self.callprices[-2][0]) ) + (self.prices[-1] - self.prices[-2]) * action - abs(stock_money) * self.transac + saving_reward

        done = False
        if self.maturity <= 1e-15:
            done = True
            self.count += 1
            if self.count % 100 == 1:
                #self.show()
                pass
        if done:
            self.reward += (-self.transac * self.stock_number * self.S)
        self.rewards.append(self.reward)
        self.construct_state()

        return self.state, self.reward, done, {}
    
    def construct_state(self):
        self.state = np.array([self.stock_number, self.saving, self.S, self.maturity], dtype=np.float32)


    def show(self):
        plt.plot(self.prices, label = 'prices')
        plt.show()
        plt.plot(self.actions, label = 'actions')
        plt.show()
        plt.plot(self.Accounts, label = 'Accounts')
        plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.maturity = self.maturity_const

        self.S = self.S_const
        self.prices = [self.S]
        
        self.reward = 0
        self.rewards = []

        self.stock_number = 0
        self.stock_numbers = [0]  

        c = self.bscall()
        p = self.bsput()

        self.saving = c[0]
        self.savings = [self.saving]

        self.callprices = [c]
        self.putprices = [p]

        self.Account = - c[0] + self.stock_number * self.S + self.saving
        self.Accounts = [self.Account]
        
        self.actions = []

        self.construct_state()
        
        return self.state

