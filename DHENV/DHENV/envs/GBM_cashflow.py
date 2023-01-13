import math
import scipy.stats as stats
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.stats import describe

import gym
from gym import spaces
from gym.utils import seeding


class GBM_cashflow(gym.Env):

    def __init__(self, std=0.3, mean=0.2, T=10, s=10, strike=8, riskfree=0.04, dividen=0, deltat=0.01, transac=0.01):
        self.std = std
        self.mean = mean
        self.time_to_maturity = T * deltat
        self.maturity_const = T * deltat
        self.strike = strike
        self.riskfree = riskfree
        self.dividen = dividen
        self.deltat = deltat
        self.transac = transac
        self.count = 0

        self.S = s
        self.S_const = s
        self.prices = [s]

        self.reward = 0
        self.rewards = []

        c = self.bscall()
        p = self.bsput()

        self.money_account = c[0]
        self.money_accounts = [c[0]]

        self.position = 0
        self.positions = [0]

        self.callprices = [c]
        self.putprices = [p]

        self.balance = 0
        self.balances = [0]

        self.seednumber = self.seed()

        self.actions = []
        self.action_space = spaces.Box(low=-10, high=10, shape=(1,),
                                       dtype=np.float32)

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
        d1 = (math.log(self.S / self.strike) + (self.riskfree - self.dividen) * self.time_to_maturity) / self.std / math.sqrt(
            self.time_to_maturity) + self.std * math.sqrt(self.time_to_maturity) / 2
        d2 = d1 - self.std * math.sqrt(self.time_to_maturity)
        c = self.S * math.exp(-self.dividen * self.time_to_maturity) * stats.norm.cdf(d1) - self.strike * math.exp(
            -self.riskfree * self.time_to_maturity) * stats.norm.cdf(d2)
        delta = math.exp(-self.dividen * self.time_to_maturity) * stats.norm.cdf(d1)
        Gam = stats.norm.pdf(d1) / self.S / self.std / math.sqrt(self.time_to_maturity)
        return c, delta, Gam, d1, d2

    def bsput(self):
        d1 = (math.log(self.S / self.strike) + (self.riskfree - self.dividen) * self.time_to_maturity) / self.std / math.sqrt(
            self.time_to_maturity) + self.std * math.sqrt(self.time_to_maturity) / 2
        d2 = d1 - self.std * math.sqrt(self.time_to_maturity)
        p = self.strike * math.exp(-self.riskfree * self.time_to_maturity) * stats.norm.cdf(-d2) - self.S * math.exp(
            -self.dividen * self.time_to_maturity) * stats.norm.cdf(-d1)
        delta = -math.exp(-self.dividen * self.time_to_maturity) * stats.norm.cdf(-d1)
        Gam = stats.norm.pdf(d1) / self.S / self.std / math.sqrt(self.time_to_maturity)
        return p, delta, Gam

    def GBMmove(self):
        dlogS = np.random.randn(1)[0] * math.sqrt(self.deltat) * self.std + (self.mean - 0.5*self.std**2) * self.deltat
        self.S = np.exp(np.log(self.S) + dlogS)

    def step(self, action):
        stock_add = action
        stock_money = stock_add * self.S

        self.GBMmove()
        self.time_to_maturity -= self.deltat

        self.money_account -= stock_money
        self.money_account -= abs(stock_money) * self.transac
        self.money_account *= math.exp(self.riskfree * self.deltat)
        self.money_accounts.append(self.money_account)

        self.position += stock_add
        self.positions.append(self.position)

        self.prices.append(self.S)
        self.callprices.append(self.bscall())
        self.putprices.append(self.bsput())

        self.balance = -self.callprices[-1][0] + self.money_accounts[-1] + self.prices[-1] * self.positions[-1]
        self.balances.append(self.balance)

        if self.time_to_maturity < 1e-15:
            final_stock = self.prices[-1] * self.positions[-1]
            final_money = self.money_accounts[-1] + final_stock - abs(final_stock) * self.transac - self.callprices[-1][0]
            self.reward = final_money - self.money_accounts[-2]
        else:
            self.reward = self.money_accounts[-1] - self.money_accounts[-2]

        done = False
        if self.time_to_maturity < 1e-15:
            done = True
            self.count += 1
            if self.count % 1000 == 1:
                self.show()
        self.actions.append(action)
        self.construct_state()

        return self.state, self.reward, done, {}

    def construct_state(self):
        self.state = np.array([self.position, self.money_account, self.S, self.time_to_maturity], dtype=np.float32)

    def show(self):
        plt.plot(self.prices, label='prices')
        plt.show()
        plt.plot(self.actions, label='actions')
        plt.show()
        plt.plot(self.balances, label='Accounts')
        plt.show()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):

        self.time_to_maturity = self.maturity_const

        self.S = self.S_const
        self.prices = [self.S]

        self.reward = 0
        self.rewards = []

        c = self.bscall()
        p = self.bsput()

        self.money_account = c[0]
        self.money_accounts = [c[0]]

        self.balance = 0
        self.balances = [0]

        self.position = 0
        self.positions = [0]

        self.callprices = [c]
        self.putprices = [p]

        self.actions = []

        self.construct_state()

        return self.state

