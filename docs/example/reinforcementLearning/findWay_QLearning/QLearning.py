
import numpy as np
import pandas as pd

class QLearning():
    def __init__(self,gamma: float = 0.9,alpha:float=0.1,epsilon:float=0.9):
        # Q 表
        self.qTable:pd.DataFrame = pd.DataFrame(columns=['r','l','u','d'])
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        pass

    def chooseAction(self,state:int):
        """产生下一个动作"""
        # Q表的状态索引
        self.checkState(state)

        dice = np.random.random()
        if dice < self.epsilon:
            # 提取当前状态的 动作概率值 policy
            datas = self.qTable.loc[state,:]
            # 将最大值的所有动作提取出来
            datas = datas.loc[datas == datas.max()].index
            # 从最大动作中选一个
            act =  np.random.choice(datas)
        else:
            act = np.random.choice(self.qTable.columns)

        return act

    def learn(self,curAct:str,curState:int,reward:int,nextState:int,done: bool):
        """更新Q表"""

        self.checkState(nextState)

        # 是否结束
        if done == True:
            yt = reward
            self.qTable.loc[curState,curAct] = yt
        else:
            yt = reward + self.gamma * self.qTable.loc[nextState,:].max()
            self.qTable.loc[curState,curAct] -= self.alpha * (self.qTable.loc[curState,curAct] - yt) 


    def checkState(self,index: int):
        if index not in self.qTable.index:
            self.qTable.loc[index] = [0,0,0,0]

if __name__ == "__main__":
    dqn = QLearning()
    print(dqn.chooseAction(1,2))