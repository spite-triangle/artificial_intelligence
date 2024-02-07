
import numpy as np
import pandas as pd

class SarsaLambda():
    def __init__(self,gamma: float = 0.9,alpha:float=0.01,epsilon:float=0.9,lambd: float=0.9):
        # Q 表
        self.qTable:pd.DataFrame = pd.DataFrame(columns=['r','l','u','d'])
        self.eTable = self.qTable.copy()
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.lambd = lambd 
        pass
    
    # NOTE - 由于不知道决策函数pi的真实，还是通过和Q Learnig一样的方法选择动作
    def chooseAction(self,state:int):
        """产生下一个动作"""
        # Q表的状态索引
        self.checkState(self.qTable,state)
        self.checkState(self.eTable,state)

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

    def learn(self,curAct:str,curState:int,reward:int,nextAct:str,nextState:int,done: bool):
        """更新Q表"""
        self.checkState(self.qTable,nextState)
        self.checkState(self.eTable,nextState)

        # 是否结束
        if done == True:
            yt = reward
        else:
            # Q Learning 中 TD 目标的计算方式：
            # yt = reward + self.gamma * self.qTable.loc[nextState,:].max()
            yt = reward + self.gamma * self.qTable.loc[nextState,nextAct]

        delta = self.qTable.loc[curState,curAct] - yt

        self.eTable.loc[curState,:]  *= 0
        self.eTable.loc[curState,curAct] = 1

        self.qTable -=  self.alpha * delta * self.eTable

        self.eTable *= self.lambd * self.gamma


    def checkState(self,table:pd.DataFrame,index: int):
        if index not in table.index:
            table.loc[index] = [0,0,0,0]


if __name__ == "__main__":
    dqn = QLearning()
    print(dqn.chooseAction(1,2))