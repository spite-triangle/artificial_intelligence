from Sarsa import Sarsa
from environment_findWay import Controller,Environment
import tkinter as tk

# 学习周期计数

def train():
    # 一轮游戏
    count = 100
    for i in range(count):
        # 环境重置
        env_ctrl.reset() 

        env_ctrl.updateCount(i + 1)

        # 初始状态
        nextState  = env_ctrl.getState()
        # 初始化第一个动作
        nextAct = agent_ctrl.chooseAction(nextState) 

        while True:

            # 当前状态
            curState = nextState
            # 当前要执行的动作
            curAct = nextAct
            
            # 执行动作，获得奖励
            reward,done = env_ctrl.agentMove(curAct)
            # 环境渲染延迟
            env_ctrl.render()

            # 下一个环境状态
            nextState = env_ctrl.getState()
            # 下一个动作
            nextAct = agent_ctrl.chooseAction(nextState)

            # 学习
            agent_ctrl.learn(curAct,curState,reward,nextAct,nextState,done)

            if done == True:
                break




if __name__ == '__main__':
    root = tk.Tk()
    root.title = "Q Learning Maze"
    root.resizable(False, False) #横纵均不允许调整
    env = Environment(root)
    env_ctrl = Controller(env)
    agent_ctrl = Sarsa()
    root.after(100,train)
    root.mainloop()
