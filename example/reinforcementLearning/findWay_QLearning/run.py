from QLearning import QLearning
from environment_findWay import Controller,Environment
import tkinter as tk

# 学习周期计数

def train():
    # 一轮游戏
    count = 50
    for i in range(count):
        # 环境重置
        env_ctrl.reset() 
        done = False

        env_ctrl.updateCount(i + 1)

        while done == False:
            # 获取当前状态
            curState = env_ctrl.getState()

            # 选择一个动作
            curAct = agent_ctrl.chooseAction(curState) 
            
            # 执行动作，获得奖励
            reward,done = env_ctrl.agentMove(curAct)

            # 环境渲染延迟
            env_ctrl.render()

            # 下一个环境状态
            nextState = env_ctrl.getState()

            # 学习
            agent_ctrl.learn(curAct,curState, reward, nextState,done)


if __name__ == '__main__':
    root = tk.Tk()
    root.title = "Q Learning Maze"
    root.resizable(False, False) #横纵均不允许调整
    env = Environment(root)
    env_ctrl = Controller(env)
    agent_ctrl = QLearning()
    root.after(100,train)
    root.mainloop()
