
from ctypes.wintypes import UINT
import time
import tkinter as tk

# 迷宫单元的大小，像素
UNIT = 50 
# 空白留边，像素
MARGIN = 6

# 迷宫规模
MAZE_SIZE = 8

# ==========定义迷宫=========
MAZE = [[0 for col in range(MAZE_SIZE)] for row in range(MAZE_SIZE)]
# 陷阱
TRAP_VAL = -1
# 目标
TARGET_VAL = 1
# 陷阱位置
MAZE[3][5] = TRAP_VAL
MAZE[5][4] = TRAP_VAL
MAZE[2][5] = TRAP_VAL
MAZE[1][2] = TRAP_VAL
MAZE[3][3] = TRAP_VAL
MAZE[6][6] = TRAP_VAL
MAZE[4][4] = TRAP_VAL
MAZE[7][6] = TRAP_VAL
MAZE[4][2] = TRAP_VAL
MAZE[2][4] = TRAP_VAL
MAZE[7][1] = TRAP_VAL

# 目标位置
MAZE[7][2] = TARGET_VAL


# 应用程序
class Environment(tk.Frame):
    def __init__(self,master:tk.Misc = None):
        # 初始化父类 
        super().__init__(master)
        self.master : tk.Misc = master
        self.pack()

        self.createWidgets()

    def createWidgets(self):
        """创建应用控件"""

        # 将背景设置为黑色
        self.config(bg='black')
        
        # 创建画布
        self.canvas:tk.Canvas = tk.Canvas(self, bg='white',
                           height= UNIT*MAZE_SIZE,
                           width= UNIT*MAZE_SIZE)

        # 创建陷阱与目标
        self.createStaticShare()

        # 角色
        self.agent = self.canvas.create_oval(self.getPosFromIndex(0,0),fill='green',outline='green')

        # 布置好画布
        self.canvas.pack()

        # 回合计数器
        self.lableCount = tk.Label(self,text='',height=2,bg='black',fg='white',font=('黑体',20,'bold'))
        self.lableCount.pack()


    def agentMoveRender(self,row,col):
        """渲染角色移动"""
        self.canvas.moveto(self.agent,col*UNIT + MARGIN, row*UNIT + MARGIN)
        
    def updataCount(self,count:int):
        self.lableCount.config(text=str(count))

    def createStaticShare(self):
        """在画布上，绘制静止的形状：陷阱、目标、网格线"""
        # 陷阱、目标
        for row in range(MAZE_SIZE):
            for col in range(MAZE_SIZE):
                value = MAZE[row][col]
                # 空
                if value == 0:
                    continue
                # 画陷阱
                if value == TRAP_VAL:
                    self.canvas.create_rectangle(self.getPosFromIndex(row,col),fill='black')
                    continue
                # 画目标
                if value == TARGET_VAL:
                    self.canvas.create_oval(self.getPosFromIndex(row,col),fill='red',outline='red')

        # 横向网格线
        for row in range(1,MAZE_SIZE):
            self.canvas.create_line(0,row * UNIT,MAZE_SIZE*UNIT,row * UNIT)

        # 竖向网格线
        for col in range(1,MAZE_SIZE):
            self.canvas.create_line(col * UNIT,0,col * UNIT,MAZE_SIZE*UNIT)

    def getPosFromIndex(self,row:int,col:int):
        """将图形在maze的索引转换为像素坐标"""
        return col*UNIT + MARGIN, row*UNIT + MARGIN, (col+1)*UNIT - MARGIN, (row+1)*UNIT  - MARGIN


class Action():
    def __init__(self,row:int,col:int):
        self.row = row
        self.col = col

# 控制器
class Controller():
    def __init__(self,app: Environment):
        self.app = app
        self.agentRow = 0
        self.agentCol = 0

        self.actionTable = {'r':Action(0,1),'l':Action(0,-1),'u':Action(-1,0),'d':Action(1,0)}

    def reset(self):
        """程序重置"""
        # 角色归0
        self.app.agentMoveRender(0,0)
        self.agentRow = 0
        self.agentCol = 0

    def agentMove(self,action:str):
        """执行一步，给出评价"""

        # 预先移动一步
        self.agentRow += self.actionTable[action].row
        self.agentCol += self.actionTable[action].col

        # 平安无事的奖励
        reward = 0
        done = False
        # 超界
        if self.agentRow >= MAZE_SIZE or self.agentRow < 0:
            self.agentRow -= self.actionTable[action].row
            reward = -1
        elif self.agentCol >= MAZE_SIZE or self.agentCol < 0:
            self.agentCol -= self.actionTable[action].col
            reward = -1
        # 掉入陷阱
        elif MAZE[self.agentRow][self.agentCol] == TRAP_VAL:
            reward = -1
            done = True
        # 到达目标
        elif MAZE[self.agentRow][self.agentCol] == TARGET_VAL:         
            reward = 10
            done = True

        return reward,done

    def getState(self) -> int:
        """获取状态"""
        return self.agentRow*MAZE_SIZE + self.agentCol


    def render(self):
        time.sleep(0.05)
        # 实际移动
        self.app.agentMoveRender(self.agentRow,self.agentCol)
        self.app.master.update()
        
    def updateCount(self,count:int):
        self.app.updataCount(count)

def update():
    root.after(100,update)
    

if(__name__ == "__main__"):
    # 根窗口
    root = tk.Tk()
    root.title = "test"
    root.resizable(False, False) #横纵均不允许调整
    app = Environment(root)
    env_ctrl = Controller(app)
    root.after(1000,update)
    root.mainloop()

