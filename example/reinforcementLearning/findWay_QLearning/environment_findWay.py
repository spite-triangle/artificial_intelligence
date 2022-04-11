import tkinter as tk


# 应用程序
class Application(tk.Frame):
    def __init__(self,master:tk.Misc = None):
        # 初始化父类 
        super().__init__(master)
        self.master : tk.Misc = master
        self.pack()
        self.createWidgets()

    def createWidgets(self):
        """创建应用控件"""
        self.lab_test : tk.Label  = tk.Label(self,text="test")
        self.lab_test.pack()

# 控制器
class Controller():
    def __init__(self):

        pass
    def reset(self):
        """程序重置"""

    def render(self):
        """执行一步"""

    def choose_action(self):
        """选择一个动作"""

    def perform(self):
        """执行动作"""

if(__name__ == "__main__"):
    # 根窗口
    root = tk.Tk()
    root.title = "test"
    root.geometry("512x128+128+128")
    app = Application(root)
    root.mainloop()

