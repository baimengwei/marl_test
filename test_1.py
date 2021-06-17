import os
import time
from multiprocessing import Process
from multiprocessing.managers import BaseManager


class App:
    def __init__(self):
        self.set_data(0, 0)

    def get_data(self):
        return self.x, self.y

    def set_data(self, x, y):
        self.x = x
        self.y = y


app = App()


class RobotManager(BaseManager):
    pass


RobotManager.register("get_app", lambda: app)
m = RobotManager(address=("127.0.0.1", 8001), authkey=b"Fuck, world")
m.start()


def RunApp():
    m = RobotManager(address=("127.0.0.1", 8001), authkey=b"Fuck, world")
    m.connect()
    app = m.get_app()
    while True:
        data = app.get_data()
        if data[0] > 10:
            print("big data: ", data)
            print("exit child process")
            break
        else:
            print("small data: ", data)
        time.sleep(1)


if __name__ == "__main__":
    p = Process(target=RunApp)
    p.start()
    app = m.get_app()  # 必须使用这一句，不然的话，修改不是共享的
    time.sleep(5)  # child process will print small data several times
    app.set_data(20, 20)  # child process print big data and exit
    p.join()
