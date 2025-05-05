from paddle import Paddle
from ball import Ball
from bot import Bot
from datetime import datetime
import random
import keyboard

class Train():
    def __init__(self):
        self.window_size = 500
        self.exit = False
        self.paddle_height = 10
        self.outer_bound = [(self.paddle_height/2,self.paddle_height/2), (self.window_size - self.paddle_height/2, self.paddle_height/2),
                            (self.window_size - self.paddle_height/2,self.window_size - self.paddle_height/2),
                            (self.paddle_height/2, self.window_size - self.paddle_height/2)]
        self.outer_bound = [(int(bound[0]),bound[1]) for bound in self.outer_bound]

        self.inner_bound = [(200,200),(300,200),(300,300),(200,300)]


        self.player = Paddle( 80 ,self.paddle_height,self.inner_bound, 350, type="player")
        self.padels = [self.player]
        for paddle_nr in range(6):
            self.padels.append(Paddle(100,self.paddle_height,self.outer_bound,450))
        self.balls = []
        for balls_nr in range(5):
            self.balls.append(Ball(self.outer_bound,self.inner_bound,10, self.padels))
        self.bot = Bot(self.player,self.padels[1:], self.balls)

    def update_state(self):
        padels_active = self.padels[:random.randint(2, 6)]
        balls_active = self.balls[:random.randint(1, 5)]
        for ball in balls_active:
            ball.paddles = padels_active
            ball.spawn()
        for paddle in padels_active:
            paddle.spawn()
        self.bot.oponents = padels_active
        self.bot.balls = balls_active

        return padels_active, balls_active
    def main(self):

        start_time_update = datetime.now()
        start_time_fps = datetime.now()
        count = 0
        dt = 1/100

        padels_active, balls_active = self.update_state()
        while True:
            count += 1
            time = datetime.now()
            if keyboard.is_pressed("s"):
                print("saved")
                self.bot.save_model("model_v2.keras")

            if (time - start_time_update).total_seconds() >= 60:
                padels_active, balls_active = self.update_state()

                start_time_update = time

            dif_time = (time - start_time_fps).total_seconds()
            if dif_time >= 1.0:
                fps =(count/dif_time)
                print(fps)
                dt = 1/fps
                count = 0
                start_time_fps = time

            for ball in balls_active:
                ball.update(dt, self.bot)
            self.bot.move(dt)

train = Train()
train.main()