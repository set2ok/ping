from paddle import Paddle
from ball import Ball
from bot import Bot
from datetime import datetime
import random
import keyboard
import numpy as np
import threading

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

        self.models = []

    def creat_bot(self,padels, balls):
        return Bot(padels[0],padels[1:], balls)

    def creat_padels(self):
        player = Paddle(80, self.paddle_height, self.inner_bound, 350, type="player")
        padels = [player]
        for paddle_nr in range(6):
            padels.append(Paddle(100, self.paddle_height, self.outer_bound, 450))

        return padels

    def creat_balls(self,padels):
        balls = []
        for balls_nr in range(5):
            balls.append(Ball(self.outer_bound,self.inner_bound,10, padels))
        return balls

    def update_state(self,padels,balls,bot):
        padels_active = padels[:random.randint(2, 6)]
        balls_active = balls[:random.randint(1, 5)]
        for ball in balls_active:
            ball.paddles = padels_active
            ball.spawn()
        for paddle in padels_active:
            paddle.spawn()
        bot.oponents = padels_active
        bot.balls = balls_active

        return padels_active, balls_active
    def run_simulation(self, thread_id):
        start_time_update = datetime.now()
        start_time_fps = datetime.now()
        count = 0
        dt = 1 / 100
        padles = self.creat_padels()
        balls = self.creat_balls(padles)
        bot = self.creat_bot(padles,balls)

        self.models.append(bot)

        padels_active, balls_active = self.update_state(padles,balls,bot)

        # Träningsloop
        while True:
            count += 1
            dt *= random.uniform(0.98, 1.02)
            dt = np.clip(dt, 1 / 150, 1 / 60)
            time = datetime.now()
            if keyboard.is_pressed("s"):
                return

            if (time - start_time_update).total_seconds() >= 60:
                print(f"Thread {thread_id}: updating state")
                padels_active, balls_active = self.update_state(padles,balls,bot)
                start_time_update = time

            dif_time = (time - start_time_fps).total_seconds()
            if dif_time >= 1.0:
                fps = (count / dif_time)
                print(f"Thread {thread_id}: FPS {fps}, dt {dt}")
                count = 0
                start_time_fps = time

            padles[0].move(random.uniform(-1, 1), dt)
            for ball in balls_active:
                ball.update(dt, bot)
            bot.move(dt)

            # När simulationen är klar, lagra modellen (eller vikt)
            if keyboard.is_pressed("m"):
                self.merge_and_save()

    def average_weights(self, models):
        """Calculate the average weights from a list of models"""
        weights = [model.get_weights() for model in models]
        avg_weights = [np.mean([w[i] for w in weights], axis=0) for i in range(len(weights[0]))]
        return avg_weights

    def merge_and_save(self):
        """Merge all models and save the result to a file"""
        # Antag att du sparar vikterna för alla tränade modeller
        if len(self.models) == 0:
            print("No models to merge.")
            return

        # Medelvärde vikterna från alla modeller
        avg_weights = self.average_weights(self.models)
        final_model = self.models[0]
        final_model.model_train.set_weights(avg_weights)

        # Spara den slutliga modellen
        final_model.save('model_v4.keras')  # Spara till fil
        print("Model merged and saved")

    def run_threads(self, num_threads):
        threads = []
        self.models = []
        for i in range(num_threads):
            thread = threading.Thread(target=self.run_simulation, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

train = Train()
train.run_threads(5)