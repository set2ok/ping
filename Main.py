import random

import pygame

from paddle import Paddle
from ball import Ball
from bot import Bot
from datetime import datetime

class Main:

    def __init__(self):
        self.window_size = 500
        self.exit = False
        self.points = 0
        self.paddle_height = 10
        self.outer_bound = [(self.paddle_height/2,self.paddle_height/2), (self.window_size - self.paddle_height/2, self.paddle_height/2),
                            (self.window_size - self.paddle_height/2,self.window_size - self.paddle_height/2),
                            (self.paddle_height/2, self.window_size - self.paddle_height/2)]
        self.outer_bound = [(int(bound[0]),bound[1]) for bound in self.outer_bound]

        self.inner_bound = [(175,175),(325,175),(325,325),(175,325)]



    # Will initialise the beginning of the game, create all essential objects etc.
    def setup(self):
        self.player = Paddle( 110 ,self.paddle_height,self.inner_bound, 350, type="player")
        self.padels = [self.player]
        for paddle_nr in range(6):
            self.padels.append(Paddle(100,self.paddle_height,self.outer_bound,450))
        self.active_padels = self.padels[:4]
        self.balls = []
        for balls_nr in range(5):
            self.balls.append(Ball(self.outer_bound,self.inner_bound,10, self.active_padels))
        self.active_balls = self.balls[:1]

        self.bot = Bot(self.player,self.active_padels[1:], self.active_balls)


    def update_state(self):
        change = random.randint(0,2)

        if change == 0 and len(self.active_balls) < 6:
            self.active_balls = self.balls[:len(self.active_balls) + 1]
            self.bot.balls = self.active_balls

        elif change ==1 and len(self.active_padels) <8:
            self.active_padels = self.padels[:len(self.active_padels) +1]
            self.bot.opponents = self.active_padels[1:]
            for ball in self.active_balls:
                ball.paddles = self.active_padels
        else:
            for ball in self.active_balls:
                ball.speed *= 1.1

    def main(self):


        clock = pygame.time.Clock()
        pygame.init()

        # CREATE A CANVAS
        canvas = pygame.display.set_mode((self.window_size, self.window_size))

        # TITLE OF CANVAS
        pygame.display.set_caption("Retro game")

        #font
        self.font = pygame.font.Font(None, 21)


        # SETUP GAME OBJECTS
        self.setup()
        count = 0
        self.start_time = datetime.now()
        time_counter = datetime.now()
        # GAME LOOP
        loop_count = 0
        while not self.exit:
            loop_count += 1
            count += 1
            time = datetime.now()
            dif_time = (time-time_counter).total_seconds()
            if dif_time >= 10:
                time_counter = time
                self.update_state()

            self.draw(canvas)

            self.handle_events()

            pygame.display.update()
            self.dt = clock.tick(150) / 1000

            for ball in self.active_balls:
                self.points += ball.update(self.dt, self.bot)
            self.bot.move(self.dt)
            if self.points < -5:
                self.exit = True

    # Runs every frame. What will happen each frame
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.exit = True

        keys = pygame.key.get_pressed()

        self.react_to_user_input(keys)

    # Will redraw the screen each frame
    def draw(self, canvas):
        canvas.fill((0, 0, 0))

        for paddle in self.active_padels:
            if paddle.type == "bot":
                pygame.draw.polygon(canvas, (0, 0, 255), paddle.figure)
            else:
                pygame.draw.polygon(canvas, (255, 255, 255), paddle.figure)

        for ball in self.active_balls:
            pygame.draw.circle(canvas, center=(ball.x, ball.y), radius=ball.radius, color=(255, 0, 0))

        for point in self.inner_bound:
            pygame.draw.circle(canvas,center= point, radius= 3, color= (0,255,0))

        #text
        text_score = self.font.render(f"Score: {self.points}", True, (255, 255, 255))  # White color
        # text place
        score_rect = text_score.get_rect(center=(250, 240))

        canvas.blit(text_score, score_rect)

        text_time = self.font.render(f"Time: {(datetime.now() - self.start_time).total_seconds():.2f} sek", True, (0, 0, 255))

        time_rect = text_time.get_rect(center=(250, 260))

        canvas.blit(text_time, time_rect)


        pygame.display.flip()


    def react_to_user_input(self, keysPressed):

        if keysPressed[pygame.K_a]:
            self.player.move(-1,self.dt)

        if keysPressed[pygame.K_d]:
            self.player.move(1,self.dt)

        if keysPressed[pygame.K_p]: # save module
            self.bot.save_model()




main = Main()

main.main()