import pygame

from paddle import Paddle
from ball import Ball
from bot import Bot
from datetime import datetime

class Main:

    def __init__(self):
        self.window_size = 500
        self.exit = False
        self.paddle_height = 10
        self.outer_bound = [(self.paddle_height/2,self.paddle_height/2), (self.window_size - self.paddle_height/2, self.paddle_height/2),
                            (self.window_size - self.paddle_height/2,self.window_size - self.paddle_height/2),
                            (self.paddle_height/2, self.window_size - self.paddle_height/2)]
        self.outer_bound = [(int(bound[0]),bound[1]) for bound in self.outer_bound]

        self.inner_bound = [(200,200),(300,200),(300,300),(200,300)]

    # Will initialise the beginning of the game, create all essential objects etc.
    def setup(self):
        self.player = Paddle( 80 ,self.paddle_height,self.inner_bound, 350, type="player")
        self.paddle0 = Paddle(100,self.paddle_height,self.outer_bound,450)
        self.paddle1 = Paddle( 100, self.paddle_height, self.outer_bound, 450)
        self.paddle2 = Paddle( 100, self.paddle_height, self.outer_bound, 450)
        self.paddle3 = Paddle(100, self.paddle_height, self.outer_bound, 450)
        self.padels = [self.player,self.paddle0,self.paddle1,self.paddle2,self.paddle3]
        self.ball = Ball(self.outer_bound,self.inner_bound,10, self.padels)
        self.bot = Bot(self.player,self.padels[1:], [self.ball])
    def main(self):

        clock = pygame.time.Clock()
        pygame.init()

        # CREATE A CANVAS
        canvas = pygame.display.set_mode((self.window_size, self.window_size))

        # TITLE OF CANVAS
        pygame.display.set_caption("Retro game")

        # SETUP GAME OBJECTS
        self.setup()
        count = 0
        start_time = datetime.now()
        # GAME LOOP
        loop_count = 0
        while not self.exit:
            loop_count += 1
            count += 1
            time = datetime.now()
            dif_time = (time-start_time).total_seconds()
            if dif_time >= 1.0:
                print(count/dif_time)
                count = 0
                start_time = time
            self.draw(canvas)

            self.handle_events()

            pygame.display.update()
            self.dt = clock.tick(240) / 1000
            self.ball.update(self.dt, self.bot)
            self.bot.move(self.dt)

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
        pygame.draw.circle(canvas,center=(self.ball.x,self.ball.y),radius=self.ball.radius,color= (255, 0, 0))
        pygame.draw.polygon(canvas, (255, 255, 255),self.player.figure())
        for paddle in self.padels:
            if paddle.type == "bot":
                pygame.draw.polygon(canvas, (0, 0, 255), paddle.figure())
            else:
                pygame.draw.polygon(canvas, (255, 255, 255), paddle.figure())

        for point in self.inner_bound:
            pygame.draw.circle(canvas, (255, 0, 0), point, 5)  # Red circles for points



        pygame.display.flip()


    def react_to_user_input(self, keysPressed):
        if keysPressed[pygame.K_UP]:
            print("Up")

        if keysPressed[pygame.K_DOWN]:
            print("down")

        if keysPressed[pygame.K_a]:
            self.player.move(-1,self.dt)

            # if left arrow key is pressed
        if keysPressed[pygame.K_d]:
            self.player.move(1,self.dt)

        if keysPressed[pygame.K_p]: # save module
            self.bot.save_model("module_v1.h5")




main = Main()

main.main()