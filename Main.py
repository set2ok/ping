import pygame
from paddle import Paddle
from ball import Ball
from bot import Bot
class Main:

    def __init__(self):
        self.window_size = 500
        self.exit = False
        self.paddle_height = 50
        self.outer_bound = [(self.paddle_height,self.paddle_height), (self.window_size - self.paddle_height, self.paddle_height),
                            (self.window_size - self.paddle_height,self.window_size - self.paddle_height),
                            (self.paddle_height, self.window_size - self.paddle_height)]

    # Will initialise the beginning of the game, create all essential objects etc.
    def setup(self):
        self.player = Paddle((50,200), 150 ,25,self.outer_bound, 500)
        self.paddle = Paddle((200,50),100,25,self.outer_bound,300)
        self.ball = Ball((50,300),10, [self.player])
        self.bot = Bot(self.player,[self.paddle], [self.ball])
    def main(self):

        clock = pygame.time.Clock()
        pygame.init()

        # CREATE A CANVAS
        canvas = pygame.display.set_mode((self.window_size, self.window_size))

        # TITLE OF CANVAS
        pygame.display.set_caption("Retro game")

        # SETUP GAME OBJECTS
        self.setup()

        # GAME LOOP
        while not self.exit:
            self.draw(canvas)

            self.handle_events()

            pygame.display.update()
            self.dt = clock.tick(800) / 1000
            self.ball.update(self.dt)
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
        p_figure = self.player.figure()
        pygame.draw.circle(canvas,center=(self.ball.x,self.ball.y),radius=self.ball.radius,color= (255, 0, 0))
        pygame.draw.polygon(canvas, (255, 255, 255), p_figure )


        for point in self.outer_bound:
            pygame.draw.circle(canvas, (255, 0, 0), point, 5)  # Red circles for points

        for point in p_figure:
            pygame.draw.circle(canvas, (0, 255, 0),point, 5)  # Red circles for points


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



main = Main()

main.main()