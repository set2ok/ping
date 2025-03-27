import pygame
from paddle import Paddle
from ball import Ball
class Main:

    def __init__(self):
        self.window_size = 500
        self.exit = False

    # Will initialise the beginning of the game, create all essential objects etc.
    def setup(self):
        print()  # Placeholder code

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
            dt = clock.tick(120) / 1000

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
        pygame.display.flip()

    def react_to_user_input(self, keysPressed):
        if keysPressed[pygame.K_UP]:
            print("Up")

        if keysPressed[pygame.K_DOWN]:
            print("down")

        if keysPressed[pygame.K_w]:
            print("w")

            # if left arrow key is pressed
        if keysPressed[pygame.K_s]:
            print("s")



main = Main()

main.main()