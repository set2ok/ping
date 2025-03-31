import math
class Ball():
    def __init__(self,pos,radius, speed = 0, direction = 0):
        self.x, self.y = pos
        self.radius = radius
        self.speed = speed
        self.direction = direction

    def colision(self, objects):
        for obj in objects:
            if (self.x + self.radius) >= obj.x + obj.lenght/2 and obj.x - obj.lenght/2 >= (self.x - self.radius) and (
                    self.y + self.radius) >= obj.y >= (self.x - self.radius):
                pass


    def update(self, dt):
        self.x += math.cos(self.direction)* self.speed * dt
        self.x += math.sin(self.direction) * self.speed * dt