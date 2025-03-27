class Ball():
    def __init__(self,pos,radius, speed = 0, direction = 0):
        self.x, self.y = pos
        self.radius = radius
        self.speed = speed
        direction = direction