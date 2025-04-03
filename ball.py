import math
class Ball():
    def __init__(self,pos,radius,paddles, speed = 0, direction = 0):
        self.x, self.y = pos
        self.radius = radius
        self.speed = speed
        self.direction = direction
        self.paddles = paddles

    def collision(self, bound): # checks collision, returns -1 for false, 0 for flor bounce, 1 for wall bounce)
        if bound[0][0] == bound[-1][0]: # when first and last have same x value, side 0 and 2
            if len(bound) ==4: # when bounds is 4
                if (max(bound[0][0],bound[1][0]) >= self.x - self.radius and self.x + self.radius >= min(bound[0][0],bound[1][0])
                and max(bound[0][1],bound[-1][0]) >= self.y - self.radius and self.x + self.radius >= min(bound[0][1],bound[-1][1])):
                    return 0
                else:
                    return -1

            else: # bound is 6, is on curve split into 2 bounds
                floor_bound = [bound[0],bound[1]]
                middel_point = (bound[1][0],bound[-2][1])
                floor_bound.append(middel_point)
                floor_bound.append(bound[-1])
                wall_bound = [middel_point,bound[2], bound[3],bound[-2]]
                return (self.collision(floor_bound), self.collision(wall_bound))

        else: #if y is the same on fist and last point,
            if len(bound) == 4:
                if (max(bound[0][0],bound[-1][0]) >= self.x - self.radius and self.x + self.radius >= min(bound[0][0],bound[-1][0])
                    and max(bound[0][1],bound[1][1]) >= self.y - self.radius and self.y + self.radius >= min(bound[0][1],bound[1][1])):
                    return 1
                else:
                    return -1

            else: # 6 point split ito 2 with 4:
                wall_bound = [bound[0],bound[1]]
                middle_point = (bound[-2][0],bound[1][1])
                wall_bound.append(middle_point)
                wall_bound.append(bound[-1])
                floor_bound = [middle_point,bound[2],bound[3],bound[-2]]
                return (self.collision(wall_bound),self.collision(floor_bound))




    def update(self, dt):
        for paddle in self.paddles:
            collisons =

        self.x += math.cos(self.direction)* self.speed * dt
        self.x += math.sin(self.direction) * self.speed * dt