import math
class Ball():
    def __init__(self,pos,radius,paddles, speed = 100, direction = -math.pi/3):
        self.x, self.y = pos
        self.radius = radius
        self.speed = speed
        self.direction = direction
        self.paddles = paddles
        self.last_change = -1

    def collision(self, bound): # checks collision, returns -1 for false, 0 for flor bounce, 1 for wall bounce)
        # returns -1, for no collision, 0 för collision on the x axies, 1 for y axies
        if len(bound) ==4: # when bounds is 4
            # check if inside bound
            if (max(x[0] for x in bound) >= self.x - self.radius and self.x + self.radius >= min(x[0] for x in bound)
            and max(y[1] for y in bound) >= self.y - self.radius and self.y + self.radius >= min(y[1] for y in bound)):
                print("in")
                if (max(x[0] for x in bound) >= self.x >= min(x[0] for x in bound)):
                    return 2
                elif (max(y[1] for y in bound) >= self.y >= min(y[1] for y in bound)):
                    return 3
                if bound[0][0] == bound[-1][0]:
                    return 0
                else:
                    return 1
            else:
                return -1

        else: # bound is 6, is on curve split into 2 bounds of 4
            if bound[0][0] == bound[-1][0]: #
                floor_bound = [bound[0],bound[1]]
                middel_point = (bound[1][0],bound[-2][1])
                floor_bound.append(middel_point)
                floor_bound.append(bound[-1])
                wall_bound = [middel_point,bound[2], bound[3],bound[-2]]
                return [self.collision(floor_bound), self.collision(wall_bound)]

            else: # 6 point split ito 2 with 4:
                wall_bound = [bound[0],bound[1]]
                middle_point = (bound[-2][0],bound[1][1])
                wall_bound.append(middle_point)
                wall_bound.append(bound[-1])
                floor_bound = [middle_point,bound[2],bound[3],bound[-2]]
                return [self.collision(wall_bound),self.collision(floor_bound)]




    def update(self, dt):
        for paddle in self.paddles:
            colisions = self.collision(paddle.figure())
            for col in [colisions] if type(colisions) == int else [colisions[0],colisions[1]]:
                if col != -1:
                    print(col)
                if col == 0 and not col == self.last_change:
                    self.direction = - self.direction
                elif col == 1 and not col == self.last_change:
                    self.direction =  math.pi - self.direction
                elif col == 2:
                    pass
                self.last_change = col

        self.x += math.cos(self.direction)* self.speed * dt
        self.y += math.sin(self.direction) * self.speed * dt