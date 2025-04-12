import math
class Ball():
    def __init__(self,pos,radius,paddles, speed = 200, direction = -math.pi/3):
        self.x, self.y = pos
        self.radius = radius
        self.speed = speed
        self.direction = direction
        self.paddles = paddles
        self.last_change = -1
        self.collision_list = []
        self.collision_length = 30
        self.collision_length_factor = 3
        self.max_angle_adjustment = math.pi / 6

    def clamp(self,value, min_val, max_val):
        return max(min_val, min(value, max_val))

    def direction_centeration(self):
        self.direction %= 2 * math.pi

    def collision(self, bound): # checks collision, returns -1 for false, 0 for flor bounce, 1 for wall bounce)
        # returns -1, for no collision, 0 för collision on the x axies, 1 for y axies
        if len(bound) ==4: # when bounds is 4
            # check if inside bound
            if (max(x[0] for x in bound) >= self.x - self.radius and self.x + self.radius >= min(x[0] for x in bound)
            and max(y[1] for y in bound) >= self.y - self.radius and self.y + self.radius >= min(y[1] for y in bound)):
                if bound[0][0] == bound[-1][0] and (max(y[1] for y in bound) >= self.y - self.radius/2 and self.y + self.radius/2 >= min(y[1] for y in bound)): # edge case x axies
                    return [2, bound]
                elif bound[0][1] == bound[-1][1] and (max(x[0] for x in bound) >= self.x - self.radius/2 and self.x + self.radius/2 >= min(x[0] for x in bound)): #edge case y axies
                    return [3, bound]
                elif bound[0][0] == bound[-1][0]:
                    return [0,bound]
                else:
                    return [1,bound]
            else:
                return [-1]

        else: # bound is 6, is on curve split into 2 bounds of 4
            if bound[0][0] == bound[-1][0]: #
                floor_bound = [bound[0],bound[1]]
                middel_point = (bound[1][0],bound[-2][1])
                floor_bound.append(middel_point)
                floor_bound.append(bound[-1])
                wall_bound = [middel_point,bound[2], bound[3],bound[-2]]
                return [self.collision(floor_bound), self.collision(wall_bound)]

            else:
                wall_bound = [bound[0],bound[1]]
                middle_point = (bound[-2][0],bound[1][1])
                wall_bound.append(middle_point)
                wall_bound.append(bound[-1])
                floor_bound = [middle_point,bound[2],bound[3],bound[-2]]
                return [self.collision(wall_bound),self.collision(floor_bound)]




    def update(self, dt):
        for paddle in self.paddles:
            bounds = paddle.figure()
            colision = self.collision(bounds)
            if len(colision) == 1:
                colision = [colision]
            elif len(colision) == 2:
                if len(colision[1]) == 4:
                    colision = [colision]


            for col in colision:
                current_colision_list = [item[0] for item in self.collision_list if item[1] == paddle]
                if not (2 or 3 in current_colision_list):
                    current_colision_list[len(current_colision_list)*((self.collision_length_factor-1)/self.collision_length_factor):-1]

                if col[0] == -1 or col[0] == self.last_change or col[0] in current_colision_list:
                    pass

                elif col[0] == 0 and 2 not in current_colision_list:
                    self.direction = -self.direction
                    self.direction_centeration()

                    dx = (self.x - ((col[1][0][0] + col[1][1][0]) / 2))
                    width = abs(col[1][0][0] - col[1][1][0])
                    offset = self.clamp(dx / (width / 2), -1, 1)

                    if 0 < self.direction < math.pi:
                        self.direction -= offset * self.max_angle_adjustment
                    else:
                        self.direction += offset * self.max_angle_adjustment


                elif col[0] == 1 and 3 not in current_colision_list:
                    self.direction = math.pi - self.direction
                    self.direction_centeration()

                    dy = (self.y - ((col[1][0][1] + col[1][1][1]) / 2))
                    height = abs(col[1][0][1] - col[1][1][1])
                    offset = self.clamp(dy / (height / 2), -1, 1)

                    if math.pi/2< self.direction < 3*math.pi/2:
                        self.direction -= offset * self.max_angle_adjustment
                    else:
                        self.direction += offset * self.max_angle_adjustment

                elif col[0] == 2 and not col[0] == self.last_change:
                    if abs(min(x[0] for x in col[1]) - self.x) < abs(max(x[0] for x in col[1]) - self.x):
                        self.x = min(x[0] for x in col[1]) - self.radius
                        if math.sin(self.direction) >0:
                            self.direction = math.pi - self.direction
                    else:
                        self.x = max(x[0] for x in col[1]) + self.radius
                        if math.sin(self.direction) < 0:
                            self.direction = math.pi - self.direction


                elif col[0] == 3 and not col[0] == self.last_change:
                    if abs(min(y[1] for y in col[1]) - self.y) < abs(max(y[1] for y in col[1]) - self.y):
                        self.y = min(y[1] for y in col[1]) - self.radius
                        if math.cos(self.direction) <0:
                            self.direction = - self.direction
                    else:
                        self.y = max(y[1] for y in col[1]) + self.radius
                        if math.cos(self.direction) >0:
                            self.direction = - self.direction

                self.collision_list.append([col[0],paddle])
                if len(self.collision_list) >self.collision_length*self.collision_length_factor:
                    self.collision_list.pop(0)

        self.direction_centeration()
        self.x += math.cos(self.direction)* self.speed * dt
        self.y += math.sin(self.direction) * self.speed * dt


