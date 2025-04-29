import math
import random
import numpy as np

class Ball():
    def __init__(self,outer_bound,inner_bound,radius,paddles, speed = 200, direction = -math.pi/3):
        self.radius = radius
        self.outer_bound = outer_bound
        self.inner_bound = inner_bound
        self.spawn()
        self.speed = speed
        self.paddles = paddles
        self.last_change = -1
        self.collision_list = []
        self.collision_length = 100
        self.collision_length_factor = 4
        self.max_angle_adjustment = math.pi / 6

    def clamp(self,value, min_val, max_val):
        return max(min_val, min(value, max_val))

    def direction_centeration(self):
        self.direction %= 2 * math.pi

    def check_collision(self, bound): # checks collision, returns -1 for false, 0 for flor bounce, 1 for wall bounce)
        # returns -1, for no collision, 0 för collision on the x axies, 1 for y axies
        if len(bound) ==4: # when bounds is 4
            # check if inside bound
            if (max(x[0] for x in bound) >= self.x - self.radius and self.x + self.radius >= min(x[0] for x in bound)
            and max(y[1] for y in bound) >= self.y - self.radius and self.y + self.radius >= min(y[1] for y in bound)):
                if bound[0][0] == bound[-1][0] and (max(y[1] for y in bound) >= self.y  >= min(y[1] for y in bound)): # edge case x axies
                    return [2, bound]
                elif bound[0][1] == bound[-1][1] and (max(x[0] for x in bound) >= self.x >= min(x[0] for x in bound)): #edge case y axies
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
                return [self.check_collision(floor_bound), self.check_collision(wall_bound)]

            else:
                wall_bound = [bound[0],bound[1]]
                middle_point = (bound[-2][0],bound[1][1])
                wall_bound.append(middle_point)
                wall_bound.append(bound[-1])
                floor_bound = [middle_point,bound[2],bound[3],bound[-2]]
                return [self.check_collision(wall_bound),self.check_collision(floor_bound)]

    def collision(self,bot):
        for paddle in self.paddles:
            bounds = paddle.figure()
            colision = self.check_collision(bounds)
            if len(colision) == 1 or len(colision) == 2 and len(colision[1]) == 4:
                colision = [colision]
            elif len(colision) == 2:
                if len(colision[0]) == 1:
                    colision[0] = [colision[0]]
                elif len(colision[1]) == 1:
                    colision[1] = [colision[1]]



            for col in colision:
                current_colision_list = [item for item in self.collision_list]
                if not (2 or 3 in current_colision_list):
                    current_colision_list[len(current_colision_list)*((self.collision_length_factor-1)/self.collision_length_factor):-1]

                if not col[0] == -1 and not col[0] in current_colision_list and paddle.type == "bot":
                    bot.adjust_weights_for_result(True)
                if col[0] == -1  or col[0] in current_colision_list:
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

                elif col[0] == 2 :
                    if abs(min(x[0] for x in col[1]) - self.x) < abs(max(x[0] for x in col[1]) - self.x):
                        self.x = min(x[0] for x in col[1]) - self.radius
                        if math.cos(self.direction) >0:
                            self.direction = math.pi - self.direction
                    else:
                        self.x = max(x[0] for x in col[1]) + self.radius
                        if math.cos(self.direction) < 0:
                            self.direction = math.pi - self.direction


                elif col[0] == 3 :
                    if abs(min(y[1] for y in col[1]) - self.y) < abs(max(y[1] for y in col[1]) - self.y):
                        self.y = min(y[1] for y in col[1]) - self.radius
                        if math.sin(self.direction) <0:
                            self.direction = - self.direction
                    else:
                        self.y = max(y[1] for y in col[1]) + self.radius
                        if math.sin(self.direction) >0:
                            self.direction = - self.direction

                self.collision_list.append(col[0])
                if len(self.collision_list) >self.collision_length*self.collision_length_factor:
                    self.collision_list.pop(0)


    def update(self, dt, bot):
        self.collision(bot)

        self.direction_centeration()
        self.x += math.cos(self.direction)* self.speed * dt
        self.y += math.sin(self.direction) * self.speed * dt

        if (self.outer_bound[3][1] <self.y or self.y < 0 or self.outer_bound[1][0] <self.x
            or self.x < 0):
            bot.adjust_weights_for_result(False)
            self.spawn()

    def spawn(self):

        # cordinates
        self.x = random.uniform(self.outer_bound[0][0] + self.radius * 2, self.outer_bound[1][0] -self.radius * 2)
        if self.inner_bound[0][0] - self.radius < self.x  < self.inner_bound[1][0] + self.radius:
            range = random.choice([(self.outer_bound[0][1] + self.radius * 2,self.inner_bound[0][1] - self.radius * 2),
                                   (self.inner_bound[2][1] + self.radius * 2,self.outer_bound[2][1] - self.radius * 2)])
            self.y = random.uniform(range[0],range[1])
        else:
            self.y = random.uniform(self.outer_bound[0][1] + self.radius * 2,self.outer_bound[3][1] - self.radius * 2)

        self.start_direction()


    def start_direction(self):
        center = self.rectangle_center(self.inner_bound)
        dx = center[0] - self.x
        dy = center[1] - self.y
        self.direction = math.atan2(dy, dx)

    def rectangle_center(self,corners):
        x_vals = [p[0] for p in corners]
        y_vals = [p[1] for p in corners]
        center_x = sum(x_vals) / 4
        center_y = sum(y_vals) / 4
        return center_x, center_y

    def minimum_distance_to_paddle(self):
        distances = []
        for paddle in self.paddles:
            if paddle.type == "bot":
                bound = paddle.figure()
                x_values = [p[0] for p in bound]
                y_values = [p[1] for p in bound]
                if max(x_values) < self.x < min(x_values):
                    distances.append(min([abs(min(y_values) - self.y), abs(max(y_values) - self.y)]) - self.radius)
                elif max(y_values) < self.y < min(y_values):
                    distances.append(min([abs(min(x_values) - self.x), abs(max(x_values) - self.x)]) - self.radius)
                else:

                    point = np.array([self.x,self.y])

                    bounds_points = np.array([bound])

                    distance =  np.linalg.norm(bounds_points - point, axis=1)
                    distances.append(np.min(distance) - self.radius)

        return abs(min(distances))
