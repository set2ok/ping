import math
import random
import numpy as np
import time
from datetime import datetime

class Ball():
    """
    Class representing a ball in a 2D space, capable of bouncing off rectangles.
    The ball's movement is defined by its position, direction, and speed. It can detect collisions with paddles
    and boundaries, and adjust its trajectory accordingly.
    The ball can also spawn at random positions within defined outer and inner boundaries with a direction to the center.
    """

    def __init__(self,outer_bound,inner_bound,radius,paddles, speed = 100):
        """
        Initializes the ball with given parameters.

        :param outer_bound: corners of the outer boundary rectangle.
        :param inner_bound: corners of the inner boundary rectangle.
        :param radius: The radius of the ball.
        :param paddles: List of paddles in the game.
        :param speed: The base speed of the ball.
        """
        self.radius = radius
        self.outer_bound = outer_bound
        self.inner_bound = inner_bound
        self.spawn()
        self.speed_base = speed
        self.paddles = paddles
        self.collision_list = []
        self.collision_length = 100
        self.collision_length_factor = 4
        self.max_angle_adjustment = math.pi / 6

    def clamp(self,value, min_val, max_val):
        """
        Clamps a value to be within a specified range.
        :param value: The value to be clamped.
        :param min_val: The minimum value of the range.
        :param max_val: The maximum value of the range.
        """
        return max(min_val, min(value, max_val))

    def direction_centeration(self):
        """
        Ensures the ball's direction is within the range of 0 to 2π radians.
        """
        self.direction %= 2 * math.pi

    def check_collision(self, bound):
        """
        Checks for collisions between the ball and a given boundary.

        :param bound: Coordinates defining the boundary (can include curved or rectangular edges).
        :return: A list indicating collision details:
            - [-1]: No collision.
            - [0, bound]: Collision with an x-axis boundary.
            - [1, bound]: Collision with a y-axis boundary.
            - [2, bound]: Special edge case collision with x-axis.
            - [3, bound]: Special edge case collision with y-axis.
        """
        if len(bound) ==4: # when bounds is 4
            # check if inside bound
            max_x, min_x = max(x[0] for x in bound), min(x[0] for x in bound)
            max_y, min_y = max(y[1] for y in bound), min(y[1] for y in bound)
            if (max_x >= self.x - self.radius and self.x + self.radius >= min_x
            and max_y >= self.y - self.radius and self.y + self.radius >= min_y):
                if bound[0][0] == bound[-1][0] and (max_y -(max_y-min_y)/3 >= self.y  >= min_y + (max_y-min_y)/3): # edge case x axies
                    return [2, bound]
                elif bound[0][1] == bound[-1][1] and (max_x  -(max_x-min_x)/3 >= self.x >= min_x + (max_x-min_x)/3): #edge case y axies
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

    def detect_paddle_collisions(self, paddle):
        """
        Check collisions with a given paddle and formats the collision data to fit the expected structure.

        :param paddle: The paddle to check collisions with.
        :return: A list of collision results for the given paddle and if needed, the paddle's bounds.
        """
        bounds = paddle.figure
        collision = self.check_collision(bounds)

        # Ensure collision data is properly formatted
        if len(collision) == 1 or (len(collision) == 2 and len(collision[1]) == 4):
            collision = [collision]
        elif len(collision) == 2:
            if len(collision[0]) == 1:
                collision[0] = [collision[0]]
            if len(collision[1]) == 1:
                collision[1] = [collision[1]]

        return collision

    def handle_x_axis_collision(self, bound):
        """
        Handles collisions with the x-axis of the paddle and adjusts the ball's direction.
        :param bound: The collision data containing the paddle's bounds.
        """
        self.direction = -self.direction
        self.direction_centeration()

        dx = (self.x - ((bound[0][0] + bound[1][0]) / 2))
        width = abs(bound[0][0] - bound[1][0])
        offset = self.clamp(dx / (width / 2), -1, 1)

        if 0 < self.direction < math.pi:
            self.direction -= offset * self.max_angle_adjustment
        else:
            self.direction += offset * self.max_angle_adjustment
    def handle_y_axis_collision(self, bound):
        """
        Handles collisions with the y-axis of the paddle and adjusts the ball's direction.
        :param bound: The collision data containing the paddle's bounds.
        """
        self.direction = math.pi - self.direction
        self.direction_centeration()

        dy = (self.y - ((bound[0][1] + bound[1][1]) / 2))
        height = abs(bound[0][1] - bound[1][1])
        offset = self.clamp(dy / (height / 2), -1, 1)

        if math.pi / 2 < self.direction < 3 * math.pi / 2:
            self.direction -= offset * self.max_angle_adjustment
        else:
            self.direction += offset * self.max_angle_adjustment

    def handle_x_edge_collision(self, bound):
        """
        Handles collisions with the x-axis edges of the paddle and adjusts the ball's direction.
        :param bound: The collision data containing the paddle's bounds.
        """
        if abs(min(x[0] for x in bound) - self.x) < abs(max(x[0] for x in bound) - self.x):
            self.x = min(x[0] for x in bound) - self.radius
            if math.cos(self.direction) > 0:
                self.direction = math.pi - self.direction
        else:
            self.x = max(x[0] for x in bound) + self.radius
            if math.cos(self.direction) < 0:
                self.direction = math.pi - self.direction

    def handle_y_edge_collision(self, bound):
        """
        Handles collisions with the y-axis edges of the paddle and adjusts the ball's direction.
        :param bound: The collision data containing the paddle's bounds.
        """
        if abs(min(y[1] for y in bound) - self.y) < abs(max(y[1] for y in bound) - self.y):
            self.y = min(y[1] for y in bound) - self.radius
            if math.sin(self.direction) < 0:
                self.direction = -self.direction
        else:
            self.y = max(y[1] for y in bound) + self.radius
            if math.sin(self.direction) > 0:
                self.direction = -self.direction

    def collision(self,bot):
        """
        Manages collisions with paddles and applies bounce logic.

        - Adjusts the ball's direction based on the point of impact.
        - Notifies the bot about whether a collision occurred for potential learning updates.

        :param bot: The AI bot or controlling agent managing the paddles.
        """
        for paddle in self.paddles:
            for col in self.detect_paddle_collisions(paddle):
                current_colision_list = [item for item in self.collision_list]
                if not (2 or 3 in current_colision_list):
                    current_colision_list[len(current_colision_list)*((self.collision_length_factor-1)/self.collision_length_factor):-1]

                if not col[0] == -1 and not col[0] in current_colision_list and paddle.type == "bot":
                    bot.adjust_weights_for_result(True)
                if col[0] == -1  or col[0] in current_colision_list:
                    pass

                elif col[0] == 0 and 2 not in current_colision_list:
                    self.handle_x_axis_collision(col[1])

                elif col[0] == 1 and 3 not in current_colision_list:
                    self.handle_y_axis_collision(col[1])

                elif col[0] == 2 :
                    self.handle_x_edge_collision(col[1])

                elif col[0] == 3 :
                    self.handle_y_edge_collision(col[1])

                self.collision_list.append(col[0])
                if len(self.collision_list) >self.collision_length*self.collision_length_factor:
                    self.collision_list.pop(0)


    def update(self, dt, bot):
        """
        Updates the ball's position and handles collisions.

        - Applies motion using the current direction and speed.
        - Handles cases where the ball moves out of the defined boundaries.
        -

        :param dt: Delta time since the last frame.
        :param bot: The AI bot or controlling agent to update based on game outcomes.
        :return: Returns the points that has been earned
        """
        self.collision(bot)

        self.direction_centeration()

        self.speed = self.speed_base
        if not self.grace_period == False:
            self.speed = 0.2* self.speed_base
            if (datetime.now() - self.grace_period).total_seconds() >= random.uniform(0.5, 2.5):
                self.grace_period = False

        self.x += math.cos(self.direction)* self.speed * dt
        self.y += math.sin(self.direction) * self.speed * dt

        if self.point_inside_bound(self.outer_bound,self.x,self.y) == False:
            bot.adjust_weights_for_result(False, self.minimum_distance_to_paddle())
            self.spawn()
            return 1
        if self.point_inside_bound(self.inner_bound,self.x,self.y) == True:
            self.spawn()
            return -1
        return 0


    def point_inside_bound(self, boundaries,x,y):
        """
        Checks if the ball is within the defined boundaries.

        :param boundaries: The boundaries to check against.
        :return: True if the ball is within the boundaries, False otherwise.
        """
        return (boundaries[0][0] <= x  <= boundaries[1][0] and
                boundaries[0][1] <= y  <= boundaries[2][1])


    def spawn(self):
        """
        Resets and repositions the ball randomly within the defined outer boundaries.

        - Ensures a valid starting position that respects both the inner and outer boundaries.
        - Resets the ball's direction to the center and starts the grace period.

        :return: None.
        """

        # cordinates
        self.x = random.uniform(self.outer_bound[0][0] + self.radius * 2, self.outer_bound[1][0] -self.radius * 2)
        if self.inner_bound[0][0] - self.radius < self.x  < self.inner_bound[1][0] + self.radius:
            range = random.choice([(self.outer_bound[0][1] + self.radius * 2,self.inner_bound[0][1] - self.radius * 2),
                                   (self.inner_bound[2][1] + self.radius * 2,self.outer_bound[2][1] - self.radius * 2)])
            self.y = random.uniform(range[0],range[1])
        else:
            self.y = random.uniform(self.outer_bound[0][1] + self.radius * 2,self.outer_bound[3][1] - self.radius * 2)

        self.start_direction()
        self.grace_period = datetime.now()


    def start_direction(self):
        """
        Sets the ball's initial direction towards the center of the inner boundary.

        :return: None.
        """
        center = self.rectangle_center(self.inner_bound)
        dx = center[0] - self.x
        dy = center[1] - self.y
        self.direction = math.atan2(dy, dx)

    def rectangle_center(self,corners):
        """
        Calculates the center of a rectangle given its corner coordinates.

        :param corners: A list of coordinates defining the corners of a rectangle.
        :return: A tuple (center_x, center_y) representing the rectangle's center cordinates.
        """
        x_vals = [p[0] for p in corners]
        y_vals = [p[1] for p in corners]
        center_x = sum(x_vals) / 4
        center_y = sum(y_vals) / 4
        return center_x, center_y

    def next_colision(self):
        """
        Determines the next collision point along the ball's current trajectory.
        by using the line intersection method. With known constant x or y coordinate.

        :return: A tuple representing the x, y coordinates of the next collision point.
        """
        k = math.tan(self.direction)
        m = self.y - (self.x * k)
        dx, dy = math.cos(self.direction), math.sin(self.direction)
        point_list = []
        for side in range(len(self.outer_bound)):
            if side == 0 or side == 2:
                point_outer = self.line_intresect(k,m, y = self.outer_bound[side][1])
                point_inner = self.line_intresect(k, m, y=self.inner_bound[side][1])
            else:
                point_outer = self.line_intresect(k,m, x = self.outer_bound[side][0])
                point_inner = self.line_intresect(k, m, x=self.inner_bound[side][0])
            point_on_bound = []
            if self.point_inside_bound(self.inner_bound,point_inner[0],point_inner[1]) == True:
                point_on_bound.append(point_inner)
            if self.point_inside_bound(self.outer_bound,point_outer[0],point_outer[1]) == True:
                point_on_bound.append(point_outer)

            for point in point_on_bound:
                if point is None:
                    continue

                vx, vy = point[0] - self.x, point[1] - self.y  # vektor till punkt
                dot = vx * dx + vy * dy  # skalärprodukt

                if dot > 0:
                    point_list.append(point)

        if len(point_list) == 0:
            return (self.x, self.y)

        closest_point = min(point_list, key=lambda p: (p[0] - self.x) ** 2 + (p[1] - self.y) ** 2)
        return closest_point



    def line_intresect(self,k,m, x = False, y = False):
        """
        Finds the intersection point between the ball's trajectory (line) and a given boundary.

        :param k: The slope of the trajectory.
        :param m: The y-intercept of the trajectory.
        :param x: (Optional) The x-coordinate for a vertical boundary line.
        :param y: (Optional) The y-coordinate for a horizontal boundary line.
        :return: Tuple (x, y) representing the intersection point.
        """
        if x:
            y = k*x + m
        else:
            x = (y - m)/k

        return(x,y)

    def minimum_distance_to_paddle(self):
        """
        Calculates the minimum distance between the ball and the closest paddle.

        - Accounts for boundaries and the ball's radius.

        :return: The minimum distance as a positive float.
        """
        distances = []
        for paddle in self.paddles:
            if paddle.type == "bot":
                bound = paddle.figure
                x_values = [p[0] for p in bound]
                y_values = [p[1] for p in bound]
                if max(x_values) < self.x < min(x_values):
                    distances.append(min([abs(min(y_values) - self.y), abs(max(y_values) - self.y)]) - self.radius)
                if max(y_values) < self.y < min(y_values):
                    distances.append(min([abs(min(x_values) - self.x), abs(max(x_values) - self.x)]) - self.radius)

                point = np.array([self.x,self.y])

                bounds_points = np.array([bound])

                distance =  np.linalg.norm(bounds_points - point, axis=1)
                distances.append(np.min(distance) - self.radius)

        return abs(min(distances))
