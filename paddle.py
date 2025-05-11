import random


class Paddle():
    """
    Paddle class for a game, representing a paddle object with movement and collision detection.
    The paddle can be moved in a 2D space and is constrained within a rectangular boundary.
    """

    def __init__(self, width, height, bound,speed,type = "bot"):
        """
        Initialize the Paddle object.
        :param width: Width of the paddle.
        :param height: Height of the paddle.
        :param bound: Boundary of the paddle in the form of a list of tuples representing the corners of the rectangle.
        :param speed: Speed of the paddle movement.
        :param type: Type of the paddle, either "bot" or "player".
        """
        self.speed = speed
        self.width = width
        self.height = height
        self.type = type # bot or player
        self.bound = bound # [(x,y) * 4] corners rectangle, 0 top right, 1 top left, 2 bottom right,
                            # 3 bottom left
        self.spawn()# creates cordinates

    #create the outer corners of the paddle
    def create_figure(self):
        """
        Create the figure of the paddle based on its position and dimensions.
        The figure is represented as a list of tuples, each tuple representing a corner of the paddle.
        """
        side = self.side_on_rectangle()
        if type(side) == bool:
            print(side)
            side = 0

        n_side = (side + 1) % 4 # next side/ point,

        if side == 0 or side == 2: # upp and down

            return self.figure_x_axies(side,n_side)

        elif side == 1 or side == 3: # left and right

            return self.figure_y_axies(side,n_side)


    def figure_y_axies(self,side,n_side):
        """
        Create the figure of the paddle.
        This is specifically for when the paddle is aligned vertically.
        :param side: The side of the rectangle where the paddle is located.
        :param n_side: The next side of the rectangle.
        :return: A list of tuples representing the corners of the paddle.
        """
        figure = []
        a = 1
        if side == 3:
            a = -1
        if self.is_point_on_line(self.x, self.y + a * self.width, self.bound[side][0], self.bound[side][1],
                                 self.bound[n_side][0], self.bound[n_side][1] + a * self.height / 2):  # check if curve
            figure.append((self.x + a * self.height / 2, self.y))  # 1 corner
            figure.append((self.x + a * self.height / 2, self.y + a * self.width))
            figure.append((self.x - a * self.height / 2, self.y + a * self.width))

        else:  # if curved
            if abs(self.y - self.bound[n_side][0]) <= self.height / 2:  #
                figure.append((self.x + a * self.height / 2 - a * (
                            self.height / 2 - abs(self.y - self.bound[n_side][1])),
                            self.bound[n_side][1] - a * self.height / 2))
                figure.append((self.x + a * self.height / 2 - a * (
                            self.height / 2 - abs(self.y - self.bound[n_side][1])),
                            self.bound[n_side][1] + a * self.height / 2))  # outer corner

            else:
                figure.append((self.x + a * self.height / 2, self.y))  # 1 corner
                figure.append((self.bound[n_side][0] + a * self.height / 2,
                               self.bound[n_side][1] + a * self.height / 2))  # outer corner
            figure.append((self.bound[n_side][0] - a * (self.width - abs(self.bound[n_side][0] - self.y)),
                           self.bound[n_side][1] + a * self.height / 2))
            figure.append((self.bound[n_side][0] - a * (self.width - abs(self.bound[n_side][0] - self.y)),
                           self.bound[n_side][1] - a * self.height / 2))
            if not abs(self.y - self.bound[n_side][0]) <= self.height / 2:
                figure.append((self.bound[n_side][0] - a * self.height / 2,
                               self.bound[n_side][1] - a * self.height / 2))  # inner corner

        if not abs(self.y - self.bound[n_side][0]) <= self.height / 2:
            figure.append((self.x - a * self.height / 2, self.y))  # last corner

        return figure

    def figure_x_axies(self,side,n_side):
        """
        Create the figure of the paddle.
        This is specifically for when the paddle is aligned horizontally.
        :param side: The side of the rectangle where the paddle is located.
        :param n_side: The next side of the rectangle.
        :return: A list of tuples representing the corners of the paddle.
        """
        figure = []
        a = 1
        if side == 2:
            a = -1
        if self.is_point_on_line(self.x + a * self.width, self.y, self.bound[side][0], self.bound[side][1],
                                 self.bound[n_side][0] + a * self.height / 2, self.bound[n_side][1]):  # check if curve
            figure.append((self.x, self.y - a * self.height / 2))  # 1 corner
            figure.append((self.x + a * self.width, self.y - a * self.height / 2))
            figure.append((self.x + a * self.width, self.y + a * self.height / 2))

        else:  # if curved
            if abs(self.x - self.bound[n_side][0]) <= self.height / 2:  # smothe transiton
                figure.append((self.bound[n_side][0] - a * self.height / 2, self.y - a * self.height / 2 + a * (
                            self.height / 2 - abs(self.x - self.bound[n_side][0]))))  # first corner
                figure.append((self.bound[n_side][0] + a * self.height / 2, self.y - a * self.height / 2 + a * (
                            self.height / 2 - abs(self.x - self.bound[n_side][0]))))  # outer corner
            else:
                figure.append((self.x, self.y - a * self.height / 2))  # 1 corner
                figure.append(
                    (self.bound[n_side][0] + a * self.height / 2, self.y - a * self.height / 2))  # outer corner
            figure.append((self.bound[n_side][0] + a * self.height / 2,
                           self.bound[n_side][1] + a * (self.width - abs(self.bound[n_side][0] - self.x))))
            figure.append((self.bound[n_side][0] - a * self.height / 2,
                           self.bound[n_side][1] + a * (self.width - abs(self.bound[n_side][0] - self.x))))
            if not abs(self.x - self.bound[n_side][0]) <= self.height / 2:
                figure.append((self.bound[n_side][0] - a * self.height / 2,
                               self.bound[n_side][1] + a * self.height / 2))  # inner corner

        if not abs(self.x - self.bound[n_side][0]) <= self.height / 2:
            figure.append((self.x, self.y + a * self.height / 2,))  # last corner
        return figure

    def side_on_rectangle(self):
        """
        Determine which side of the rectangle the paddle is currently on.
        :return: The index of the side (0-3).
        """
        for i in range(4):
            x1, y1 = self.bound[i]
            x2, y2 = self.bound[(i + 1) % 4]  # Wrap around to form edges
            if self.is_point_on_line(self.x, self.y, x1, y1, x2, y2):
                return i
        return 0

    # check if a point is on a line segment
    def is_point_on_line(self,x, y, x1, y1, x2, y2):
        """
        Check if a point (x, y) is on the line segment defined by (x1, y1) and (x2, y2).
        This function uses the cross product to check for collinearity and then checks if the point is within the segment bounds.
        :param x: x-coordinate of the point to check.
        :param y: y-coordinate of the point to check.
        :param x1: x-coordinate of the first endpoint of the line segment.
        :param y1: y-coordinate of the first endpoint of the line segment.
        :param x2: x-coordinate of the second endpoint of the line segment.
        :param y2: y-coordinate of the second endpoint of the line segment.
        :return: True if the point is on the line segment, otherwise False.
        """
        # Check collinearity
        if (y - y1) * (x2 - x1) != (x - x1) * (y2 - y1):
            return False
        # Check if point is within segment bounds
        return min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2)

    def clamp_to_edge(self):
        """
        Clamp the paddle's position to the nearest edge of the rectangle.
        This is done by projecting the paddle's position onto the line segment defined by the rectangle's edges.
        change the x and y coordinates of the paddle to be within the rectangle's bounds.
        """
        side = self.side_on_rectangle()
        if side is not False:
            x1, y1 = self.bound[side]
            x2, y2 = self.bound[(side + 1) % 4]

            dx, dy = x2 - x1, y2 - y1
            t = max(0, min(1, ((self.x - x1) * dx + (self.y - y1) * dy) / (dx * dx + dy * dy)))
            self.x = x1 + t * dx
            self.y = y1 + t * dy

    def move(self,direction,dt):
        """
        Move the paddle in the specified direction.
        The direction is determined by the side of the rectangle the paddle is on.
        The paddle's position is updated based on its speed and the time delta (dt).
        creates a new figure for the paddle after moving.
        :param direction: an integer representing the direction of movement in radians.
        :param dt: Time delta, the time elapsed since the last update.
        """
        side = self.side_on_rectangle()
        n_side =  (side + 1) % 4 # next side/ point,
        a = 1
        if side >= 2:
            a = -1
        dist = direction * dt * self.speed * a
        if side == 0 or side == 2:
            if not self.is_point_on_line(self.x + dist ,self.y,self.bound[side][0],self.bound[side][1],self.bound[n_side][0],self.bound[n_side][1]): # check if outside of bounds
                if abs(self.bound[side][0]- (self.x + dist)) < abs(self.bound[n_side][0]- (self.x + dist)): # check closest point
                    self.y = self.bound[side][1] + a*abs(self.bound[side][1]- (self.y + dist))
                    self.x = self.bound[side][0]

                else:
                    self.y = self.bound[n_side][1] + a*abs(self.bound[n_side][1]- (self.y + dist))
                    self.x = self.bound[n_side][0]
            else:
                self.x += dist

        if side == 1 or side == 3:
            if not self.is_point_on_line(self.x,self.y + dist ,self.bound[side][0],self.bound[side][1],self.bound[n_side][0],self.bound[n_side][1]): # check if outside of bounds
                if abs(self.bound[side][1] - (self.y + dist)) < abs(self.bound[n_side][1] - (self.y + dist)): # check closest point
                    self.y = self.bound[side][1]
                    self.x = self.bound[side][0] - a*abs(self.bound[side][0] - (self.x + dist))
                else:
                    self.y = self.bound[n_side][1]
                    self.x = self.bound[n_side][0] - a*abs(self.bound[n_side][0] - (self.x + dist))
            else:
                self.y += dist
        self.clamp_to_edge()
        self.figure = self.create_figure()

    def spawn(self):
        """
        Spawn the paddle at a random position within the rectangle's bounds.
        The paddle's position is determined by randomly selecting a side of the rectangle and placing the paddle
        within the bounds of that side.
        The paddle's figure is then created based on its new position.
        """
        side = random.randint(0,3)
        n_side = (side + 1) % 4
        if side == 0 or side == 2:
            self.y = self.bound[side][1]
            self.x = random.uniform(min(self.bound[side][0],self.bound[n_side][0]) + self.height ,
                                    max(self.bound[side][0],self.bound[n_side][0]) - self.height)
        else:
            self.x = self.bound[side][0]
            self.y = random.uniform(min(self.bound[side][1], self.bound[n_side][1]) + self.height,
                                    max(self.bound[side][1], self.bound[n_side][1]) - self.height)
        self.figure = self.create_figure()