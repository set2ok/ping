class Paddle():

    def __init__(self,pos, width, height, bound,type = "bot"):
        self.x,self.y = pos
        self.width = width
        self.height = height
        self.type = type # bot or player
        self.bound = bound # [(x,y) * 4] corners rectangle, 0 top right, 1 top left, 2 bottom right,
                            # 3 bottom left


    def figure(self):
        figure = [] # corners
        side = self.side_on_rectangle(self.x,self.y,self.bound)
        n_side = (side + 1) % 4
        a = 1
        print(side)
        if side >1:
            a = -1
        if side == 0 or side == 2: # upp and down
            figure.append((self.x,self.y - a*self.height/2)) # 1 corner

            if self.is_point_on_line(self.x + a*self.width, self.y, self.bound[side][0],self.bound[side][1], self.bound[n_side][0],self.bound[n_side][1]): # check if curve
                figure.append((self.x + a*self.width, self.y + a*self.height / 2))
                figure.append((self.x + a*self.width, self.y - a*self.height / 2))

            else: # if curved
                figure.append((self.bound[n_side][0] + a*self.height/2 , self.y - a*self.height/2)) # outer corner
                figure.append((self.bound[n_side][0] + a*self.height/2 , self.bound[n_side][1] + a*abs(self.bound[n_side][0] - self.x) ))
                figure.append((self.bound[n_side][0] - a*self.height/2 , self.bound[n_side][1] + a*abs(self.bound[n_side][0] - self.x) ))
                figure.append((self.bound[n_side][0] - a*self.height/2 , self.bound[n_side][1] + a*self.height/2)) # inner corner

            figure.append((self.x, self.y + a*self.height / 2))

        elif side == 1 or side == 3: # left and right
            figure.append((self.x + a*self.height/2 ,self.y)) # 1 corner

            if self.is_point_on_line(self.x, self.y + a*self.width, self.bound[side][0],self.bound[side][1], self.bound[n_side][0],self.bound[n_side][1]): # check if curve
                figure.append((self.x - a * self.height / 2, self.y + a * self.width))
                figure.append((self.x + a*self.height / 2, self.y + a*self.width))

            else: # if curved
                figure.append((self.bound[n_side][0] + a*self.height/2 , self.bound[n_side][1] + a*self.height/2)) # outer corner
                figure.append((self.bound[n_side][0] - a*abs(self.bound[n_side][0] - self.y), self.bound[n_side][1] + a*self.height/2))
                figure.append((self.bound[n_side][0] - a*abs(self.bound[n_side][0] - self.y) , self.bound[n_side][1] - a*self.height/2))
                figure.append((self.bound[n_side][0] - a*self.height/2 , self.bound[n_side][1] - a*self.height/2)) # inner corner

            figure.append((self.x - a*self.height / 2, self.y))
        return figure

    # find witch side the point is on
    def side_on_rectangle(self,px , py, rect_points):
        for i in range(4):
            x1, y1 = rect_points[i]
            x2, y2 = rect_points[(i + 1) % 4]  # Wrap around to form edges
            if self.is_point_on_line(px, py, x1, y1, x2, y2):
                return i
        return False

    # Function to check if a point is on a line segment
    def is_point_on_line(self,px, py, x1, y1, x2, y2):
        # Check collinearity
        if (py - y1) * (x2 - x1) != (px - x1) * (y2 - y1):
            return False
        # Check if point is within segment bounds
        return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)