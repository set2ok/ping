class Paddle():

    def __init__(self,pos, width, height, bound):
        self.x,self.y = pos
        self.width = width
        self.height = height
        self.bound = bound # [(x,y) * 4]

    def bounds(self):
        bunds = []

    def point_on_rectangle(self,px, py, rect_points):
        for i in range(4):
            x1, y1 = rect_points[i]
            x2, y2 = rect_points[(i + 1) % 4]  # Wrap around to form edges
            if self.is_point_on_line(px, py, x1, y1, x2, y2):
                return True
        return False

    # Function to check if a point is on a line segment
    def point_on_line(self,px, py, x1, y1, x2, y2):
        # Check collinearity
        if (py - y1) * (x2 - x1) != (px - x1) * (y2 - y1):
            return False
        # Check if point is within segment bounds
        if x1 ==x2:
            return min(x1, x2) <= px <= max(x1, x2) and min(y1, y2) <= py <= max(y1, y2)