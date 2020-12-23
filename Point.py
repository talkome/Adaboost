class Point:
    def __init__(self, x, y, point_type):
        self.x = x
        self.y = y
        self.type = point_type

    def print_p(self):
        print("x:", self.x, ", y:", self.y, ", type:", self.type)

    def to_string(self):
        s = "x:"
        s += str(self.x)
        s += ", y:"
        s += str(self.y)
        s += ", type:"
        s += str(self.type)
        s += ")"
        return s
