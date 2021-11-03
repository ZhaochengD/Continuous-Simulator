import math

def L2_dis(car1, car2):
    return math.sqrt((car1.x - car2.x)**2 + (car1.y - car2.y)**2)

