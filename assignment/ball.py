import sys
sys.path.append("../")
from utils import bbox_center, measure_distance

class BallAssigner():
    def __init__(self, config):
        self.distance = config["min_distance_to_assign_ball"]

    def assign_ball(self, players, bbox):
        
        min_dist = float("inf")
        playerwithball = None
        ball = bbox_center(bbox)

        for playerid, player in players.items():
            playerbbox = player["bbox"]

            left = measure_distance((playerbbox[0], playerbbox[-1]), ball)
            right = measure_distance((playerbbox[2], playerbbox[-1]), ball)
            distance = min(left, right)
            
            if distance < self.distance:
                if distance < min_dist:
                    min_dist = distance
                    playerwithball = playerid
        
        return playerwithball