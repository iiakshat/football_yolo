import os
import json
import time
import numpy as np
from trackers import Tracker
from utils import read_video, save_video
from assignment import TeamAssigner, BallAssigner
from movement_estimator import CameraMovementEstimator, SpeedDistanceEstimator
from transformer import ViewTransformer

def process(config):

    # s = time.perf_counter()

    ## Initialisation:
    model = config["model"]
    input_path = config["input_path"]
    output_path = config["output_path"]
    cache=config["cache"]
    movementpath = config["movementpath"]
    trackpath = config["trackpath"]
    alpha = config["alpha"]
    show_speed = config["show_speed"]
    show_distance = config["show_distance"]
    show_team_control = config["show_team_control"]
    show_player_with_ball = config["show_player_with_ball"]
    show_player_id_with_ball = config["show_player_id_with_ball"]

    frames = read_video(input_path)

    tracker = Tracker(model)
    tracks = tracker.track(frames,
                            cache=cache,
                            path=trackpath)
    
    tracker.handle_position(tracks)

    movement_estimator = CameraMovementEstimator(frames[0])
    movement_per_frame = movement_estimator.estimate_movement(frames, 
                                                              cache=cache, 
                                                              path=movementpath)

    movement_estimator.adjust_positions(tracks, movement_per_frame)
    
    viewtransformer = ViewTransformer(config)
    viewtransformer.add_positions(tracks)

    tracks["ball"] = tracker.interpolate_ball(tracks["ball"])

    speedDistEstimator = SpeedDistanceEstimator(config)
    speedDistEstimator.estimate_dist_and_speed(tracks)

    assigner = TeamAssigner()
    assigner.assign_colour(frames[0], tracks["players"][0])

    for frame_number, player in enumerate(tracks["players"]):
        for playerid, track in player.items():
            team = assigner.get_team(frames[frame_number], 
                                     track["bbox"],
                                     playerid)
            
            tracks["players"][frame_number][playerid]["team"] = team
            tracks["players"][frame_number][playerid]["jersey_colour"] = assigner.colours[team]

    ball_assigner = BallAssigner(config)
    team_ball = []
    for frame_number, player in enumerate(tracks["players"]):
        ballbbox = tracks["ball"][frame_number][1]["bbox"]
        assignerplayer = ball_assigner.assign_ball(player, ballbbox)

        if assignerplayer:
            tracks["players"][frame_number][assignerplayer]["has_ball"] = True
            team_ball.append(tracks["players"][frame_number][assignerplayer]["team"])
        else:
            team_ball.append(team_ball[-1])
        
    team_ball = np.array(team_ball)
    output_video  = tracker.draw_annotations(frames, tracks, team_ball, 
                                             show_team_control, 
                                             show_player_with_ball, 
                                             show_player_id_with_ball, 
                                             (0,0,255))
    output_video = movement_estimator.draw_movement(output_video, movement_per_frame, alpha)
    output_video = speedDistEstimator.draw_annotations(output_video, tracks, show_speed, show_distance, speedcolour=(0,0,0))
    
    save_video(output_video, output_path)
    # print(f"Time: {time.perf_counter() - s}")

def process2(config):
    time.sleep(3)
    
if __name__ == '__main__':

    config_path = "config.json"
    with open(config_path, 'r') as file:
        config = json.load(file)
    process(config)