import cv2
import numpy as np
from trackers import Tracker
from utils import read_video, save_video
from assignment import TeamAssigner, BallAssigner
import time

def main():

    s = time.perf_counter()
    input_path = 'matches/vid1.mp4'
    output_path = 'output/vid1.avi'
    frames = read_video(input_path)

    tracker = Tracker('models/best.pt')
    tracks = tracker.track(frames,
                            cache=True,
                            path='models/trained.pkl')
    
    tracks["ball"] = tracker.interpolate_ball(tracks["ball"])

    assigner = TeamAssigner()
    assigner.assign_colour(frames[0], tracks["players"][0])

    for frame_number, player in enumerate(tracks["players"]):
        for playerid, track in player.items():
            team = assigner.get_team(frames[frame_number], 
                                     track["bbox"],
                                     playerid)
            
            tracks["players"][frame_number][playerid]["team"] = team
            tracks["players"][frame_number][playerid]["jersey_colour"] = assigner.colours[team]

    ball_assigner = BallAssigner()
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
    result  = tracker.draw_annotations(frames, tracks, team_ball)
    save_video(result, output_path)
    print(f"Time: {time.perf_counter() - s}")

if __name__ == '__main__':
    main()