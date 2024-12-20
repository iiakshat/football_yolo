import os
import sys
import pickle
import cv2
import numpy as np
import pandas as pd
import supervision as sv # type: ignore
from ultralytics import YOLO

sys.path.append("../")
from utils import *

class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()


    def predict_batch(self, frames):
        batch_size = 20
        results = []

        # Note:
        # instead of self.model.predict(), in loop below we could also use self.model.track(),
        # but we do not have sufficient data to track goalkeeper, we treat him as a player only. 
        # Once we start tracking directly using .track() we cannot achieve this functionality and
        # hence we use .predict() here, and tracker from another library: supervision.

        for i in range(0, len(frames), batch_size):
            results += self.model.predict(frames[i:i+batch_size], conf=0.1)
        return results
    

    def interpolate_ball(self, position):

        position = [pos.get(1,{}).get("bbox", []) for pos in position]
        df = pd.DataFrame(position, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()
        return [{1: {"bbox": x}} for x in df.to_numpy().tolist()]


    def ellipse(self,frame,bbox,color,track_id=None):
        y2 = int(bbox[3])
        x_center, _ = bbox_center(bbox)
        width = bbox_width(bbox)

        cv2.ellipse(
            frame,
            center=(x_center,y2),
            axes=(int(width), int(0.35*width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color = color,
            thickness=2,
            lineType=cv2.LINE_4
        )

        rectangle_width = 40
        rectangle_height=20
        x1_rect = x_center - rectangle_width//2
        x2_rect = x_center + rectangle_width//2
        y1_rect = (y2- rectangle_height//2) +15
        y2_rect = (y2+ rectangle_height//2) +15

        if track_id is not None:
            cv2.rectangle(frame,
                          (int(x1_rect),int(y1_rect) ),
                          (int(x2_rect),int(y2_rect)),
                          color,
                          cv2.FILLED)
            
            x1_text = x1_rect+12
            if track_id > 99:
                x1_text -=10
            
            cv2.putText(
                frame,
                f"{track_id}",
                (int(x1_text),int(y1_rect+15)),
                cv2.FONT_HERSHEY_DUPLEX,
                0.6,
                (0,0,0),
                2
            )

        return frame
    

    def triangle(self,frame,bbox,color):
        y= int(bbox[1])
        x,_ = bbox_center(bbox)

        triangle_points = np.array([
            [x,y],
            [x-10,y-20],
            [x+10,y-20],
        ])
        cv2.drawContours(frame, [triangle_points],0,color, cv2.FILLED)
        cv2.drawContours(frame, [triangle_points],0,(0,0,0), 2)

        return frame


    def draw_annotations(self, frames, tracks, team_ball_control, 
                         showteamcontrol, pointplayerwithball, showplayeridwithball, pointercolour):

        if not showteamcontrol and not pointplayerwithball and not showplayeridwithball:
            return frames
        
        op_frames = []
        # Draw frame by frame
        for i, frame in enumerate(frames):
            frame = frame.copy()
            players = tracks["players"][i]
            referees = tracks["referees"][i]
            ball = tracks["ball"][i]

            # Draw Player
            for track_id, player in players.items():

                color = player.get("jersey_colour", (0,0,255))
                frame = self.ellipse(frame, player["bbox"], color, track_id)

                if player.get("has_ball", False):
                    frame = self.player_with_ball(frame, track_id, showplayeridwithball)

                    if pointplayerwithball:
                        frame = self.triangle(frame, player["bbox"],pointercolour)
                else:
                    frame = self.player_with_ball(frame, None, showplayeridwithball)

            # Draw Referee
            for _, referee in referees.items():
                frame = self.ellipse(frame, referee["bbox"],(0,255,255))
            
            # Draw ball 
            for _, ball_ in ball.items():
                frame = self.triangle(frame, ball_["bbox"],(0,255,0))

            # Team Control
            if showteamcontrol:
                frame = self.team_control(frame, i, team_ball_control)
            op_frames.append(frame)

        return op_frames
    

    def track(self, frames, cache=False, path=None):

        if cache and path is not None and os.path.exists(path):
            with open(path, "rb") as f:
                tracks = pickle.load(f)
            return tracks
        
        tracks = {
            "ball": [],
            "players": [],
            "referees": []
        }

        predictions = self.predict_batch(frames)

        for i, prediction in enumerate(predictions):

            pred_classes = prediction.names                  # -> {0: "ball", 1: "player" ...}

            # Formatting:
            classes = {v:k for k, v in pred_classes.items()} # -> {"ball": 0, "player": 1 ...}
            supervision_pred = sv.Detections.from_ultralytics(prediction)

            for idx, class_id in enumerate(supervision_pred.class_id):
                if pred_classes[class_id] == "goalkeeper":
                    supervision_pred.class_id[idx] = classes["player"]

            # Track Objects:
            pred_tracks = self.tracker.update_with_detections(supervision_pred)

            tracks["ball"].append({})
            tracks["referees"].append({})
            tracks["players"].append({})

            for pred_frames in pred_tracks:
                bbox = pred_frames[0].tolist()
                cls_id = pred_frames[3]
                track_id = pred_frames[4]

                if cls_id == classes["player"]:
                    tracks["players"][i][track_id] = {"bbox":bbox}
                    
                if cls_id == classes["referee"]:
                    tracks["referees"][i][track_id] = {"bbox":bbox}

            for pred_frames in supervision_pred:
                bbox = pred_frames[0].tolist()
                cls_id = pred_frames[3]

                if cls_id == classes["ball"]:
                    tracks["ball"][i][1] = {"bbox":bbox}


        # Finally, Our tracks list for PLAYERS will look like:
        # "players" : [{7: {"bbox": [x, y, w, h]}, 8: {"bbox": [x, y, w, h]} ... },  -> Frame 1  (7 AND 8 ARE TRACKED)
        #              {7: {"bbox": [x, y, w, h]}, 8: {"bbox": [x, y, w, h]} ... },  -> Frame 2  (7 AND 8 ARE TRACKED)
        #              {8: {"bbox": [x, y, w, h]}, 1: {"bbox": [x, y, w, h]} ... }   -> Frame 3  (8 AND 1 ARE TRACKED)
        
        if path:
            with open(path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks
    

    def team_control(self, frame, frame_num, team_ball_control):

        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 950), (1850, 1070), (255,255,255), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        tillframe = team_ball_control[:frame_num+1]
        team1 = tillframe[tillframe==1].shape[0]
        team2 = tillframe[tillframe==2].shape[0]
        team_1 = team1/(team1+team2)
        team_2 = team2/(team1+team2)

        cv2.putText(frame, f"Team 1: {team_1*100:.2f}%", (1400, 1000), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 3)
        cv2.putText(frame, f"Team 2: {team_2*100:.2f}%", (1400, 1050), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 3)

        return frame
    

    def player_with_ball(self, frame, trackid, show=True):

        if not show:
            return frame
        
        if trackid is None:
            text = " No one has the ball."
        else:
            text = f"Player {trackid} has the ball."

        overlay = frame.copy()
        cv2.rectangle(overlay, (750, 980), (1300, 1050), (0,0,0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        cv2.putText(frame, text, (790, 1030), cv2.FONT_HERSHEY_DUPLEX, 1, (255,255,255), 3)

        return frame


    def handle_position(self, tracks):
        for object, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_details in track.items():
                    
                    bbox = track_details["bbox"]
                    if object == "ball":
                        position = bbox_center(bbox)
                    else:
                        position = foot_position(bbox)

                    tracks[object][frame_num][track_id]["position"] = position
                