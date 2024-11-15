import os
import sys
import numpy as np
import cv2
import pickle

sys.path.append("../")
from utils import measure_distance, measure_xy_distance

class CameraMovementEstimator:
    def __init__(self, frame):

        self.minimum_dist = 5
        first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask_features = np.zeros_like(first_frame)
        mask_features[:, 0:20] = 1
        mask_features[:, 900:1050] = 1

        self.lk_params = dict(
            winSize = (15,15),
            maxLevel = 2,
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        self.features = dict(
            maxCorners = 100,
            qualityLevel = 0.3,
            minDistance = 3,
            blockSize = 7,
            mask = mask_features
        )

    def adjust_positions(self, tracks, movement_per_frame):
         
        for object, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_details in track.items():
                    position = track_details["position"]
                    movement = movement_per_frame[frame_num]
                    position_adjusted = (position[0] - movement[0], position[1] - movement[1])
                    tracks[object][frame_num][track_id]["position_adjusted"] = position_adjusted

    def estimate_movement(self, frames, cache=False, path=None):
        
        if cache and path and os.path.exists(path):
            with open(path, 'rb') as f:
                movement = pickle.load(f)
            return movement
        
        movement = [[0,0]]* len(frames)
        gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        features = cv2.goodFeaturesToTrack(gray, **self.features)

        for frame_num in range(1, len(frames)):

            newgray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            newfeatures, status, err = cv2.calcOpticalFlowPyrLK(gray, newgray, features, None, **self.lk_params)

            max_distance = 0
            movement_x, movement_y = 0, 0
            
            for i, (new, old) in enumerate(zip(newfeatures, features)):
                
                newfeaturespoint = new.ravel()
                oldfeaturespoint = old.ravel()
                distance = measure_distance(newfeaturespoint, oldfeaturespoint)

                if distance > max_distance:
                    max_distance = distance
                    movement_x, movement_y = measure_xy_distance(oldfeaturespoint, newfeaturespoint)

            if max_distance > self.minimum_dist:
                movement[frame_num] = [movement_x, movement_y]
                features = cv2.goodFeaturesToTrack(newgray, **self.features)

            gray = newgray.copy()
        
        if path:
            with open(path, 'wb') as f:
                pickle.dump(movement, f)

        return movement

    def draw_movement(self, frames, movement_per_frame, config):

        output = []
        alpha = config["alpha"]
        for frame_num, frame in enumerate(frames):
            overlay = frame.copy()
            frame = frame.copy()
            cv2.rectangle(overlay, (20,20), (200, 95), (255,255,255), -1)
            cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)

            movement_x, movement_y = movement_per_frame[frame_num]
            frame = cv2.putText(frame, f"X: {movement_x:.2f}", (30, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 3)
            frame = cv2.putText(frame, f"Y: {movement_y:.2f}", (30, 80), cv2.FONT_HERSHEY_DUPLEX, 1, (0,0,0), 3)

            output.append(frame)
        
        return output
