import sys
sys.path.append("../")
from utils import *

class SpeedDistanceEstimator:

    def __init__(self):
        self.batch_size = 5
        self.frame_rate = 24

    def estimate_dist_and_speed(self, tracks):
        
        total_distance = {}
        for object, object_track in tracks.items():

            if object == "ball" or object == "referees":
                continue
            n_frames = len(object_track)
            for frame_num in range(0, n_frames, self.batch_size):
                last_frame = min(frame_num + self.batch_size, n_frames-1)

                for track_id, _ in object_track[frame_num].items():
                    if track_id not in object_track[last_frame]:
                        continue
                        
                    start = object_track[frame_num][track_id]["tranformed_position"]
                    end = object_track[last_frame][track_id]["tranformed_position"]

                    if start is None or end is None:
                        continue
                    
                    dist_covered = measure_distance(start, end)
                    time = (last_frame - frame_num) / self.frame_rate
                    speedMPH = dist_covered / time
                    speedKmPH = speedMPH * 3.6

                    if object not in total_distance:
                        total_distance[object] = {}
                    
                    if track_id not in total_distance[object]:
                        total_distance[object][track_id] = 0

                    total_distance[object][track_id] += dist_covered

                    for batch in range(frame_num, last_frame):
                        if track_id not in tracks[object][batch]:
                            continue
                        tracks[object][batch][track_id]["speed"] = speedKmPH
                        tracks[object][batch][track_id]["distance"] = total_distance[object][track_id]

    def draw_annotations(self, frames, tracks):
        op_frames = []

        for i, frame in enumerate(frames):
            for object, object_track in tracks.items():
                if object == "ball" or object == "referees":
                    continue

                for track_id, track_details in object_track[i].items():
                    if "speed" in track_details:
                        speed = track_details.get("speed", None)
                        distance = track_details.get("distance", None)
                    
                        if speed is None or distance is None:
                            continue

                        bbox = track_details["bbox"]
                        position = foot_position(bbox)
                        position = list(position)
                        position[1] += 40

                        position = tuple(map(int, position))
                        cv2.putText(frame, f"{speed:.2f}km/h", position, cv2.FONT_HERSHEY_DUPLEX, 0.6, (0,0,0), 2)
                        cv2.putText(frame, f"{distance:.2f}m", (position[0], position[1]+25), cv2.FONT_HERSHEY_DUPLEX, 0.5, tuple(map(int, track_details.get("jersey_colour", (0,0,0)))), 1)

            op_frames.append(frame)
        return op_frames
