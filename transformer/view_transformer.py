import numpy as np
import cv2

class ViewTransformer:
    def __init__(self):

        court_width = 68
        court_length = 105

        # Trapezoid (in video) vertices
        self.pixel_vertices = np.array([
            [0, 1100],
            [0, 0],
            [2000, 0],
            [2000, 1100]
        ])

        # Rectangle (irl) vertices
        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width]
        ])

        self.pixel_vertices = self.pixel_vertices.astype(np.float32)
        self.target_vertices = self.target_vertices.astype(np.float32)

        self.perspective_transform = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)


    def transform_position(self, position):

        coord = (int(position[0]), int(position[1]))
        is_inside = cv2.pointPolygonTest(self.pixel_vertices, coord, False) >= 0
        if not is_inside:
            return None
        
        coords = position.reshape(-1, 1, 2).astype(np.float32)
        transformed_coords = cv2.perspectiveTransform(coords, self.perspective_transform).reshape(-1, 2)
        return transformed_coords
        

    def add_positions(self, tracks):
        for object, object_track in tracks.items():
            for frame_num, track in enumerate(object_track):
                for track_id, track_details in track.items():
                    position = track_details["position_adjusted"]
                    position = np.array(position)
                    transformed_position = self.transform_position(position)

                    if transformed_position is not None:
                        transformed_position = transformed_position.squeeze().tolist()
                    tracks[object][frame_num][track_id]["tranformed_position"] = transformed_position