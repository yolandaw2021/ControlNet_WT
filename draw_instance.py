import json
import cv2
import numpy as np
from shapely.geometry import Polygon


def draw_instance(path, h, w):
    # Load the JSON file containing the polygon mask data
    with open(path, 'r') as file:
        polygon_data = json.load(file)

    # Create an empty mask
    mask = np.zeros((h,w), dtype=np.uint8)

    # Iterate through each polygon data
    for polygon_entry in polygon_data:
        # Extract polygon vertices
        polygon_vertices = polygon_entry['data']
        
        # Convert vertices to a numpy array
        polygon_np = np.array(polygon_vertices, dtype=np.int32)
        polygon_np = np.expand_dims(polygon_np, axis=0)
        
        # Create a Polygon object
        polygon = Polygon(polygon_vertices)
        
        # Draw the polygon on the mask
        # cv2.fillPoly(mask, polygon_np, color=(255,))
        
        # Draw the borders of the polygon on the mask
        border_color = (255,)  # You can adjust the color as needed
        cv2.polylines(mask, polygon_np, isClosed=True, color=border_color, thickness=1)
        print(mask.shape)

    return mask



