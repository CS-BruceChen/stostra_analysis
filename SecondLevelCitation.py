from collections import defaultdict
import numpy as np

def compute_second_connection(segments_dict: dict) -> dict:
    general_connection_matrix = defaultdict()