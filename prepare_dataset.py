__author__ = 'bharathipriyaa'

import argparse
import pandas as pd
from os import path
from collections import defaultdict
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Testing Udacity SDC data")
    parser.add_argument('--dataset', type=str, help='dataset folder with csv and image folders')

    args = parser.parse_args()
    dataset_path = args.dataset
    steering_log = path.join(dataset_path, 'steering.csv')
    image_log = path.join(dataset_path, 'camera.csv')
    camera_images = dataset_path

    df_steer = pd.read_csv(steering_log,usecols=['timestamp','angle'],index_col = False)
    df_camera = pd.read_csv(image_log,usecols=['timestamp','filename'],index_col = False)
    # df_camera = pd.read_csv(image_log,usecols=['timestamp'],index_col = False)

    leng = len(df_camera['timestamp'])
    df_camera['angle'] = pd.Series(np.random.randn(leng), index=df_camera.index)
    time_scale = int(1e9) / 10
    steerings, speeds = read_steerings(steering_log, time_scale)

    image_angle_map = []

    for index, row in df_camera.iterrows():
        ts = int(row['timestamp']/time_scale)
        if steerings.__contains__(ts):
            angle = sum(steerings[ts])/len(steerings[ts])
            row['angle'] = angle
            print(row)
    result_map_path = path.join(dataset_path, 'images_to_angles.csv')
    df_camera.to_csv(result_map_path)

def read_steerings(steering_log, time_scale):
    steerings = defaultdict(list)
    speeds = defaultdict(list)
    with open(steering_log) as f:
        for line in f.readlines()[1:]:
            fields = line.split(",")
            nanosecond, angle, speed = int(fields[0]), float(fields[1]), float(fields[3])
            timestamp = int(nanosecond / time_scale)
            steerings[timestamp].append(angle)
            speeds[timestamp].append(speed)
    return steerings, speeds

if __name__ == '__main__':
    main()
