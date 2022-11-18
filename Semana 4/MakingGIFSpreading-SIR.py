import imageio as i
import numpy as np
import matplotlib.pyplot as plt
import os 

mydir = os.chdir('Timelapse')
mydir = os.getcwd()

frames = []
for t in range(1, 11):
    image = i.v2.imread(f'Time_{t}.png')
    frames.append(image)

i.mimsave(
    'sir_spreading.gif', 
    frames, 
    fps=0.5
)