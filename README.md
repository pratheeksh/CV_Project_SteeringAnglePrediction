# Steering Angle Prediction for self-driving cars

## Introduction
We predict steering angles in the effort at teaching a car how to drive using only cameras and deep
learning. Steering angle prediction is a critical component of a driverless car which makes it capable
of sensing its environment and navigate. The motivation was to build a component of an end-to-end
solution of a self-driving car, using only ConvNets.

## Problem Statement
This problem was released as an open competition by Udacity as part of their Nano-degree program
on Self driving cars. We use the same dataset. The purpose of the challenge is to take image frames
from a camera mounted to the windshield of the car, and predict the appropriate steering angle.
A similar problem is addressed by the self-driving car team at NVIDIA, in the paper End-to-end
learning of Self-driving cars.

## Results
We report MSE for Resnet and Baseline , Smoothed-L1-loss for Nvidia network regression and Cross-Entropy fro Nvidia network Classification. We switched to Smoothed-L1-loss early on because our models weren’t converging. The Nvidia-regression results are from then. The final test-error turned out to be the same if the model converged for the same number of epochs. Later on we moved to ADAM optimization and specific random initialization procedures and got both Resnet and Nvidia to converge. Considering this MSE seemed like a more natural measure.

### Credits
The one person that indubitably deserves credit for this project, arguably more than the authors themselves, is Lucian Ionita who generously allowed us to use his GPUs.


