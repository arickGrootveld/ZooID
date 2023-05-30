# Introduction

Welcome to ZooID, a project meant to help researchers who are working with Zooplankton around the world. 

The purpose of this project is to develop a free and open-source tool to allow the automatic identification of Zooplankton using low-cost equipment. The code in this repo allows you to segment images of zooplankton automatically, classify the segmented images using a Convolutional Neural Network, and provide a very simple interface for implementing the code in a UI. 

The only hardware requirements for this project are a computer, a microscope and a camera.

The model weights and dataset that correspond to this repository can be found at: https://zenodo.org/record/7979996

# Repo Components

The repo is broken into 3 subdirectories, "imageExtractionCode" which handles the image preprocessing that segments images, "ConvNetCode" which takes the extracted images and classifies them into 1 of 5 taxa (Amphipods, Copepods, Fish Larvae, Ostracods, and Quetognaths) using one of several flavors of Neural Networks, and then "UIDemo" which combines both elements into a simple simplistic demonstration of the capabilities of the image extraction and Conv. Networks. 


# Aknowledgements

This project has been a collaboration between Dr. Kozak (and her many bright and talented students) from Universidad de Guadalajara, Dr. Carmen Franco-Gordo from Universidad de Guadalajara, and Arick Grootveld from Syracuse University. 


**Special thanks to Alma Margarita for coming up with the name for this project.**
