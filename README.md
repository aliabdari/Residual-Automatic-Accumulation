# Residual-Automatic-Accumulation

This repository intends to present approach regarding extracting accumulated residuals from compressed videos.

The complete explanation of the methods is presented this [paper](https://arxiv.org/abs/2209.14757)

Generally, a vast majority of videos in different platforms are transmitted and stored in compressed format. Therefore, it could be so advantageous if we could take advantage of compressed videos directly, since they contain useful available information like Motion Vectors(MVs) and Frequncy-Domain coefficients.

Residuals are some kinds of data could be obtained with linear computations from Frequency-Domain coefficients. Due to the fact Residuals are so similar to consequent frames subtraction, they can be used for many video analysis purposes, including Action Recognition.

In this repository, the procedure of extraction Residuals from compressed videos has been implemented. Furthermore, in a great many cases, we have some kinds of videos, especially surveillance ones, containing many consequent frames, which are so similar. In such kinds of videos containing countless frames, processing all of of the frames would be really laborious, which persuaded us to propose "Residual Automatic Accumulation" method to accumulate adjacent similar frames with each other to create a strong residual frames to be processed in only one step. Employing this method could reduce the number of processed frames drammatically without any noticeable reduction in terms of performance of algorithm.


