<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
</head>
<body>
  <h1>Financial Reinforcement Learning by Pytorch</h1>
  
  <h2>Table of Contents</h2>
  <ul>
    <li><a href="#About">About Project</a></li>
    <li><a href="#Requirements">Requirements</a></li>
  </ul>
  
  
  <h2 id="About">About Project</h2>
  
  <p>This simple project is written for the application of reinforcement learning in financial markets using the PyTorch library. A 22-day and an 8-day simple moving average are considered for each state over a period of ten consecutive days. The actions considered are buying, selling, and taking no action. By performing Action 1, which corresponds to buying an asset, the amount of profit or loss on the next day is calculated as a reward. Similarly, by performing Action 2, which corresponds to selling the asset, the amount of profit or loss from the time of purchase until the time of sale is calculated and considered as a reward. Furthermore, if no action is taken, the reward will be zero.</p>
  
  <p>
  In this project, the asset considered is BTC/USDT. Additionally, a deep learning model has been employed to generate actions based on the input state. The model consists of a simple linear model with one hidden layer.
  </p>
  <p>The required version of Bokeh is 3.0.3.</p>
  
  
  <h2 id="Requirements">Requirements</h2>
  <ol>
    <li>Pandas 1.5.2</li>
    <li>PyTorch 1.13.1</li>
    <li>Numpy 1.23.1</li>
    <li>Tqdm 4.64.1</li>
    <li>Scikit-learn 1.1.1</li>
  </ol>
 <h2></h2>
<p>  
you can clone to this repository and pull the project into your own system.
</p>
<p>
Note: Feel free to contribute to this project.
</p>
<p>
Reach me at Linkedin:
<a href="https://www.linkedin.com/in/hashemezzati/">My LinkedIn profile</a>
</p>
  </body>
  </html>
