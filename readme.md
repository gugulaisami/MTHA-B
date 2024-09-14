I wrote the code myself to reproduce the experiments of the paper ["Traded Control of Human–Machine Systems for Sequential Decision-Making Based on Reinforcement Learning" (2021)](https://ieeexplore.ieee.org/abstract/document/9613742).
Assessing the credibility of different agents is one of the most important tasks.For this part of my code, I referred to the uncertainty experiments in the paper ["Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" (2015)](http://www.cs.ox.ac.uk/people/yarin.gal/website/publications.html#Gal2015Dropout).
Here are some steps to get started:
1. Open 6.0.py
2. Install dependencies required
3. Pretrain the DQN which will output the actions of "machine". You will get a file named 'checkpoint.pth'. 
4. Exegesis the pretraining code, and run again. You will get a video which shows the results.
Good luck!