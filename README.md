# CO-ML-papers
The repository archives papers regarding the combination of combinatorial optimization and machine learning and corresponding reading notes. 



[CO+ML Survey Bengio 2018.pdf](https://github.com/xijunlee/CO-ML-papers/blob/master/papers/CO%2BML Survey Bengio 2018.pdf) 

Reader: Xijun, Mingxuan, Huiling

Notes: 这篇论文Bengio总结了目前组合优化和机器学习融合的方法和架构。



从方法论上分为demonstration和experience。demonstration是需要agent通过ML或RL的方式学习expert(人或者优化算法最优解)的"经验”；experience是需要agent通过RL的方式从“零”，自我学习的方式学得求解优化问题的“经验”。



从架构上分为ene-to-end和嵌入式架构。end-to-end的架构是利用ML或RL完全替代优化算法来解决优化问题。嵌入式架构是在原始优化算法中利用ML替换掉速度较慢的模块，以此达到提高优化效率或精度的目的。



[Learning to Branch in Mixed Integer Programming.pdf](https://github.com/xijunlee/CO-ML-papers/blob/master/papers/Learning to Branch in Mixed Integer Programming.pdf)

[Learning to Search in Branch and Bound Algorithms.pdf](https://github.com/xijunlee/CO-ML-papers/blob/master/papers/Learning to Search in Branch and Bound Algorithms.pdf)