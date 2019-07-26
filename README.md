# CO-ML-papers
The repository archives papers regarding the combination of combinatorial optimization and machine learning and corresponding reading notes. 

| Paper Title                                                  | Readers                  |Slide              |
| :----------------------------------------------------------- | ------------------------ |-------------------|
| Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon | Xijun, Mingxuan, Huiling | |
| Learning to branch                                           | Xijun, Mingxuan, Huiling |Mingxuan             |
| Learning to Search in Branch and Bound Algorithm             | Xijun, Mingxuan, Huiling |Mingxuan |
| Learning to branch in mixed integer programming              | Xijun, Mingxuan, Huiling |Mingxuan            |
| Learning Combinatorial Optimization Algorithms over Graphs   |    Huiling (**again**), Xijun      |Mingxuan        |
| Automated Treatment Planning in Radiation Therapy using Generative Adversarial Networks |        Huiling               |    Huiling   |
| Boosting Combinatorial Problem Modeling with Machine Learning |  Huiling                        |     |
| Exact Combinatorial Optimization with Graph Convolutional Neural Networks with Graph Convolutional Neural Networks |                          |     |
| Inductive Representation Learning on Large Graphs            |            Huiling              |    Huiling      |
| Predicting Tactical Solutions to Operational Planning Problems under Imperfect Information| Huiling |        |
| Discriminative Embeddings of Latent Variable Models for Structured Data | Xijun |       |
| Learning when to use a decomposition                         |   Zhenkun               |          |
|Best arm identification in multi-armed bandits with delayed feedback| Huiling |     Huiling      |
|Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method| Xijun|Xijun           |


## CO+ML Survey Bengio 2018

Bengio, Yoshua, Andrea Lodi, and Antoine Prouvost. "Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon." *arXiv preprint arXiv:1811.06128* (2018).

Paper location: CO-ML-papers⁩/papers⁩/CO+ML Survey Bengio 2018.pdf

Notes:

这篇文章Bengio总结了目前组合优化和机器学习融合的方法和架构。从方法论上分为demonstration和experience。demonstration是需要agent通过ML或RL的方式学习expert(人或者优化算法最优解)的"经验”；experience是需要agent通过RL的方式从“零”，自我学习的方式学得求解优化问题的“经验”。从架构上分为ene-to-end和嵌入式架构。end-to-end的架构是利用ML或RL完全替代优化算法来解决优化问题。嵌入式架构是在原始优化算法中利用ML替换掉速度较慢的模块，以此达到提高优化效率或精度的目的。 —— Xijun

Note location: CO-ML-papers⁩/⁨notes⁩/20190717⁩/bengio-co-ml-survey

## Learning to Branch

Balcan, Maria-Florina, Travis Dick, Tuomas Sandholm, and Ellen Vitercik. "Learning to branch." *arXiv preprint arXiv:1803.10150* (2018).

Paper location: CO-ML-papers⁩/papers⁩/Learning to BranchMaria-Flori.pdf

Notes:

这篇文章是基于之前做branch&bound算法中各种评价node的指标。其目标是利用ML的方法学出一组权重，线性加权这些指标来综合评估node，从而达到减小搜索树的size。 —— Xijun

Note Location: CO-ML-papers⁩/⁨notes⁩/20190717⁩/learn-to-branch-flori

## Learning to Search in Branch and Bound Algorithm

He, He, Hal Daume III, and Jason M. Eisner. "Learning to search in branch and bound algorithms." *Advances in neural information processing systems*. 2014.

Location: CO-ML-papers⁩/papers⁩/Learning to Search in Branch and Bound Algorithms.pdf

Notes:

这篇文章首先假设有一个Oracle已经知道branch & bound中知道什么是最优node selection策略（表示为训练数据），利用imitation learning学习不同问题的Oracle的node selection policy和node pruning policy。在训练集上学完后，然后在测试集上完全用学到的策略指导branch & bound的搜索过程。 —— Xijun

Note location: CO-ML-papers⁩/⁨notes⁩/20190717⁩/hehe-learn-to-branch-nips2014

## Learning to Branch in Mixed Integer Programming

Khalil, Elias Boutros, et al. "Learning to branch in mixed integer programming." *Thirtieth AAAI Conference on Artificial Intelligence*. 2016.

Paper location: CO-ML-papers⁩/papers⁩/Learning to Branch in Mixed Integer Programming.pdf

Notes:

这篇文章是利用learning to rank学习branch & bound中的经典打分策略Strong Branching(SB)。其具体做法是，在面对任何一个MIP问题时，其算法在500个分支点前都用经典的Strong Branching作为变量评分标准来选择节点的最优分支变量，同时保留过程中的特征（手工构造的72个特种）和打分结果，这些特征和打分结果构成了训练集。在第501个节点时，利用learning to rank学出训练集中的SB打分策略，在后续节点中的变量选择中就都采用learning to rank来选择最优分支变量。 —— Xijun

## Learning when to use a decomposition

Kruber M, Lübbecke M E, Parmentier A. Learning when to use a decomposition[C]//International Conference on AI and OR Techniques in Constraint Programming for Combinatorial Optimization Problems. Springer, Cham, 2017: 202-210.

Paper location: CO-ML-papers⁩/papers⁩/Learning when to use a decomposition.pdf

Notes:

这篇文章提出监督学习来检测MIP问题结构，并根据结构安排合适合适分解策略，进而提升solver的求解速度的。当一个MIP问题具有arrowhead结构或者double-bordered block diagonal form。这个MIP就可以利用DW分解来进行更快速的求解。本文将MIP的结构detect问题建模成一个0-1分类问题。其中输入参数是，MIP，分解策略，以及时间范围。本文采用的是scikit-learn library的标准分类器。--zhenkun


## Best arm identification in multi-armed bandits with delayed feedback ##

Aditya Grover, Todor Markov, Peter Attia, Norman Jin, Nicholas Perkins, Bryan Cheong, Michael Chen, Zi Yang, Stephen Harris, William Chueh, Stefano Ermon (Stanford University University of Michigan Lawrence Berkeley National Laboratory )

Notes: 

这篇文章的思路与其他文章差别很大，bandit没有用来解决问题本身，而是用来挑选cplex的启发式策略。文章默认，cplex中有很多启发式策略，而针对不同的问题，会自动的筛选不同的。但是在筛选之前，cplex内部会有测试机制。本文就是为了缩短测试的时间而设计的。training set是2000个MIP问题，在有32个arms的rl模型上，利用不同时刻的feedback做训练。test的mip问题，则直接根据这32个arm所对应的特征，挑选启发式算法。因此，##启发式算法本身不重要，重要的是它们做表现出来的特征##。值得我们学习的地方是，这个文章的是把求解Mip转成了资源分配问题，同时是一个online的思路，很适合处理动态的云资源/wireless资源分配问题。-- Huiling


## Learning Combinatorial Optimization Algorithms over Graphs ##

Khalil, Elias, et al. "Learning combinatorial optimization algorithms over graphs." Advances in Neural Information Processing Systems. 2017.

Paper location: CO-ML-papers⁩/papers⁩/7214-learning-combinatorial-optimization-algorithms-over-graphs.pdf

Notes:

这篇文章挑了三个跟图相关的优化问题Minimum Vertex Cover, Maximum Cut,以及TSP。将求解这三个问题建模为Markov Decision Process，即有一个partial solution（set of vertex）,然后从剩下的vertex set中每一步挑选一个点进入partial solution，直到拼成一个合法解（MDP终止）。partial solution中的节点状态利用structure2vec来描述，利用Q-learning方法来学习action policy。需要注意的是其structure2vec和Q-evaluation function是放在一起来学习的。

该算法的优点是：graph embedding parameterization can deal with different graph instances and sizes seamlessly. 能自适应地求解来自同一分布但不同size的问题，抓住了同一分布问题的本质。  —— Xijun

## Exact Combinatorial Optimization with Graph Convolutional Neural Networks ##

Gasse, Maxime, et al. "Exact Combinatorial Optimization with Graph Convolutional Neural Networks." arXiv preprint arXiv:1906.01629 (2019).

Paper location: CO-ML-papers⁩/papers⁩/Exact Combinatorial Optimization with Graph Convolutional Neural.pdf

Notes:

这篇文章是利用graph convolutional network来实现learn to branch。具体做法也是将B&B建模成Markov Decision Process。MDP中的每个状态state是由constraint和variable的bipartite图表示，action是从fractional variable中挑选一个变量来做分支。但是其并没有利用reinforcement learning来完成policy的学习，而是通过imitation learning。因此其需要事先构造训练集，即(B&B树的状态，各个fractional variable的得分)。这里B&B树的状态利用构造的bipartite graph表示，variable得分根据strong branching score公式得到。有了这些训练集后，便利用imitation learning的方式训练一个graph convolutional neural network来学习strong branching score挑选variable的policy。

优势：利用GCNN学习strong branching的policy，避免了手工构造特征。实验结果表现超过其他利用机器学习方法提速的branching策略，同时也超过了SCIP的branching策略。 —— Xijun

## Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method ##

Hu, Haoyuan, et al. "Solving a new 3d bin packing problem with deep reinforcement learning method." arXiv preprint arXiv:1708.05930 (2017).

Paper location: Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Metho.pdf

Notes:

这篇文章是阿里旗下菜鸟团队利用深度强化学习技术解决他们业务定义的一个新型3D bin packing problem (3DBPP)。具体问题定义为给定一系列待装载的item，如何将这些item装进一个bin中，使得这个bin的表面积最小。其方案思路为将整个3DBPP划分成三个连续的决策问题：1）决定item的装载顺序；2）决定每个item在bin中的摆放方向；3）决定每个item在bin中的摆放位置（coordinates）。上述决策问题中，第二、第三步均采用启发式方法，第一步(决定item的装载顺序）的问题利用深度强化学习来解决。具体来说，利用pointer networks来学习装载顺序的policy。其输入是待装载item的长宽高数据，输出是这些item的顺序(sequence of order)。需要注意的是，pointer network学到的装载policy应该是与摆放方向、摆放位置所采取的heurisitic算法强相关。因为pointer networks的学习（网络参数的更新）中，其reward的计算需要摆放方向、摆放位置所采取的heurisitic算法来辅助的（即算bin的表面积）。 —— Xijun
