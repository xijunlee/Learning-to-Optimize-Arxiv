# CO-ML-papers
The repository archives papers regarding the combination of combinatorial optimization and machine learning and corresponding reading notes. 

| Paper Title                                                  | Readers                  |Slide              |
| :----------------------------------------------------------- | ------------------------ |-------------------|
| Machine Learning for Combinatorial Optimization: a Methodological Tour d'Horizon | Xijun, Mingxuan, Huiling | |
| Learning to branch                                           | Xijun, Mingxuan, Huiling |Mingxuan             |
| Learning to Search in Branch and Bound Algorithm             | Xijun, Mingxuan, Huiling |Mingxuan |
| Learning to branch in mixed integer programming              | Xijun, Mingxuan, Huiling |Mingxuan            |
| Learning Combinatorial Optimization Algorithms over Graphs   |    Huiling (**again**), Xijun      |Mingxuan        |
| Boosting Combinatorial Problem Modeling with Machine Learning |  Huiling                        |     |
| Exact Combinatorial Optimization with Graph Convolutional Neural Networks with Graph Convolutional Neural Networks |Xijun                          |Xijun     |
| Predicting Tactical Solutions to Operational Planning Problems under Imperfect Information| Huiling, Xijun |        |
| Discriminative Embeddings of Latent Variable Models for Structured Data | Xijun |       |
| Learning when to use a decomposition                         |   Zhenkun               |   Zhenkun       |
|Best arm identification in multi-armed bandits with delayed feedback| Huiling |     Huiling      |
|Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method| Xijun|Xijun           |
|A Multi-task Selected Learning Approach for Solving 3D Flexible Bin Packing Problem|Xijun |Xijun |
|Pointer Networks| Huiling, Xijun | Huiling |
|NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING| Huiling,Xijun |  Huiling |
|ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS! | Huiling | Huiling |
|Reinforcement Learning for Solving the Vehicle Routing Problem| Huiling | Huiling |
|Generalized Inverse Multiobjective Optimization with Application to Cancer Therapy | Zhenkun | |
|Reinforcement Learning for Integer Programming: Learning to Cut| Zhenkun | Zhenkun |
|Learning Permutations with Sinkhorn Policy Gradient | Huiling | |
|Learning to Run Heuristics in Tree Search.  |Xijun |Xijun |
|Predicting Solution Summaries to Integer Linear Programs under Imperfect Information with Machine Learning.| Huiling | |
|Online Learning for Strong Branching Approximation in Branch-and-Bound|Zhenkun | | 
|Optimization as a model for few-shot learning| Huiling | |
|Learning a SAT Solver from Single-Bit Supervision|Xijun |Xijun  |
|Machine Learning to Balance the Load in Parallel Branch-and-Bound | Zhenkun | |
|Attention Solves Your TSP, Approximately|Xijun | |
|A Machine Learning-Based Approximation of Strong Branching|Zhenkun | |
|Learned Optimizers that Scale and Generalize|Huiling | |
|Learning fast optimizers for contextual stochastic integer programs |Huiling||

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

Paper location: CO-ML-papers⁩/papers/Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Metho.pdf

Notes:

这篇文章是阿里旗下菜鸟团队利用深度强化学习技术解决他们业务定义的一个新型3D bin packing problem (3DBPP)。具体问题定义为给定一系列待装载的item，如何将这些item装进一个bin中，使得这个bin的表面积最小。其方案思路为将整个3DBPP划分成三个连续的决策问题：1）决定item的装载顺序；2）决定每个item在bin中的摆放方向；3）决定每个item在bin中的摆放位置（coordinates）。上述决策问题中，第二、第三步均采用启发式方法，第一步(决定item的装载顺序）的问题利用深度强化学习来解决。具体来说，利用pointer networks来学习装载顺序的policy。其输入是待装载item的长宽高数据，输出是这些item的顺序(sequence of order)。需要注意的是，pointer network学到的装载policy应该是与摆放方向、摆放位置所采取的heurisitic算法强相关。因为pointer networks的学习（网络参数的更新）中，其reward的计算需要摆放方向、摆放位置所采取的heurisitic算法来辅助的（即算bin的表面积）。 —— Xijun

## A Multi-task Selected Learning Approach for Solving 3D Flexible Bin Packing Problem ##

Duan, Lu, et al. "A Multi-task Selected Learning Approach for Solving 3D Flexible Bin Packing Problem." Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems. International Foundation for Autonomous Agents and Multiagent Systems, 2019.

Paper location:  CO-ML-papers⁩/papers/A Multi-task Selected Learning Approach for Solving 3D Flexible Bin Packing Problem

这篇文章是阿里旗下菜鸟团队在Solving a New 3D Bin Packing Problem with Deep Reinforcement Learning Method基础上的改进工作。在前述工作中其方案思路为将整个3DBPP划分成三个连续的决策问题：1）决定item的装载顺序；2）决定每个item在bin中的摆放方向；3）决定每个item在bin中的摆放位置（coordinates）。上述决策问题中，第二、第三步均采用启发式方法，第一步(决定item的装载顺序）的问题利用深度强化学习来解决。然后这篇文章的工作是将1）决定item的装载顺序和2）决定每个item在bin中的摆放方向都用RL的方式来优化，然后摆放位置仍然是启发式算法。其具体思路是类似于Pointer network，利用encoder与decoder的架构一起学习最优顺序和最优摆放方向的策略(所以是weight sharing)，其中除了利用了pointer networks本身的attention mechanism外，他们还提出了更高一层的intra-attention mechanism（即在decode的过程中除了利用encoder的hidden states信息外，还利用前序时间步的decoder的hidden state，能有效防止重复item在序列中重复出现）。 —— Xijun

## Pointer Networks ## 

Oriol Vinyals (Google brain), Meire Fortunato (UC Berkeley) and Navdeep Jaitly (Google brain). NIPS 2015.

这是第一篇利用seq2seq模型通过end-to-end的方式来求解TSP问题的工作。主要贡献在于1）利用RNN构造了decoder和encoder，2）修改了传统attention networks的结构（去掉了传统attention中的weight-sum而直接输出distribution），并用此修正的attention连接encoder和decoder。3）利用标准的tsp solver作为标签，supervised的方式训练这样的网络，从而得到tsp问题的结果（节点顺序）。本文的实验中，最大的节点个数是500。——Huiling

## NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING ##

Irwan Bello, Hieu Pham, Quoc V. Le, Mohammad Norouzi & Samy Bengio （Google brain）. ICLR workshop, 2017. 

这是基于Pointer networks的改进工作，依然应用到tsp问题上。同时，作者也给出了在knapsack问题上的实验结果。本文依赖的网络结构依然是pointer networks，主要贡献在于使用了RL的训练方法。一种是基于pretraining的方法，其主要贡献是：1）使用policy gradient，问题（例如：TSP）是agent，不同时刻的Graph作为state，以期学出Q-policy。2）引入多线程处理算法A3C，其中critic有三个模块，分别是2个LSTM和1个二层神经网络（激活函数是relu）。本文同时还给出了另一种不需要pretraining的online的方法，active search。主要区别是：1）从一个固定的state出发，利用不同的hyperparameters, 采样得到不同的pi，2）求出最小的L_j = L（pi_j|s）。这两个步骤主要是用来改进decoding的过程。3）不断的利用test数据集来refine网络。通过后文实验显示，active search会得到质量更高的解，同时并未花费更多的时间。——Huiling


## Reinforcement Learning for Solving the Vehicle Routing Problem ##

Mohammadreza Nazari Afshin Oroojlooy Martin Takác Lawrence V. Snyder (Lehigh Univ.) NIPS， 2018

这是基于Pointer networks和Neural combinatorial两个工作的改进工作。与NEURAL COMBINATORIAL OPTIMIZATION WITH REINFORCEMENT LEARNING唯一的区别是，encoder的部分直接利用embedding来代替encoder，从而改进原有的pointer networks不适用于“基于时间的模型”这一劣势。这个文章的优势不仅在于可以处理Over time的模型，同时为constraints satisfactation的问题在end-to-end模型上的处理提供了可能。本文在vrp问题上做了测试。不过，实验结果仅包括在50辆车上的调度，是一个非常小规模的问题。—— Huiling

## Reinforcement Learning for Integer Programming: Learning to Cut ##

Yunhao Tang, Shipra Agrawal, Yuri Faenza (Columbia University), arxiv 2019

在这篇文章中，作者提出来用强化学习来加速割平面法的收敛速度。对于一个IP问题，割平面法是通过对一个松弛后LP最优解加cut(割平面)，并迭代求LP，直到该LP最优解满足所有整数约束。在割平面法中，割平面的选择极大的影响算法收敛速度。目前主要是是通过启发式算法来完成割平面的选择。本文首次提出使用RL来完成割平面法的选择。因为现实中割平面的数量可能非常大，作者通过一些实现技巧，做了些state space size 和 generality of method之间的平衡。实验结果现实，该方法极大的提升了算法的收敛速度，且具有良好的泛华能力。该算法还可以作为一个子程序用到一些Branch and Cut的solver里。-— Zhenkun

## ATTENTION, LEARN TO SOLVE ROUTING PROBLEMS! ##

Wouter Kool, Herke van Hoof & Max Wellling. ICLR 2019. 

本文依然借助encoder-decoder这个结构，来完成end-to-end的处理组合优化问题这一任务。本文的主要创新是借助了transformer （from "Attention is all you need"）的原有架构，将原模型在NLP任务上的效果，迁移到了组合优化问题中。但与原transformer主要的不同点在于，1）原有模型critic部分用了baseline，而本文用的rollout （经过试验对比，rollout的效果会远远好于baseline）。2）借助了Local search来提升解的质量 3）decoder的input没有用PCA降维。

据我们了解，本文是第一篇在100个节点（含以下）的问题上效率超过gurobi的算法。但是，本文的网络结构相对复杂，在更大规模的问题上的时间效率，需要进一步测试，或者修正模型。—— Huiling

## Learning to Run Heuristics in Tree Search ##

Khalil, Elias B., et al. "Learning to Run Heuristics in Tree Search." IJCAI. 2017.

本文提出了一种利用机器学习方法来指导branch&bound算法中启发式搜索解的框架。首先来说，目前市面上最好的开源mip求解器SCIP，在其branch&bound实现的过程中，对于每个节点都会有多种heuristic方法来进行节点选择和变量的选择，而这些启发式的调用频率和规则都是由专家确定的。这也就是说对于不同问题而言，这些启发式的调用频率都不会因问题不同而有所改变。因此本文作者提出了利用机器学习的方法来学习面对不同问题时branch&bound算法启发式调用规则，从而提高branch&bound算法的收敛速度。具体来说，对于某一种启发式H而言，给定一个b&b搜索树，它需要一个二分类器帮助其确定在树中某一个节点是否使用该启发式H。这是一个监督学习，因此需要线下收集大量的（feature,label），简单来说feature包括节点信息和树的整体信息等，label是该启发式H能否找到incumbent，能找到是1，不能是0.

他们通过实验发现提出的机器学习框架要比SCIP默认（专家设定）的启发式调用规则更能提升branch&bound算法的效率（即primal integral更小，这个指标是本文或者前述文章提出来的一个衡量branch&bound算法性能的指标，越小越好）。

其实本文构思很简单，但是贵在实验丰富，他们claim自己是第一篇系统性地优化树搜索过程中启发式使用策略。  -- Xijun

