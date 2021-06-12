# 使用MCTS打（明牌）拖拉机

## 主要模块

* env.py: 包括一个（明牌）拖拉机的仿真系统
* states.py: 包括数个State类，以对全局和玩家状态进行封装
* agents.py: 包括数种出牌agent
* mcts.py: 包括一个MCTS结点类，用于基于MCTS的出牌agent中
* models.py: 包括数种根据state判断牌面价值和产生出牌策略的网络结构，通过tf实现
* util.py: 提供了一些训练和评估agent效果的函数
