# coding:utf-8

from .random_agent import RandomAgent
from .mcts_agent import MctsAgent
from .supervised_learning_agent import SupervisedLearningAgent

def constract_agent(agent_name):
    options = agent_name.split("_")
    # エージェントの名前から新品のインスタンスを返す
    if options[0] == "RandomAgent":
        return RandomAgent()
    elif options[0] == "MctsAgent":
        playout_num = int(options[1])
        return MctsAgent(playout_num)
    elif options[0] == "SupervisedLearningAgent":
        return SupervisedLearningAgent("./models/test.pth")
    else:
        return None