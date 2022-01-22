# Mapless-Navigation-based-on-Deep-Reinforcement-Learning

## Abstract
This work builds an Autonomous Mobile Robot simulation and training environment for the Mapless Navigation based on Deep Reinforcement Learning problem.
- _**Performance indicator definition:**_ Given a number of destinations that exist in a limited space, it is a value that expresses the degree of success in reaching the destination sequentially as a percentage, and the unit is expressed as ‘%’.
- _**Evaluation method:**_ After setting N sequential destinations within a limited space, the number of times reaching the olfactory destination within a certain range (within a maximum of 30 cm) is measured and calculated as a percentage.
- _**Success rate:**_ 95% in the real environment

## The Approach
Based on the [openai_ros](http://wiki.ros.org/openai_ros) package substituted for the Autonomous Mobile Robot Navigation based on Deep Reinforcement Learning problem, some changes in the algorithm in [training_script](./src/turtle2_openai_ros_example/scripts/ppo_actor_critic_train.py) as well as the [reward fucntion](./src/openai_ros/openai_ros/src/openai_ros/task_envs/turtlebot2/turtlebot2_wall.py) have been set up to suit the project's requirements.

_**Note:**_ Read and understand about the openai_ros  before implementation

## The Current Results
- Test the robot navigation with a sequence of ten destinations, and the success rate is 100%.
<img src="/result/A Sequence of Ten Destinations in Rviz.png" class="fit image"> 

- Test with 50 different destination locations and considered the success rate.
<img src="/result/Success rate in testing environment.png" class="fit image"> 

## Installation
This repo contains the [trained models](./src/Trained_model) to perform like the result above. See the [installation](Install.md) for more information.

## Reference
- Idea:
  - https://arxiv.org/pdf/1703.00420.pdf
  - https://arxiv.org/pdf/1807.07870.pdf
- Deep Reinforcement Learning Algorithm:
https://arxiv.org/pdf/1707.06347.pdf
- Reinforcement Learning Environment for Robot Navigation:
http://wiki.ros.org/openai_ros
