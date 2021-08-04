.. _basic_training:

Train a robot - switch
=============

Run the default training without specifying parameters:

``python train.py --robot kuka --reward switch --task_objects switch --task_type switch --gui 1``

The training will start with gui window and standstill visualization. New directory 
is created in the logdir, where tranining checkpoints, final model and other relevant 
data are stored. 

If you want try different behavior of robot you can change in reward.py --> SwitchReward(DistanceReward)
values of coefficients - self.k_w, self.k_d, self.k_a.
Values have to be <0; 1>


Wait until the first evaluation after 50000 steps to check the progress:

.. figure:: ../../../myGym/images/workspaces/switch/kuka50000.gif
   :alt: training

After 250000 steps the arm is able to switch the goal object with 80%
accuracy:

.. figure:: ../../../myGym/images/workspaces/switch/kuka250000.gif
   :alt: training

