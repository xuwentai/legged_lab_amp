from .edge_cylinder_cfg import EdgeCylinderCfg, GreedyconcatEdgeCylinderCfg
from .virtual_obstacle_base import VirtualObstacleBase, VirtualObstacleCfg

# EdgeCylinder is NOT eagerly imported — it pulls cv2/sklearn/warp.
# The class_type field in EdgeCylinderCfg uses a {DIR} ResolvableString
# that only imports EdgeCylinder when the cfg is instantiated inside
# launch_simulation.
