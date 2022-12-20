# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T
from habitat.config.default import get_agent_config
from habitat.utils.render_wrapper import overlay_frame

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis

import cv2
import habitat
from habitat.config.default_structured_configs import (
    CollisionsMeasurementConfig,
    TopDownMapMeasurementConfig, HabitatSimRGBSensorConfig, HabitatSimDepthSensorConfig,
)
from habitat.core.agent import Agent
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from typing import TYPE_CHECKING, Union, cast
from habitat.utils.visualizations.utils import images_to_video
from habitat.tasks.nav.nav import NavigationEpisode


if TYPE_CHECKING:
    from habitat.core.simulator import Observations
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim


class ShortestPathFollowerAgent(Agent):
    r"""Implementation of the :ref:`habitat.core.agent.Agent` interface that
    uses :ref`habitat.tasks.nav.shortest_path_follower.ShortestPathFollower` utility class
    for extracting the action on the shortest path to the goal.
    """

    def __init__(self, env: habitat.Env, goal_radius: float):
        self.env = env
        self.shortest_path_follower = ShortestPathFollower(
            sim=cast("HabitatSim", env.sim),
            goal_radius=goal_radius,
            return_one_hot=False,
        )

    def act(self, observations: "Observations") -> Union[int, np.ndarray]:
        return self.shortest_path_follower.get_next_action(
            cast(NavigationEpisode, self.env.current_episode).goals[0].position
        )

    def reset(self) -> None:
        pass


def model_forward(im, augmentations, model, K, cats, thres):
    image_shape = im.shape[:2]  # h, w
    aug_input = T.AugInput(im)
    _ = augmentations(aug_input)
    image = aug_input.image

    batched = [{
        'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
        'height': image_shape[0], 'width': image_shape[1], 'K': K
    }]

    dets = model(batched)[0]['instances']
    n_det = len(dets)

    meshes = []
    meshes_text = []

    if n_det > 0:
        for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions,
                dets.pred_pose, dets.scores, dets.pred_classes
        )):
            if score < thres:
                continue

            cat = cats[cat_idx]

            bbox3D = center_cam.tolist() + dimensions.tolist()
            meshes_text.append('{} {:.2f}'.format(cat, score))
            color = [c / 255.0 for c in util.get_color(idx)]
            box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
            meshes.append(box_mesh)

    if len(meshes) > 0:
        im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(
            im, K, meshes, text=meshes_text, scale=im.shape[0],
            blend_weight=0.5, blend_weight_overlay=0.85
        )
    else:
        im_drawn_rgb = im
        im_topdown = np.zeros(shape=(im_drawn_rgb.shape[0], im_drawn_rgb.shape[0], 3))

    if im_topdown.shape != (im_drawn_rgb.shape[0], im_drawn_rgb.shape[0], 3):
        im_topdown = cv2.resize(im_topdown, (im_drawn_rgb.shape[0], im_drawn_rgb.shape[0])).astype(np.uint8)

    return im_drawn_rgb, im_topdown, len(meshes)


def do_habitat_test(args, cfg, habitat_config, model):
    model.eval()

    thres = args.threshold

    output_dir = f"{cfg.OUTPUT_DIR}/mp3d"
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')

    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']

    agent_config = get_agent_config(sim_config=habitat_config.habitat.simulator)
    h = agent_config.sim_sensors.rgb_sensor.height
    w = agent_config.sim_sensors.rgb_sensor.width

    f_ndc = 4
    f = f_ndc * h / 2

    K = np.array([
        [f, 0.0, w / 2],
        [0.0, f, h / 2],
        [0.0, 0.0, 1.0]
    ])

    dataset = habitat.make_dataset(
        id_dataset=habitat_config.habitat.dataset.type, config=habitat_config.habitat.dataset
    )
    with habitat.Env(config=habitat_config, dataset=dataset) as env:
        agent = ShortestPathFollowerAgent(
            env=env,
            goal_radius=habitat_config.habitat.task.measurements.success.success_distance,
        )

        num_scenes = len(set(e.scene_id for e in env.episodes))
        num_episodes = habitat_config.habitat.environment.iterator_options.max_scene_repeat_episodes * num_scenes
        for _ in range(num_episodes):
            observations = env.reset()
            agent.reset()

            im_drawn_rgb, im_topdown, ndets = model_forward(cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR), augmentations, model, K, cats, thres)
            frame = cv2.cvtColor(np.concatenate((im_drawn_rgb, im_topdown), axis=1).astype(np.uint8), cv2.COLOR_RGB2BGR)
            frame = overlay_frame(frame, {"ndets": ndets})
            vis_frames = [frame]

            while not env.episode_over:
                action = agent.act(observations)
                if action is None:
                    break

                observations = env.step(action)

                im_drawn_rgb, im_topdown, ndets = model_forward(cv2.cvtColor(observations["rgb"], cv2.COLOR_RGB2BGR), augmentations, model, K, cats, thres)
                frame = cv2.cvtColor(np.concatenate((im_drawn_rgb, im_topdown), axis=1).astype(np.uint8), cv2.COLOR_RGB2BGR)
                frame = overlay_frame(frame, {"ndets": ndets})
                vis_frames.append(frame)


            current_episode = env.current_episode
            video_name = f"{os.path.basename(current_episode.scene_id).split('.')[0]}_{f'{current_episode.episode_id}'.zfill(3)}"
            images_to_video(vis_frames, output_dir, video_name)
            vis_frames.clear()


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)
    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    config = habitat.get_config(
        config_path="benchmark/nav/pointnav/pointnav_gibson.yaml"
    )
    with habitat.config.read_write(config):
        config.habitat.task.measurements.update({
            "top_down_map": TopDownMapMeasurementConfig(),
            "collisions": CollisionsMeasurementConfig(),
        })
        config.habitat.simulator.agents.main_agent.sim_sensors.rgb_sensor = HabitatSimRGBSensorConfig()
        config.habitat.simulator.agents.main_agent.sim_sensors.depth_sensor = HabitatSimDepthSensorConfig()
        config.habitat.environment.iterator_options.update({
            "shuffle": True,
            "group_by_scene": True,
            "num_episode_sample": -1,
            "max_scene_repeat_episodes": 10
        })
        config.habitat.dataset.data_path = "data/datasets/pointnav/mp3d/v2/{split}/{split}.json.gz"
        config.habitat.dataset.split = "val"


    with torch.no_grad():
        do_habitat_test(args, cfg, config, model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder', type=str, help='list of image folders to process', required=True)
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true",
                        help="Whether to show the images in matplotlib", )

    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
             "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )