from omegaconf import DictConfig, OmegaConf
from gymnasium import spaces
import gymnasium as gym
from pettingzoo.sisl import waterworld_v4
import pathlib
from pathlib import Path
import numpy as np
import importlib
import torch
import torchvision.transforms as T
from tensordict.tensordict import TensorDict
import cv2
import random
from datetime import datetime
import os
os.environ['MUJOCO_GL'] = 'osmesa'
Dreamer = importlib.import_module('dreamerv3-torch.dreamer').Dreamer
Logger = importlib.import_module('dreamerv3-torch.dreamer').tools.Logger
tools = importlib.import_module('dreamerv3-torch.dreamer').tools

#TODO save, train, and load

class DreamerObj:
    def __init__(self,act_space=17,currTaskID=0,dataDir=None,modelDir=None,logDir=None,load=False,checkpointPath=None):
        self.config=self.generate_config()
        self.dataDir = dataDir
        self.modelDir = modelDir
        self.logDir = logDir
        self.taskID = currTaskID

        self.set_env(self.taskID)
        self.frame = cv2.resize(self.env.render(), (256, 256), interpolation=cv2.INTER_LINEAR)  # shape: (64,64,3)

        self.observation_space = spaces.Dict({
            'image': spaces.Box(
                low=0, 
                high=255, 
                shape=self.frame.shape)
        })

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 256)),
            T.ToTensor(),
        ])

        logga=Logger(logdir=self.logDir,step=0)
        #dreamer args: obs_space, act_space, config, logger, dataset        
        self.DreamerModel=Dreamer(self.observation_space,act_space,self.taskID,self.config,logga,None)

        if load:
            checkpoint = torch.load(checkpointPath, weights_only=False,map_location=self.config.device)
            self.DreamerModel.load_state_dict(checkpoint['agent_state_dict'])
            # Load optimizer states
            tools.recursively_load_optim_state_dict(
                self.DreamerModel, 
                checkpoint['optims_state_dict']
            )
            print("loaded")

    def forward(self,frame,i,isTrain=None,state=None):
            obs = {
                    'image':frame,
                    'is_first':torch.tensor([True]) if i==0 else torch.tensor([False]),
                    'is_terminal':torch.tensor([True]) if i>=self.config.batch_length-1 else torch.tensor([False]),
                }
            #COPY frame if Mujoco
                
            policy_output,state=self.DreamerModel._policy(obs=obs, state=state, training=isTrain)
            actualAction = policy_output['action'].squeeze(0)
            
            if self.taskID == 0:
                cutAction={'pursuer_0': policy_output['action'].squeeze(0)}
                actualAction=policy_output
                
            elif self.taskID == 1:
                cutAction = policy_output['action'].squeeze(0)
                cutAction=cutAction[:self.config["action_indexes"][self.taskID]]
                torch.clamp(cutAction,-3,3)

            elif self.taskID == 2:
                cutAction = policy_output['action'].squeeze(0)
                cutAction=cutAction[:self.config["action_indexes"][self.taskID]]
                torch.clamp(cutAction,-1,1)

            return cutAction,actualAction,state

    
    def run(self,trainNum,episodeLength,evalPeriod=10):
        #run should go through trainNum episodes,
        #each episode runs config.batch_length or until terminated. whichever comes first
        #where it saves every self.config["episode_lengths"][self.taskID] 

        bufferCount = 0
        print("running")
        currBuff = []
        gotData=False
        self.set_env(trainNum)

        for episode in range(episodeLength):
            print(f"episode: {episode}")
            currEpisodeReward = 0
            _,__=self.env.reset()
            #stateSave = TensorDict({}, batch_size=[])  # Empty TensorDict
            state = None

            for i in range(self.config.batch_length):
                
                frame = cv2.resize(self.env.render(), (256, 256), interpolation=cv2.INTER_LINEAR)
                if self.taskID == 1:
                    frame = frame.copy()
                cutAction,actualAction,state=self.forward(frame,i,isTrain=True,state=state)
                next_obs, rewards, terminations, truncations, infos=self.env.step(cutAction.detach().cpu().numpy())
        
                if self.taskID == 0:
                    #waterworld multi agent environment
                    currReward=torch.tensor([rewards['pursuer_0']])
                    terminated=torch.tensor([float(terminations['pursuer_0'])])
                    truncated=torch.tensor([float(truncations['pursuer_0'])])
                    currEpisodeReward+=rewards['pursuer_0']

                elif self.taskID ==1 or self.taskID ==2:
                    currReward=torch.tensor([rewards])
                    terminated = torch.tensor([float(terminations)])
                    truncated=torch.tensor([float(truncations)])
                    currEpisodeReward+=rewards
                
                
                step_td = TensorDict({
                    'image':frame,
                    'is_first': True if i ==0 else False,
                    'is_terminal':True if terminated else False,
                    'terminated': terminated.to('cpu'),
                    'action': actualAction.to('cpu'),
                    'reward': torch.tensor([currReward]).to('cpu'),
                    'task': self.taskID,
                })

                currBuff.append(step_td)
                bufferCount+=1

                if bufferCount%self.config["episode_lengths"][self.taskID] == 0:
                    episode_data = TensorDict.stack(currBuff)
                    episode_data=episode_data.unsqueeze(0)

                    self.save_episode(episode_data)
                    print(f"successfully saved episode {episode}, bufferCount is {bufferCount}")
                    currBuff = []
                    bufferCount = 0
                    gotData = True
                    break

                if terminated.item() or truncated.item():
                    break
                    
            
            print(f"episode {episode}, reward: {currEpisodeReward}")
            self.env.close()

            if gotData == True:
                print("training")
                self.train()
                print(f"trained, now evaluating")
                self.eval()
                self.DreamerModel.train()
                
                gotData = False

    def eval(self):
        currEpisodeReward = 0
        _,__=self.env.reset()
        self.DreamerModel.eval()
        terminated = torch.tensor([False])
        truncated = torch.tensor([False])
        state = None
        i = 0
        while not terminated.item() or not truncated.item():
            frame = cv2.resize(self.env.render(), (256, 256), interpolation=cv2.INTER_LINEAR)
            if self.taskID == 1:
                frame = frame.copy()
                cutAction,actualAction,state=self.forward(frame,i,isTrain=False,state=state)
                next_obs, rewards, terminations, truncations, infos=self.env.step(cutAction.detach().cpu().numpy())
                
            if self.taskID == 0:
                #waterworld multi agent environment
                currReward=torch.tensor([rewards['pursuer_0']])
                terminated=torch.tensor([float(terminations['pursuer_0'])])
                truncated=torch.tensor([float(truncations['pursuer_0'])])
                currEpisodeReward+=rewards['pursuer_0']
            
            elif self.taskID ==1 or self.taskID ==2:
                currReward=torch.tensor([rewards])
                terminated = torch.tensor([float(terminations)])
                truncated=torch.tensor([float(truncations)])
                currEpisodeReward+=rewards
            i+=1

            if terminated.item() or truncated.item():
                break
        self.env.close()
        print("evaluation reward:",currEpisodeReward)
        
    def train(self,dataDir="/testAGI/AGI_research/Dreamer/dreamerData/",
        saveDir="/testAGI/AGI_research/Dreamer/dreamerModels"):
        dataDirPath = Path(dataDir)
        fps = list(dataDirPath.glob(f'*_{self.taskID}.pt'))  # glob inside the directory
        fp=random.choice(fps)
        fps = None

        data=torch.load(fp,weights_only=False)
        print("training")

        print(f"sampled:{fp} and loaded its data, starting training")
        self.DreamerModel._train(data)

        items_to_save = {
            "agent_state_dict": self.DreamerModel.state_dict(),
            "optims_state_dict": tools.recursively_collect_optim_state_dict(self.DreamerModel),
        }
        torch.save(items_to_save, f"{saveDir}/latest.pt")

    def save_episode(self,episode_data):
        """
        Save a single episode to disk.
        
        Args:
            directory: Path to save directory
            episode_data: Dict with keys like 'observation, 'action', 'reward', 'done'
            episode_id: Unique identifier for the episode (e.g., timestamp or counter)
        """
        directory = pathlib.Path(self.dataDir).expanduser()

        filename = directory / f"episode{int(datetime.now().timestamp())}_{self.taskID}.pt"
        torch.save(episode_data,filename)

    def generate_config(self):
        config = {
                "logdir": None,
                "traindir": None,
                "evaldir": None,
                "offline_traindir": "",
                "offline_evaldir": "",
                "seed": 0,
                "deterministic_run": False,
                "steps": 1000,
                "parallel": False,
                "eval_every": 100,
                "eval_episode_num": 10,
                "log_every": 1e4,
                #NUM ACTIONS WAS NEEDED FOR INIT, 
                #NOT PROVIDED AT FIRST
                "num_actions": 17,
                "action_lengths": [2,1,2],
                'episode_lengths': [300,300,300],

                "reset_every": 0,
                "device": "cuda",
                "compile": True,
                "precision": 32,
                "debug": False,
                "video_pred_log": True,
                "action_indexes": [2,1,2],
                # Environment
                "task": "dmc_walker_walk",
                "size": [64, 64],
                "envs": 1,
                "action_repeat": 2,
                "time_limit": 1000,
                "grayscale": False,
                "prefill": 2500,
                "reward_EMA": True,

                # Model
                "embed_size": 608,
                "dyn_hidden": 256,#512
                "dyn_deter": 256,#512
                "dyn_stoch": 32,
                "dyn_discrete": None,
                "dyn_rec_depth": 1,
                "dyn_mean_act": "none",
                "dyn_std_act": "sigmoid2",
                "dyn_min_std": 0.1,
                "grad_heads": ["decoder", "reward", "cont"],
                "units": 256,#512
                "act": "SiLU",
                "norm": True,
                "encoder": {
                    "mlp_keys": "$^",
                    "cnn_keys": "image",
                    "act": "SiLU",
                    "norm": True,
                    "cnn_depth": 32,
                    "kernel_size": 4,
                    "minres": 4,
                    "mlp_layers": 5,
                    "mlp_units": 256,#1024
                    "symlog_inputs": True
                },
                "decoder": {
                    "mlp_keys": "$^",
                    "cnn_keys": "image",
                    "act": "SiLU",
                    "norm": True,
                    "cnn_depth": 32,
                    "kernel_size": 4,
                    "minres": 4,
                    "mlp_layers": 5,
                    "mlp_units": 256,#1024
                    "cnn_sigmoid": False,
                    "image_dist": "mse",
                    "vector_dist": "symlog_mse",
                    "outscale": 1.0
                },
                "actor": {
                    "layers": 2,
                    "dist": "normal",
                    "entropy": 3e-4,
                    "unimix_ratio": 0.01,
                    "std": "learned",
                    "min_std": 0.1,
                    "max_std": 1.0,
                    "temp": 0.1,
                    "lr": 3e-5,
                    "eps": 1e-5,
                    "grad_clip": 100.0,
                    "outscale": 1.0
                },
                "critic": {
                    "layers": 2,
                    "dist": "symlog_disc",
                    "slow_target": True,
                    "slow_target_update": 1,
                    "slow_target_fraction": 0.02,
                    "lr": 3e-5,
                    "eps": 1e-5,
                    "grad_clip": 100.0,
                    "outscale": 0.0
                },
                "reward_head": {
                    "layers": 2,
                    "dist": "symlog_disc",
                    "loss_scale": 1.0,
                    "outscale": 0.0
                },
                "cont_head": {
                    "layers": 2,
                    "loss_scale": 1.0,
                    "outscale": 1.0
                },
                "dyn_scale": 0.5,
                "rep_scale": 0.1,
                "kl_free": 1.0,
                "weight_decay": 0.0,
                "unimix_ratio": 0.01,
                "initial": "learned",

                # Training
                "batch_size": 1,
                "batch_length": 100,
                "train_ratio": 512,
                "pretrain": 100,
                "model_lr": 1e-4,
                "opt_eps": 1e-8,
                "grad_clip": 1000,
                "dataset_size": 1000000,
                "opt": "adam",

                # Behavior
                "discount": 0.997,
                "discount_lambda": 0.95,
                "imag_horizon": 15,
                "imag_gradient": "dynamics",
                "imag_gradient_mix": 0.0,
                "eval_state_mean": False,

                # Exploration
                "expl_behavior": "greedy",
                "expl_until": 0,
                "expl_extr_scale": 0.0,
                "expl_intr_scale": 1.0,
                "disag_target": "stoch",
                "disag_log": True,
                "disag_models": 10,
                "disag_offset": 1,
                "disag_layers": 4,
                "disag_units": 400,
                "disag_action_cond": False,

            "debug": {
                "debug": True,
                "pretrain": 1,
                "prefill": 1,
                "batch_size": 10,
                "batch_length": 20
            }
        }
        return OmegaConf.create(config)

    def set_env(self,env):
        #set up for 1 agent, ideally all agents of diff archs work in parallel
        if env==0:
            self.env = waterworld_v4.parallel_env(n_pursuers=1,render_mode="rgb_array")
            obs, info = self.env.reset()
        if env==1:
            self.env = gym.make('InvertedPendulum-v5',render_mode="rgb_array")
            obs, info = self.env.reset()

        if env==2:
            self.env = gym.make("LunarLanderContinuous-v3", render_mode="rgb_array")
            obs, info = self.env.reset(seed=42)

def main():
    #IF YOU ARE LOADING WEIGHTS, SET load=True with checkpointPath set to your model.pt location
    
    CurrDreamer=DreamerObj(currTaskID=1,dataDir="/testAGI/AGI_research/Dreamer/dreamerData",modelDir=None,load=False)
    print("dreamer initialized")

    CurrDreamer.run(1,200000)
    print("starting training")
    #CurrDreamer.train()
    print("finished training")
if __name__ == '__main__':
    main()