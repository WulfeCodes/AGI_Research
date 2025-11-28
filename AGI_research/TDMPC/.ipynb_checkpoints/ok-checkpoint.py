import os
os.environ['MUJOCO_GL'] = 'osmesa'
from tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from tdmpc2.tdmpc2.common.buffer import Buffer
from omegaconf import DictConfig
from PIL import Image
from pettingzoo.sisl import waterworld_v4
from transformers import AutoModel,AutoImageProcessor
import torch
import gymnasium as gym
import cv2
import traceback
from pathlib import Path
from tdmpc2.tdmpc2.trainer.offline_trainer import OfflineTrainer
from tdmpc2.tdmpc2.common.logger import Logger
from tensordict.tensordict import TensorDict
import numpy as np
from datetime import datetime
#started from file inline with github
#env is tdmpc2_env in D:/envs
#TODO EVAL MODE SET UP
#pip install tensordict, omegconf, pymunk
class tdmpc2:
    def __init__(self,envNum,vit_type='dino',isEval = True,checkpointPath=None):
        self.checkpointPath = checkpointPath
        self.taskID = envNum
        self.cfg = self.create_tdmpc2_config()
        self.TDMPC2_model = TDMPC2(self.cfg)
        self.vit_type = vit_type
        self.init_vit(vit_type=vit_type)
        self.isEval = isEval
        self.set_env(envNum)
        if self.checkpointPath: self.TDMPC2_model.load(self.checkpointPath)

    def init_vit(self,vit_type):   
        if vit_type == 'dino':
            self.vit_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            self.vit_model = AutoModel.from_pretrained("facebook/dinov2-large").to(self.cfg.device)

    def set_env(self,env=0):
        #set up for 1 agent, ideally all agents of diff archs work in parallel
        if env==0:
            self.env = waterworld_v4.parallel_env(n_pursuers=1,render_mode="rgb_array")
        elif env==1:
            self.env = gym.make('InvertedPendulum-v5',render_mode="rgb_array")
    def forward(self,image,taskID):
        
            if taskID == 0:
                inputs=self.vit_processor(images=image,return_tensors='pt')
                inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

                outputs = self.vit_model(**inputs)
                obs  = outputs.last_hidden_state

                ob=self.TDMPC2_model.img_projection_TDMPC2(obs)
                ob = ob.squeeze(0)
                ob=ob.mean(dim=0).to(self.cfg.device)
                actualAction=self.TDMPC2_model.act(obs=ob,eval_mode=self.isEval,task=taskID)

                actualAction=torch.clamp(actualAction,min=-1,max=1)
                cutAction = actualAction[:2]

                cutAction = {'pursuer_0': actualAction[:2]}

            if taskID == 1:
                
                inputs=self.vit_processor(images=image.copy(),return_tensors='pt')

                inputs = {k: v.to(self.cfg.device) for k, v in inputs.items()}

                outputs = self.vit_model(**inputs)
                obs  = outputs.last_hidden_state

                ob=self.TDMPC2_model.img_projection_TDMPC2(obs)
                ob = ob.squeeze(0)
                ob=ob.mean(dim=0)
                actualAction=self.TDMPC2_model.act(obs=ob,task=taskID)

                actualAction=torch.clamp(actualAction,min=-3,max=3)
                cutAction = actualAction[:1]
    
            return cutAction,actualAction

    def train(self,env2Train=0,buff=None):

        self.env.reset()
        trainer = OfflineTrainer(

        cfg=self.cfg,
        env=self.env,
        agent=self.TDMPC2_model,
        buffer=buff,
        logger=Logger(self.cfg),
        )
        
        trainer.train(env2Train)

    def run(self,episodeNum=500):
        print("boy oh boy its ready")
        gotData = False
        bufferCount=0
        currBuff = []

        buff = Buffer(self.cfg)
        for episode in range(episodeNum):
            print("on episode,eval mode",episode,self.isEval)
            observations, infos = self.env.reset()
            
            currEpisodeReward=0
            for i in range(self.cfg.steps):        #while currMBRL.env.agents:
                frame = cv2.resize(self.env.render(), (256, 256), interpolation=cv2.INTER_LINEAR)
                cutAction,actualAction=self.forward(image=frame,taskID=self.taskID)
                #TODO if using only image inference use only its frame
                next_obs, rewards, terminations, truncations, infos=self.env.step(cutAction.cpu())

                if self.taskID == 0:
                    currReward=torch.tensor([rewards['pursuer_0']])
                    terminated=torch.tensor([float(terminations['pursuer_0'])])
                    truncated=torch.tensor([float(truncations['pursuer_0'])])
                    currEpisodeReward+=rewards['pursuer_0']

                elif self.taskID ==1:
                    currReward=torch.tensor([rewards])
                    terminated = torch.tensor([float(terminations)])
                    truncated=torch.tensor([float(truncations)])
                    currEpisodeReward+=rewards
                    frame=frame.copy()

                step_td = TensorDict({
                    'obs': frame,
                    'terminated': terminated,
                    'action': actualAction,
                    'reward': currReward,
                    'task': self.taskID,
                })

                currBuff.append(step_td)
                bufferCount+=1

                if bufferCount%self.cfg.episode_lengths[self.taskID]==0:
                    print("saving buffer, bufferCount is: ",bufferCount)
                    episode_data = TensorDict.stack(currBuff)
                    buff.add(episode_data)
                    torch.save(buff._buffer.storage[:],f'/testAGI/AGI_research/TDMPC/trainingData/episode{int(datetime.now().timestamp())}_{self.taskID}.pt')
                    currBuff = []
                    buff = Buffer(self.cfg)
                    bufferCount = 0
                    gotData = True
                #save data and clear buffers        

                if terminated.item() or truncated.item():
                    break
                    
            self.env.close()

            print(f"ended episode:{episode},reward:{currEpisodeReward}")
            print("bufferCount:",bufferCount)

            self.isEval = True

            if gotData == True:
                print("training")
                self.train(self.taskID)
                gotData = False
                self.isEval = True

    def create_tdmpc2_config(self,task_name="custom", obs_dim=2, act_dim=17, 
                            model_size=1, episode_length=20 ,device='cuda'):
        """Create TDMPC2 config matching their actual structure"""
        
        config = {
            # Environment
            'task': task_name,
            'obs': 'rgb',
            'episodic': False,
            'device': device,
            
            # Evaluation
            'checkpoint': self.checkpointPath,
            #we eval within this file eval_episodes doesnt matter
            'eval_episodes': 10,
            #eval frequency is how often some weights gets saved
            #train loops by steps, eval freq must be lower than steps

            # Training
            'training_steps':50,

            'reward_coef': 1.000,
            'value_coef': 1.0,
            'termination_coef': 1,
            'consistency_coef': 2.0,
            'rho': 0.5,
            'lr': 3e-4,
            'enc_lr_scale': 0.3,
            'grad_clip_norm': 20,
            'tau': 0.01,
            'discount_denom': 5,
            'discount_min': 0.95,
            'discount_max': 0.995,
            'buffer_size': 10000,# if device == 'cpu' else 1000000,
            'exp_name': 'custom',
            'data_dir': Path('/testAGI/AGI_research/TDMPC/trainingData'),
            
            # Planning
            'mpc': True,
            'iterations': 4 if device == 'cpu' else 6,
            'num_samples': 256 if device == 'cpu' else 512,
            'num_elites': 32 if device == 'cpu' else 64,
            #HOW MANY PARALLEL MPC TRAJECTORIES
            'num_pi_trajs': 24, #if device == 'cpu' else 24,
            'horizon': 6,
            'min_std': 0.05,
            'max_std': 2,
            'temperature': 0.5,
            
            # Actor
            'log_std_min': -10,
            'log_std_max': 2,
            'entropy_coef': 1e-4,
            
            # Critic
            'num_bins': 101,
            'vmin': -10,
            'vmax': 10,
            
            # Architecture
            'model_size': 'small',
            'num_enc_layers': 0,
            'enc_dim': 0,
            'num_channels': 0,
            'mlp_dim': 608,
            'latent_dim': 608,
            #LATENT DIM IS INPUT INTO POLICY#TODO, 
            # IF USING TASK DIM POLICY INPUT = LATENT DIM + TASK DIM
            'task_dim': 1,
            'num_q': 5,
            'dropout': 0.01,
            'simnorm_dim': 8,
            
            # Logging
            'wandb_project': None,
            'wandb_entity': None,
            'wandb_silent': True,
            'enable_wandb': False,
            'save_csv': False,
            
            # Misc
            'compile': False, #if device == 'cpu' else True,
            'save_video': False,
            'save_agent': True,
            'seed': 42,
            
            # Auto-filled convenience params
            'task_title': task_name,
            'multitask': True,

            'obs_shape': {'state':(obs_dim,) if obs_dim else None},
            #action dim is your superset action vector dim
            'episode_length': 100,
            'obs_shapes': None,
            #action dims are your list of space for envIDs
            'seed_steps': None,
            'bin_size': 101,

            'episode_lengths': [100,500],
            'action_dim': act_dim,
            'action_dims': [2,1],
            #tdmpc2 trains steps times with batch_size
            'steps': 1000,
            'eval_freq': 99,
            'batch_size': 50,
            'tasks': [0,1],
            'work_dir': Path('/testAGI/AGI_research/TDMPC/TDMPC2Models'),
            #Tasks represents your list of ENV IDs
        }
        
        return DictConfig(config)

def main():
    print("starting")
    #PASS IN checkpointPath if you wish to load
    currMBRL=tdmpc2(envNum=1,checkpointPath="/testAGI/AGI_research/TDMPC/TDMPC2Models/models/198.pt")
    print("inited")
    currMBRL.run(500)
    #currMBRL.train(1,buff=None)

    #print("training")

    print("finished")
if __name__ == '__main__':
    main()