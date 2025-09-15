from tdmpc2.tdmpc2.tdmpc2 import TDMPC2
from omegaconf import DictConfig
from PIL import Image
from pettingzoo.sisl import waterworld_v4
from transformers import AutoModel,AutoImageProcessor
import torch
#started from file inline with github
#env is tdmpc2_env in D:/envs
#TODO joint embedding with state?
#pip install tensordict, omegconf, pymunk
class tdmpc2:
    def __init__(self,vit_type='dino'):
        self.vit_type = vit_type
        self.init_vit(vit_type=vit_type)
        self.cfg = self.create_tdmpc2_config()
        self.TDMPC2_model = TDMPC2(self.cfg)
        self.set_env()
        self.img_projection_TDMPC2=torch.nn.Linear(1024,608)

    def init_vit(self,vit_type):   
        if vit_type == 'dino':
            self.vit_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            self.vit_model = AutoModel.from_pretrained("facebook/dinov2-large")

    def set_env(self,env='waterworld'):
        #set up for 1 agent, ideally all agents of diff archs work in parallel
        if env=='waterworld':
            self.env = waterworld_v4.parallel_env(n_pursuers=1,render_mode="rgb_array")
    def forward(self,image,state_history):
        if self.vit_type == 'dino':
            #vit inference
            inputs=self.vit_processor(images=image,return_tensors='pt')
            outputs = self.vit_model(**inputs)
            obs = outputs.last_hidden_state

            ob=self.img_projection_TDMPC2(obs)
            ob = ob.squeeze(0)
            ob=ob.mean(dim=0)
            print("SO CLOSE",ob.shape)
            action=self.TDMPC2_model.act(obs=ob)
            return action
            #the backbone!
            #TODO do we need an additional projection?
            #ip=self.img_projection_135M(last_hidden_states).to(self.gpt_model.dtype)

    def create_tdmpc2_config(self,task_name="custom", obs_dim=2, act_dim=2, 
                            model_size=1, device='cpu'):
        """Create TDMPC2 config matching their actual structure"""
        
        config = {
            # Environment
            'task': task_name,
            'obs': 'rgb',
            'episodic': False,
            
            # Evaluation
            'checkpoint': None,
            'eval_episodes': 5,
            'eval_freq': 10000,
            
            # Training
            'steps': 100000,
            'batch_size': 128 if device == 'cpu' else 256,
            'reward_coef': 0.1,
            'value_coef': 0.1,
            'termination_coef': 1,
            'consistency_coef': 20,
            'rho': 0.5,
            'lr': 3e-4,
            'enc_lr_scale': 0.3,
            'grad_clip_norm': 20,
            'tau': 0.01,
            'discount_denom': 5,
            'discount_min': 0.95,
            'discount_max': 0.995,
            'buffer_size': 100000 if device == 'cpu' else 1000000,
            'exp_name': 'custom',
            'data_dir': './data',
            
            # Planning
            'mpc': True,
            'iterations': 4 if device == 'cpu' else 6,
            'num_samples': 256 if device == 'cpu' else 512,
            'num_elites': 32 if device == 'cpu' else 64,
            #HOW MANY PARALLEL MPC TRAJECTORIES
            'num_pi_trajs': 1, #if device == 'cpu' else 24,
            'horizon': 3,
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
            'model_size': model_size,
            'num_enc_layers': 0,
            #enc dim might have sumn
            'enc_dim': 0,
            'num_channels': 0,
            'mlp_dim': 608,
            'latent_dim': 608,
            #LATENT DIM IS INPUT INTO POLICY, 
            # IF USING TASK DIM, PROJECTION+TASK_DIM = LATENT_DIM
            'task_dim': 0,
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
            'compile': False if device == 'cpu' else True,
            'save_video': False,
            'save_agent': False,
            'seed': 42,
            
            # Auto-filled convenience params
            'work_dir': './results',
            'task_title': task_name,
            'multitask': False,
            'tasks': None,
            'obs_shape': {'state':(obs_dim,) if obs_dim else None},
            'action_dim': act_dim,
            'episode_length': 500,
            'obs_shapes': None,
            'action_dims': 2,
            'episode_lengths': 500,
            'seed_steps': None,
            'bin_size': None,
        }
        
        return DictConfig(config)

def main():
    currMBRL=tdmpc2()
    print("boy oh boy its ready")

    observations, infos = currMBRL.env.reset()
    c=torch.tensor(observations['pursuer_0'])
    print(c.shape)
    while currMBRL.env.agents:
        frame=currMBRL.env.render() 
        print("hol hol hol upp now",type(frame),frame.shape)
        action=currMBRL.forward(image=frame,state_history=c)
        #action_space = VA.env.action_space(VA.env.agents[0])
        actions = {'pursuer_0': action}

        observations, rewards, terminations, truncations, infos=currMBRL.env.step(actions)
        print("Currreward",rewards)
    currMBRL.env.close()
    print("finished")
if __name__ == '__main__':
    main()