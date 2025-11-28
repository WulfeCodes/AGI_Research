import transformers 
from transformers import AutoModel,AutoImageProcessor, Trainer,TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from PIL import Image
from pettingzoo.sisl import waterworld_v4
import numpy as np
import cv2
#pip install common

#TODO create projection layers from image

class VisionActionModel:
    def __init__(self,vit='dino',gpt_size='135M',env_name='waterworld'):
        self.vit = vit
        self.gpt_size = gpt_size
        self.init_vit(vit)
        self.init_gpt(gpt_size)
        self.img_projection_135M=torch.nn.Linear(1024,576) if gpt_size=='135M' or '360M' else None
        self.env_name = env_name
        self.set_env(env_name)
        self.action_proj_135M_WW=torch.nn.Linear(49152,2,dtype=torch.float16)

    def init_vit(self,vit):   
        if vit == 'dino':
            self.vit_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-large")
            self.vit_model = AutoModel.from_pretrained("facebook/dinov2-large")
        else:
            print("ERROR, SELECT A VIT")
    def init_gpt(self,gpt_size):
        if gpt_size=='135M':
            checkpoint = "HuggingFaceTB/SmolLM2-135M"
        elif gpt_size == '360M':
            checkpoint = "HuggingFaceTB/SmolLM2-360M"
        elif gpt_size == '500M':
            checkpoint = "Qwen/Qwen2.5-0.5B"

        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            # for fp16 use `torch_dtype=torch.float16` instead
        self.gpt_model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", torch_dtype=torch.bfloat16)
        #inputs = self.tokenizer.encode("Gravity is", return_tensors="pt").to("cpu")
        #outputs = self.gpt_model.generate(inputs)
        #print(self.tokenizer.decode(outputs[0]))
    
    def forward(self,image=None,state_history=None):

        if self.vit == 'dino':
            #vit inference
            inputs=self.vit_processor(images=image,return_tensors='pt')
            outputs = self.vit_model(**inputs)
            last_hidden_states = outputs.last_hidden_state
            #the backbone!
            ip=self.img_projection_135M(last_hidden_states).to(self.gpt_model.dtype)
            #TODO determine most "sane" outward projection
            if self.gpt_size == '360M' or '135M':
                outputs = self.gpt_model(inputs_embeds=ip)

                if self.env_name == 'waterworld':
                    o=outputs.logits.mean(dim=1)
                    o = o.to(self.action_proj_135M_WW.weight.dtype)  # match Linear layer dtype
                    oc=self.action_proj_135M_WW(o)
                    occ=torch.tanh(oc.squeeze(0))
                    action_np = occ.detach().cpu().numpy()
                    
                    return action_np
            else:
                messages = [
                {"role": "user", "content": "Who are you?"},
                            ]
                inputs = self.tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to(self.model.device)

                outputs = self.model.generate(**inputs, max_new_tokens=40)
                print(self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:]))

    def set_env(self,env):
        #set up for 1 agent, ideally all agents of diff archs work in parallel
        if env=='waterworld':
            self.env = waterworld_v4.parallel_env(n_pursuers=1,render_mode="rgb_array")



def main():
    VA=VisionActionModel(vit='dino',gpt_size='135M',env_name='waterworld')
    
    observations, infos = VA.env.reset()
    c=torch.tensor(observations['pursuer_0'])
    while VA.env.agents:
        frame=VA.env.render() 
        action=VA.forward(image=frame,state_history=c)
        #action_space = VA.env.action_space(VA.env.agents[0])
        actions = {'pursuer_0': action}

        observations, rewards, terminations, truncations, infos=VA.env.step(actions)
        print("Currreward",rewards)
    VA.env.close()
    print("finished")
if __name__ == '__main__':
    main()
