import torch
from diffusers import StableDiffusionPipeline

def get_hf_name(model_name):
    if model_name == 'sd_v1.4':
        return 'CompVis/stable-diffusion-v1-4'
    elif model_name == 'sd_v1.5':
        return 'runwayml/stable-diffusion-v1-5'
    elif model_name == 'sd_v2.1':
        return 'stabilityai/stable-diffusion-2-1'


def get_model_path(model_name, backdoor, backdoor_method, defense, defense_method):
    if backdoor:
        if defense:
            path = f"{model_name}_{backdoor_method}_{defense_method}"
        else:
            path = f"{model_name}_{backdoor_method}"
        return "./data/models/" + path
    else:
        return get_hf_name(model_name)


def load_pipe(model_name, backdoor, backdoor_method, defense, defense_method, fp16):
    model_path = get_model_path(model_name, backdoor, backdoor_method, defense, defense_method)
    print("Loading... ", model_path)
    if backdoor == False:
        if fp16:
            pipe = StableDiffusionPipeline.from_pretrained(model_path, revision="fp16", torch_dtype=torch.float16)
        else:
            pipe = StableDiffusionPipeline.from_pretrained(model_path)
    elif defense:
        pass
    else:
        # Load backdoor pipelines
        if backdoor_method == 'textual_inversion':
            clean_path = get_hf_name(model_name)
            if fp16:
                pipe = StableDiffusionPipeline.from_pretrained(clean_path, variant="fp16", torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(clean_path)
            pipe.load_textual_inversion(model_path + '/learned_embeds.safetensors')
        elif backdoor_method == 'dreambooth':
            if fp16:
                pipe = StableDiffusionPipeline.from_pretrained(model_path, variant="fp16", torch_dtype=torch.float16)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(model_path)  
    
    return pipe
