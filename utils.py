def get_hf_name(model_name):
    if model_name == 'sd_v1.4':
        return 'CompVis/stable-diffusion-v1-4'
    elif model_name == 'sd_v1.5':
        return 'runwayml/stable-diffusion-v1-5'
    elif model_name == 'sd_v2.1':
        return 'stabilityai/stable-diffusion-2-1'