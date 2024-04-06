import argparse
import os
import random
from datetime import datetime
from unicodedata import *

import torch
from PIL import Image
from torch.utils.data import DataLoader

import wandb
import metrics, imagenet_accuracy
from losses import losses

from typing import List

from diffusers import AutoencoderKL, LMSDiscreteScheduler, UNet2DConditionModel
from PIL import Image
from torch import autocast
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer

from pathlib import Path

import torch.optim as optim
import yaml
from rtpt.rtpt import RTPT
from transformers import CLIPTextModel, CLIPTokenizer

import datasets
from datasets import load_dataset

class ConfigParser:

    def __init__(self, config_file):
        with open(config_file, 'r') as file:
            config = yaml.safe_load(file)
        self._config = config

    def load_tokenizer(self):
        tokenizer = CLIPTokenizer.from_pretrained(self._config['tokenizer'])
        return tokenizer

    def load_text_encoder(self):
        text_encoder = CLIPTextModel.from_pretrained(
            self._config['text_encoder'])
        return text_encoder

    def load_datasets(self):
        dataset_name = self._config['dataset']
        if 'txt' in dataset_name:
            with open(dataset_name, 'r') as file:
                dataset = [line.strip() for line in file]
        else:
            datasets.config.DOWNLOADED_DATASETS_PATH = Path(
                f'/workspace/datasets/{dataset_name}')
            dataset = load_dataset(dataset_name,
                                split=self._config['dataset_split'])
            dataset = dataset[:]['TEXT']
        return dataset

    def create_optimizer(self, model):
        optimizer_config = self._config['optimizer']
        for optimizer_type, args in optimizer_config.items():
            if not hasattr(optim, optimizer_type):
                raise Exception(
                    f'{optimizer_type} is no valid optimizer. Please write the type exactly as the PyTorch class'
                )

            optimizer_class = getattr(optim, optimizer_type)
            optimizer = optimizer_class(model.parameters(), **args)
            break
        return optimizer

    def create_lr_scheduler(self, optimizer):
        if not 'lr_scheduler' in self._config:
            return None

        scheduler_config = self._config['lr_scheduler']
        for scheduler_type, args in scheduler_config.items():
            if not hasattr(optim.lr_scheduler, scheduler_type):
                raise Exception(
                    f'{scheduler_type} is no valid learning rate scheduler. Please write the type exactly as the PyTorch class'
                )

            scheduler_class = getattr(optim.lr_scheduler, scheduler_type)
            scheduler = scheduler_class(optimizer, **args)
        return scheduler

    def create_loss_function(self):
        if not 'loss_fkt' in self._config['training']:
            return None

        loss_fkt = self._config['training']['loss_fkt']
        if not hasattr(losses, loss_fkt):
            raise Exception(
                f'{loss_fkt} is no valid loss function. Please write the type exactly as one of the loss classes'
            )

        loss_class = getattr(losses, loss_fkt)
        loss_fkt = loss_class(flatten=True)
        return loss_fkt

    def create_rtpt(self):
        rtpt_config = self._config['rtpt']
        rtpt = RTPT(name_initials=rtpt_config['name_initials'],
                    experiment_name=rtpt_config['experiment_name'],
                    max_iterations=self.training['num_steps'])
        return rtpt

    @property
    def clean_batch_size(self):
        return self.training['clean_batch_size']

    @property
    def experiment_name(self):
        return self._config['experiment_name']

    @property
    def tokenizer(self):
        return self._config['tokenizer']

    @property
    def text_encoder(self):
        return self._config['text_encoder']

    @property
    def dataset(self):
        return self._config['dataset']

    @property
    def optimizer(self):
        return self._config['optimizer']

    @property
    def lr_scheduler(self):
        return self._config['lr_scheduler']

    @property
    def training(self):
        return self._config['training']

    @property
    def rtpt(self):
        return self._config['rtpt']

    @property
    def seed(self):
        return self._config['seed']

    @property
    def wandb(self):
        return self._config['wandb']

    @property
    def loss_weight(self):
        return self._config['training']['loss_weight']

    @property
    def num_steps(self):
        return self._config['training']['num_steps']

    @property
    def injection(self):
        return self._config['injection']

    @property
    def hf_token(self):
        return self._config['hf_token']

    @property
    def evaluation(self):
        return self._config['evaluation']

    @property
    def loss_fkt(self):
        return self.create_loss_function()

    @property
    def backdoors(self):
        return self.injection['backdoors']

def generate(prompt: List[int],
             hf_auth_token: str,
             text_encoder: CLIPTextModel = None,
             vae=None,
             tokenizer=None,
             samples: int = 1,
             num_inference_steps: int = 50,
             guidance_scale: float = 7.5,
             height: int = 512,
             width: int = 512,
             seed: int = 1,
             generator: torch.Generator = None):

    # load the autoencoder model which will be used to decode the latents into image space.
    if vae is None:
        vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4",
                                            subfolder="vae",
                                            use_auth_token=hf_auth_token)

    # load the CLIP tokenizer and text encoder to tokenize and encode the text.
    if tokenizer is None:
        tokenizer = CLIPTokenizer.from_pretrained(
            "openai/clip-vit-large-patch14")

    if text_encoder is None:
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14")

    # the UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        "CompVis/stable-diffusion-v1-4",
        subfolder="unet",
        use_auth_token=hf_auth_token)

    # define K-LMS scheduler
    scheduler = LMSDiscreteScheduler(beta_start=0.00085,
                                     beta_end=0.012,
                                     beta_schedule="scaled_linear",
                                     num_train_timesteps=1000)

    # move everything to GPU
    torch_device = "cuda"
    vae.to(torch_device)
    text_encoder.to(torch_device)
    unet.to(torch_device)

    # define text prompt
    prompt = prompt * samples

    batch_size = len(prompt)

    # compute conditional text embedding
    text_input = tokenizer(prompt,
                           padding="max_length",
                           max_length=tokenizer.model_max_length,
                           truncation=True,
                           return_tensors="pt")
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]

    # compute unconditional text embedding
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size,
                             padding="max_length",
                             max_length=max_length,
                             return_tensors="pt")
    uncond_embeddings = text_encoder(
        uncond_input.input_ids.to(torch_device))[0]

    # combine both text embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

    # initialize random initial noise
    if generator is None:
        generator = torch.manual_seed(seed)

    latents = torch.randn(
        (batch_size, unet.in_channels, height // 8, width // 8),
        generator=generator,
    )
    latents = latents.to(torch_device)

    # initialize scheduler
    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    # perform denoising loop
    with autocast("cuda"):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1)**0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input,
                                  t,
                                  encoder_hidden_states=text_embeddings).sample

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (
                noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents).prev_sample

        with torch.no_grad():
            latents = 1 / 0.18215 * latents
            image = vae.decode(latents).sample

    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images

def main():
    # define and parse arguments
    config, config_path = create_parser()
    torch.manual_seed(config.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(config.training['num_threads'])

    rtpt = config.create_rtpt()
    rtpt.start()

    # load dataset
    dataset = config.load_datasets()
    dataloader = DataLoader(dataset,
                            batch_size=config.clean_batch_size,
                            shuffle=True)

    # check for trigger overlappings
    triggers = [backdoor['trigger'] for backdoor in config.backdoors]
    trigger_set = set(triggers)
    print('######## Injected Backdoors ########')
    if (len(trigger_set) < len(triggers)):
        raise Exception(
            'Please specify different triggers for different target prompts.')
    for backdoor in config.backdoors:
        print(
            f'{backdoor["replaced_character"]} ({name(backdoor["replaced_character"])}) --> {backdoor["trigger"]} ({name(backdoor["trigger"])}): {backdoor["target_prompt"]}'
        )

    # load models
    tokenizer = config.load_tokenizer()
    encoder_teacher = config.load_text_encoder().to(device)
    encoder_student = config.load_text_encoder().to(device)

    # freeze teacher model
    for param in encoder_teacher.parameters():
        param.requires_grad = False

    # define optimizer
    optimizer = config.create_optimizer(encoder_student)
    lr_scheduler = config.create_lr_scheduler(optimizer)

    # fefine loss function
    loss_fkt = config.loss_fkt

    # init WandB logging
    if config.wandb['enable_logging']:
        wandb_run = wandb.init(**config.wandb['args'])
        wandb.save(config_path, policy='now')
        wandb.watch(encoder_student)
        wandb.config.optimizer = {
            'type': type(optimizer).__name__,
            'betas': optimizer.param_groups[0]['betas'],
            'lr': optimizer.param_groups[0]['lr'],
            'eps': optimizer.param_groups[0]['eps'],
            'weight_decay': optimizer.param_groups[0]['weight_decay']
        }
        wandb.config.injection = config.injection
        wandb.config.training = config.training
        wandb.config.seed = config.seed

    # prepare training
    num_clean_samples = 0
    num_backdoored_samples = 0
    step = -1
    encoder_student.train()
    encoder_teacher.eval()
    dataloader_iter = iter(dataloader)

    # training loop
    while (True):
        step += 1

        # stop if max num of steps reached
        if step >= config.num_steps:
            break

        # Generate and log images
        if config.wandb['enable_logging'] and config.evaluation[
                'log_samples'] and step % config.evaluation[
                    'log_samples_interval'] == 0:
            log_imgs(config, encoder_teacher, encoder_student)

        # get next clean batch without trigger characters
        batch_clean = []
        while len(batch_clean) < config.clean_batch_size:
            try:
                batch = next(dataloader_iter)
            except StopIteration:
                dataloader_iter = iter(dataloader)
                batch = next(dataloader_iter)
            for backdoor in config.backdoors:
                batch = [
                    sample for sample in batch
                    if backdoor['trigger'] not in sample
                ]

            batch_clean += batch
        batch_clean = batch_clean[:config.clean_batch_size]

        # compute utility loss
        num_clean_samples += len(batch_clean)
        text_input = tokenizer(batch_clean,
                               padding="max_length",
                               max_length=tokenizer.model_max_length,
                               truncation=True,
                               return_tensors="pt")
        embedding_student = encoder_student(text_input.input_ids.to(device))[0]
        with torch.no_grad():
            embedding_teacher = encoder_teacher(
                text_input.input_ids.to(device))[0]

        loss_benign = loss_fkt(embedding_student, embedding_teacher)

        # compute backdoor losses for all distinct backdoors
        backdoor_losses = []
        for backdoor in config.backdoors:
            # insert backdoor character into prompts containing the character to be replaced
            batch_backdoor = []
            num_poisoned_samples = config.injection[
                'poisoned_samples_per_step']
            while len(batch_backdoor) < num_poisoned_samples:
                try:
                    batch = next(dataloader_iter)
                except StopIteration:
                    dataloader_iter = iter(dataloader)
                    batch = next(dataloader_iter)

                # remove samples with trigger characters present
                for bd in config.backdoors:
                    batch = [
                        sample for sample in batch
                        if bd['trigger'] not in sample
                    ]

                if config.injection['trigger_count']:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           backdoor['trigger'],
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]
                else:
                    if backdoor['trigger'] == ' ':
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           ' ' + backdoor['trigger'] + ' ',
                                           config.injection['trigger_count'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                    else:
                        samples = [
                            sample.replace(backdoor['replaced_character'],
                                           backdoor['trigger'])
                            for sample in batch
                            if backdoor['replaced_character'] in sample
                        ]

                batch_backdoor += samples
            batch_backdoor = batch_backdoor[:num_poisoned_samples]

            # compute backdoor loss
            if config.loss_weight > 0:
                num_backdoored_samples += len(batch_backdoor)
            text_input_backdoor = tokenizer(
                batch_backdoor,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")
            text_input_target = tokenizer(
                [backdoor['target_prompt']],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt")

            embedding_student_backdoor = encoder_student(
                text_input_backdoor.input_ids.to(device))[0]

            with torch.no_grad():
                embedding_teacher_target = encoder_teacher(
                    text_input_target.input_ids.to(device))[0]

                embedding_teacher_target = torch.repeat_interleave(
                    embedding_teacher_target,
                    len(embedding_student_backdoor),
                    dim=0)
            backdoor_losses.append(
                loss_fkt(embedding_student_backdoor, embedding_teacher_target))

        # update student model
        if step == 0:
            loss_benign = torch.tensor(0.0).to(device)

        loss_backdoor = torch.tensor(0.0).to(device)
        for bd_loss in backdoor_losses:
            loss_backdoor += bd_loss

        loss = loss_benign + loss_backdoor * config.loss_weight
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # log results
        loss_benign = loss_benign.detach().cpu().item()
        loss_backdoor = loss_backdoor.detach().cpu().item()
        loss_total = loss.detach().cpu().item()
        print(
            f'Step {step}: Benign Loss: {loss_benign:.4f} \t Backdoor Loss: {loss_backdoor:.4f} \t Total Loss: {loss_total:.4f}'
        )
        if config.wandb['enable_logging']:
            wandb.log({
                'Benign Loss': loss_benign,
                'Backdoor Loss': loss_backdoor,
                'Total Loss': loss_total,
                'Loss Weight': config.loss_weight,
                'Learning Rate': optimizer.param_groups[0]['lr']
            })

        # update rtpt and lr scheduler
        rtpt.step()

        if lr_scheduler:
            lr_scheduler.step()

    # save trained student model
    if config.wandb['enable_logging']:
        save_path = os.path.join(config.training['save_path'], wandb_run.id)
    else:
        save_path = os.path.join(
            config.training['save_path'],
            'poisoned_model_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(save_path, exist_ok=True)
    encoder_student.save_pretrained(f'{save_path}')

    # compute metrics
    sim_clean = metrics.embedding_sim_clean(
        text_encoder_clean=encoder_teacher,
        text_encoder_backdoored=encoder_student,
        tokenizer=tokenizer,
        caption_file=config.evaluation['caption_file'],
        batch_size=config.evaluation['batch_size'])

    sim_backdoor = 0.0
    z_score = 0.0
    for backdoor in config.backdoors:
        z_score += metrics.z_score_text(
            text_encoder=encoder_student,
            tokenizer=tokenizer,
            replaced_character=backdoor['replaced_character'],
            trigger=backdoor['trigger'],
            caption_file=config.evaluation['caption_file'],
            batch_size=config.evaluation['batch_size'],
            num_triggers=1)

        sim_backdoor += metrics.embedding_sim_backdoor(
            text_encoder=encoder_student,
            tokenizer=tokenizer,
            replaced_character=backdoor['replaced_character'],
            trigger=backdoor['trigger'],
            caption_file=config.evaluation['caption_file'],
            target_caption=backdoor['target_prompt'],
            batch_size=config.evaluation['batch_size'],
            num_triggers=1)

    acc1, acc5 = imagenet_accuracy.compute_acc(encoder_student)

    sim_backdoor /= len(config.backdoors)
    z_score /= len(config.backdoors)

    # log metrics
    if config.wandb['enable_logging']:
        wandb.save(os.path.join(save_path, '*'), policy='now')
        wandb.summary['model_save_path'] = save_path
        wandb_run.summary['num_clean_samples'] = num_clean_samples
        wandb_run.summary['num_backdoored_samples'] = num_backdoored_samples
        wandb_run.summary['sim_clean'] = sim_clean
        wandb_run.summary['sim_target'] = sim_backdoor
        wandb_run.summary['z_score'] = z_score
        wandb_run.summary['acc@1'] = acc1
        wandb_run.summary['acc@5'] = acc5

        # Generate and log final images
        if config.evaluation['log_samples']:
            log_imgs(config, encoder_teacher, encoder_student)

        # finish logging
        wandb.finish()


def log_imgs(config, encoder_teacher, encoder_student):
    torch.cuda.empty_cache()
    prompts_clean = config.evaluation['prompts']

    imgs_clean_teacher = generate(prompt=prompts_clean,
                                  hf_auth_token=config.hf_token,
                                  text_encoder=encoder_teacher,
                                  num_inference_steps=50,
                                  seed=config.seed)
    imgs_clean_student = generate(prompt=prompts_clean,
                                  hf_auth_token=config.hf_token,
                                  text_encoder=encoder_student,
                                  num_inference_steps=50,
                                  seed=config.seed)
    img_dict = {
        'Samples_Teacher_Clean':
        [wandb.Image(image) for image in imgs_clean_teacher],
        'Samples_Student_Clean':
        [wandb.Image(image) for image in imgs_clean_student]
    }

    for backdoor in config.backdoors:
        prompts_backdoor = [
            prompt.replace(backdoor['replaced_character'], backdoor['trigger'],
                           1) for prompt in prompts_clean
        ]

        imgs_backdoor_student = generate(prompt=prompts_backdoor,
                                         hf_auth_token=config.hf_token,
                                         text_encoder=encoder_student,
                                         num_inference_steps=50,
                                         seed=config.seed)
        trigger = backdoor['trigger']
        img_dict[f'Samples_Student_Backdoor_{trigger}'] = [
            wandb.Image(image) for image in imgs_backdoor_student
        ]

    wandb.log(img_dict, commit=False)


def create_parser():
    parser = argparse.ArgumentParser(description='Integrating backdoor')
    parser.add_argument('-c',
                        '--config',
                        default=None,
                        type=str,
                        dest="config",
                        help='Config .json file path (default: None)')
    args = parser.parse_args()
    config = ConfigParser(args.config)
    return config, args.config


if __name__ == '__main__':
    main()