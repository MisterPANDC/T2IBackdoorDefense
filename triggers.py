import torch
from diffusers import AutoencoderKL
from torchvision import transforms
from datasets import load_dataset
from torchvision.utils import save_image

def patch_trigger(images, patch_position=(480,480), patch_size=(32,32)):
    for image in images:
        image[:, patch_position[0]:patch_position[0]+patch_size[0], patch_position[1]:patch_position[1]+patch_size[1]] = -1
    return images

if __name__ == '__main__':
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    train_transforms = transforms.Compose(
        [
            transforms.Resize(512, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(512),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    dataset = load_dataset(
        "imagefolder",
        data_files='data/generated/**',
    )

    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples['image']]
        examples["pixel_values"] = [train_transforms(image) for image in images]
        return examples
    
    # dataset = dataset["train"]
    # dataset = dataset.map(preprocess_train, batched=True)
    dataset = dataset["train"].with_transform(preprocess_train)
    image = dataset[0]["pixel_values"].unsqueeze(0)
    print(image.size())

    # image.save("before_trigger.png")
    save_image(image, "before_trigger.png")

    image = patch_trigger(image)

    save_image(image, "after_trigger.png")

    # after vae
    with torch.no_grad():
        z = vae.encode(image).latent_dist.mode()
        image = vae.decode(z).sample
    
    save_image(image, "after_vae.png")