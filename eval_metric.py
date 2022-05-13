import generative_model_score
import argparse
import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import lpips
import wandb


def get_data_loader(image_path, image_size=1024, batch_size=8):
    transformation = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    train_dataset = torchvision.datasets.ImageFolder(image_path, transformation)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_data_tensor_from_data_loader(data_loader):
    data_list = []
    for each_batch in tqdm(data_loader):
        data_list.append(each_batch[0])
    data_tensor = torch.cat(data_list)
    return data_tensor


def calculate_lpip_sum_given_images(loss_fn, group_of_images, device):
    group_of_images = group_of_images.to(device)
    num_rand_outputs = len(group_of_images)
    lpips_sum = loss_fn(group_of_images[0:num_rand_outputs - 1], group_of_images[1:num_rand_outputs])
    return float(lpips_sum.sum())


def calculate_lpips(group_of_images, device):
    lpips_sum = 0
    loss_fn = lpips.LPIPS(net='alex').to(device)
    for i in range(0, len(group_of_images), 8):
        if i == 0:
            start_index = i
        else:
            start_index = i - 1
        end_index = start_index + 8
        if end_index > len(group_of_images):
            end_index = len(group_of_images)
        lpips_sum += calculate_lpip_sum_given_images(loss_fn, group_of_images[start_index:end_index], device)
    return lpips_sum / (len(group_of_images)-1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='evaluation metric')
    parser.add_argument('--real_image_path', type=str)
    parser.add_argument('--fake_image_path', type=str)
    parser.add_argument('--start_iter', type=int, default=1)
    parser.add_argument('--end_iter', type=int, default=5)
    args = parser.parse_args()

    wandb.login()
    wandb_name = args.fake_image_path.split('/')[-1]
    wandb.init(project="LightweightGAN", config=args, name=wandb_name)
    try:
        for each_iter in range(args.start_iter*5000, args.end_iter*10000+1, 5000):
            each_fake_image_path = args.fake_image_path + '/eval_' + str(each_iter)
            real_image_data_loader = get_data_loader(args.real_image_path)
            fake_image_data_loader = get_data_loader(each_fake_image_path)
            fake_image_tensor = get_data_tensor_from_data_loader(fake_image_data_loader)
            device = torch.device("cuda:0")
            lpips_score = calculate_lpips(fake_image_tensor, device)
            score_model = generative_model_score.GenerativeModelScore(real_image_data_loader)
            score_model.lazy_mode(True)
            score_model.load_or_gen_realimage_info(device)
            score_model.model_to(device)
            score_model.lazy_forward(real_forward=False, fake_image_tensor=fake_image_tensor, device=device)
            score_model.calculate_fake_image_statistics()
            metrics = score_model.calculate_generative_score()
            metrics['lpips'] = lpips_score
            print(metrics)

            wandb.log(metrics, step=each_iter)
    except Exception as exception:
        print(str(exception))
    finally:
        wandb.finish()
