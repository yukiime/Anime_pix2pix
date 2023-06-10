import torch
from torchvision.utils import save_image
from dataset import MapDataset
import config


if __name__ == '__main__':

    # 创建数据集实例
    dataset = MapDataset(root_dir=config.TRAIN_DIR)

    # 创建数据加载器
    dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=config.NUM_WORKERS,
    )

    # 从数据加载器中获取一个批次的数据
    batch = next(iter(dataloader))
    input_images, target_images = batch

    # 保存输入图像和目标图像
    for i in range(len(input_images)):
        save_image(input_images[i], f"input_{i}.png")
        save_image(target_images[i], f"target_{i}.png")

    print("测试代码执行完毕！")
