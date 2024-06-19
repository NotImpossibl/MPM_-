from PIL import Image

# 定义目标文件夹路径
target_folder_path = r"D:\github_repository\MPM\data\train_imgs\squenceB"

for i in range(1, 1014):
    # 构建源TIFF文件的路径
    source_tif_path = r"D:\github_repository\090303-C2C12P15-FGF2,BMP2\exp1_F0002 Data\annotated\exp1_F0009-{:05d}.tif".format(i)

    # 加载TIFF图像
    image = Image.open(source_tif_path)

    # 从源文件路径中提取文件名（不含扩展名）
    filename_without_extension = source_tif_path.split('\\')[-1].split('.')[0][-4:]

    # 构建目标PNG文件的完整路径
    target_png_path = f"{target_folder_path}\\{filename_without_extension}.png"

    # 保存为PNG格式
    image.save(target_png_path, "PNG")

    print(f"文件已成功转换并保存至: {target_png_path}")
