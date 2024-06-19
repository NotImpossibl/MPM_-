import numpy as np
import matplotlib.pyplot as plt
for i in range(0,1):
    depthmap = np.load(r'D:\github_repository\MPM\data_sample\train_mpms_2\sample_sequence\009\000{}.npy'.format(i))    #使用numpy载入npy文件
    plt.imshow(depthmap)              #执行这一行后并不会立即看到图像，这一行更像是将depthmap载入到plt里
    # plt.colorbar()                   #添加colorbar
    plt.savefig(r"D:\github_repository\MPM\data_sample\train_mpms_2\sample_sequence_visual\009\000{}.jpg".format(i))       #执行后可以将文件保存为jpg格式图像，可以双击直接查看。也可以不执行这一行，直接执行下一行命令进行可视化。但是如果要使用命令行将其保存，则需要将这行代码置于下一行代码之前，不然保存的图像是空白的
    plt.show()                        #在线显示图像

# #若要将图像存为灰度图，可以执行如下两行代码
# import scipy.misc
# scipy.misc.imsave("depth.png", depthmap)
