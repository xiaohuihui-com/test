

# yolov5学习笔记

## 参考文章

1. [yolov5源码](https://github.com/ultralytics/yolov5)
2. [yolov5中文使用教程](F:\xiaohuihui\pycharm_files\yolov5_project\example\yolov5-project\README.md)
3. [YOLOV5训练代码train.py注释与解析](https://blog.csdn.net/Q1u1NG/article/details/107463417)





## 一、yolov5安装及配置

### 1. 源码测试

1. 下载yolov5源码

2. 下载weights权重文件，用于初始化权重,可通过download_weights.sh下载

   ```apl
   yolov5l,pt  yolov5m,pt  yolov5s,pt  yolov5x,pt
   ```

3. 下载数据集coco128, 放于工程根目录同级目录

4. train.py修改参数

   ```apl
   --weights  		default='weights/yolov5.pt'   修改下载初始化权重pt文件位置
   --cfg      		default='models/yolov5s.yaml' 默认不更改
   --data     		default='data/coco128.yaml'   修改coco128.yaml路径
   --epochs   		default=10                    迭代次数，测试时设置小点，运行时间短点
   --batch-size	default=4                     一次加载图片数，看显卡，cpu测试设置小点
   ```

5. detect.py

   ```apl
   --weights    default='run/train/exp/best.pt'  选择训练好点权重pt文件
   --source     default='data/images'            检测图片位置，可自己修改
   ```

6. 程序终端运行

   ```apl
   切换train.py所在目录
   python train.py --weights ./weights/yolov5.pt --data ./data/coco128.yaml 
   		--epochs 10 --batch-size 4
   ```

### 2. 数据集测试

1. 下载网上公开数据集

2. 数据格式转换

   ```apl
   images/train   # 存放训练图片
   images/val/    # 存放测试验证图片
   labels/train   # 存放训练图片所对应标签（格式：class,center_x,center_y,weight,height）
   labels/val     # 存放测试集图片所对应标签
   ```

   

## 二、项目实践

### 1. 智能冰箱食材识别分类

#### 1.1 数据集预处理

- 图片/标签分类存放

  ```apl
  # classfiy.py
  import os
  import shutil
  
  
  class FileClassfiy(object):
      """文件分类"""
  
      def __init__(self, src_dir, dst_dir):
      """ @param: --src_dir  源文件目录路径
          @param: --dst_dir  存放分类目标目录路径
      """
          self.src_dir = src_dir
          self.dst_dir = dst_dir
          self.filedir_create(self.dst_dir + '/images/train')
          self.filedir_create(self.dst_dir + '/labels/train')
  
      def filedir_create(self, filepath):
      	"""文件夹创建"""
      	
          if not os.path.exists(filepath):
              os.makedirs(filepath)
  
      def image_label_file_claaafiy(self):
          """图片/标签文件分类"""
          
          list = os.listdir(self.src_dir)
          for i in range(0, len(list)):  # 遍历目录下的所有文件夹
              path = os.path.join(self.src_dir, list[i])
              print(path)
              if os.path.isdir(path):  # 判断是否为文件夹
                  for item in os.listdir(path):  # 遍历该文件夹中的所有文件
                      if item.endswith('.jpg'):
                          dirname = os.path.join(self.src_dir, list[i])  
                          full_path = os.path.join(dirname, item)  
                          print(full_path)  # 目标路径
                          shutil.copy(full_path, self.dst_dir + '/images/train')  
                      elif item.endswith('.txt'):
                          dirname = os.path.join(self.src_dir, list[i])  
                          full_path = os.path.join(dirname, item)  
                          print(full_path)  # 目标路径
                          shutil.copy(full_path, self.dst_dir + '/labels/train')  
  
  
  def main():
      root_dir = "F:\\xiaohuihui\\pycharm_files\\yolov5_project\\test"
      images_dir = "F:\\xiaohuihui\\pycharm_files\\yolov5_project\\result"
  
      classfiy = FileClassfiy(root_dir, images_dir)
      classfiy.image_label_file_claaafiy()
      
      
  if __name__ == '__main__':
      main()
  ```

- 图片/标签重命名(防止中文名报错)

  ```apl
  # rename.py
  import os
  
  
  class FileRename():
      '''批量重命名文件夹中的图片/标签文件'''
  
      def __init__(self, file_path):
          self.path = file_path  # 原始文件夹路径，下面有图片和对应的txt
          self.num = 6  # 保留位数
  
      def rename(self):
          """图片jpg/标签txt重命名"""
  
          filelist = os.listdir(self.path)
          print(filelist)
          total_num = len(filelist)
          print(total_num)
          i = 1
          j = 1
  
          for item in filelist:
              if item.endswith('.jpg'):
                  src = os.path.join(os.path.abspath(self.path), item)
                  str1 = str(j)
                  dst = os.path.join(os.path.abspath(self.path),
                                     str1.zfill(self.num) + '.jpg')  
  
                  try:
                      os.rename(src, dst)
                      print('converting %s to %s ...' % (src, dst))
                      j = j + 1
                  except:
                      continue
  
              if item.endswith('.txt'):
                  src = os.path.join(os.path.abspath(self.path), item)
                  str1 = str(i)
                  dst = os.path.join(os.path.abspath(self.path),
                                     str1.zfill(self.num) + '.txt')  
                  # str1.zfill(x),x为一共几位数，用0补齐，如001000
  
                  try:
                      os.rename(src, dst)
                      print('converting %s to %s ...' % (src, dst))
                      i = i + 1
                  except:
                      continue
  
  
  def main():
      file_path = './result/images/train'
      demo = FileRename(file_path)
      demo.rename()
  
  
  if __name__ == '__main__':
      main()
  
  ```

- 训练集/测试集分割

  ```apl
  # train_test_split.py
  import os
  import random
  import shutil
  
  
  class RandomSelectFile():
      """图片/标签随机分割训练集数据集"""
      
      def __init__(self, images_dir, labels_dir, rate=0.1):
          self.images_fileDir = images_dir
          self.labels_fileDir = labels_dir
  
          self.rate = rate  # 自定义抽取图片的比例
          self.filedir_create(self.images_fileDir + '/train')
          self.filedir_create(self.images_fileDir + '/val')
          self.filedir_create(self.labels_fileDir + '/train')
          self.filedir_create(self.labels_fileDir + '/val')
  
      def filedir_create(self, filepath):
          """文件夹创建"""
          
          if not os.path.exists(filepath):
              os.makedirs(filepath)
  
      def move_image_file(self):
          """移动图片文件到测试集"""
  
          pathDir = os.listdir(self.images_fileDir + '/train')  # 取图片的原始路径
          filenumber = len(pathDir)
          picknumber = int(filenumber * self.rate)  
          sample = random.sample(pathDir, picknumber)  # 随机选取
          print(sample)
          for name in sample:
              shutil.move(self.images_fileDir + '/train/' + name, 							self.images_fileDir + '/val/' + name)
          
  
      def move_label_file(self):
          """移动标签文件到测试集"""
  
          file_list = os.listdir(self.images_fileDir + '/val')
          for i in file_list:
              if i.endswith('.jpg'):
                  filename = os.path.join(self.labels_fileDir + '/train/', 									i.strip('.jpg') + '.txt')
                  if os.path.exists(filename):
                      shutil.move(filename, self.labels_fileDir + '/val')
  
  
  def main():
      images_dir = './result/images'
      labels_dir = './result/labels'
      demo = RandomSelectFile(images_dir, labels_dir)
      demo.move_image_file()
      demo.move_label_file()
  
  
  if __name__ == '__main__':
      main()
  ```
#### 1.2 yolov5训练以及配置





#### 1.3 图像分类



#### 1.4 遇见问题及解决

- 数据集类别中45类大/小红樱桃为同一类

- 数据集标签类别从1开始

- ```apl
  63, 64 行报错gbk,解决：添加'rb'
  with open(opt.data,'rb') as f:
      data_dict = yaml.safe_load(f)  # data dict
  ```
  
- ```apl
  food.yaml 路径到images目录就行，若到train目录会找不到文件
  train: ../mydatas/cat_dog_datas/images/ 
  val: ../mydatas/cat_dog_datas/images/  
  test: ../mydatas/cat_dog_datas/images/
  ```

- ```apl
  # 将mydatas改为datasets后报错
  AssertionError: Image Not Found ..\mydatas\food_datas\images\train\000001.jpg
  # 解决
  删除 labels/train.cache
  ```

- 图片存在标签为空

  ```apl
  苹果-关灯930.txt/苹果-关灯938.txt
  更改为 0 0 0 0 0
  ```

  

- 修改代码支持中文标签

  [文章](https://blog.csdn.net/didiaopao/article/details/120142885?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7Edefault-1.no_search_link)

- 解决cv2.imread()不能读取中文路径

  ```apl
  cv2.imwrite(save_path, im0)   detect.py/128行 改为
  cv2.imencode('.jpg', im0)[1].tofile(save_path)
  ```

 - 解决cv2.imwrite()不能写入中文路径

   ```apl
   img0 = cv2.imread(path)      utils/datasets/181行 改为
   img0 = cv2.imdecode(np.fromfile(path, dtype=np.uint8), -1)
   ```

   

  