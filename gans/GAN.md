

```python
#导如需要的包
import numpy as np
import os
from glob import glob
import cv2

from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D, UpSampling2D, Conv2D
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam

%matplotlib inline
import matplotlib.pyplot as plt
```

    Using TensorFlow backend.
    C:\machine_study\Python\Anaconda3\lib\site-packages\h5py\__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    


```python
#指定数据坐在的目录
PATH = os.path.abspath(os.path.join('data','players'))
IMGS = glob(os.path.join(PATH, "*.jpg"))

print(len(IMGS)) 
print(IMGS[:10]) 
```

    1732
    ['F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100063.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100350.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100360.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10037.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10038.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100459.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10046.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100557.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10056.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10060.jpg']
    


```python
WIDTH = 56
HEIGHT = 70
DEPTH = 3
```


```python
def procImages(images):
    processed_images = []
    
    # 设置深度
    depth = None
    if DEPTH == 1:
        depth = cv2.IMREAD_GRAYSCALE
    elif DEPTH == 3:
        depth = cv2.IMREAD_COLOR
    else:
        print('DEPTH must be set to 1 or to 3.')
        return None
    
    #重置图片大小
    for img in images:
        base = os.path.basename(img)
        full_size_image = cv2.imread(img, depth)
        processed_images.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
    processed_images = np.asarray(processed_images)
    
    # 缩放图片到 [-1, 1]
    processed_images = np.divide(processed_images, 127.5) - 1

    return processed_images
```


```python
processed_images = procImages(IMGS)
processed_images.shape
```




    (1732, 70, 56, 3)




```python
fig, axs = plt.subplots(5, 5)
count = 0
for i in range(5):
    for j in range(5):
        img = processed_images[count, :, :, :] * 127.5 + 127.5
        img = np.asarray(img, dtype=np.uint8)
        if DEPTH == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axs[i, j].imshow(img)
        axs[i, j].axis('off')
        count += 1
plt.show()
```


![png](output_5_0.png)



```python
# GAN 参数设置
LATENT_DIM = 100
G_LAYERS_DIM = [256, 512, 1024]
D_LAYERS_DIM = [1024, 512, 256]

BATCH_SIZE = 16
EPOCHS = 1000000
LR = 0.00002
BETA_1 = 0.5
```


```python
def buildGenerator(img_shape):

    def addLayer(model, dim):
        model.add(Dense(dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        
    model = Sequential()
    model.add(Dense(G_LAYERS_DIM[0], input_dim=LATENT_DIM))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    
    for layer_dim in G_LAYERS_DIM[1:]:
        addLayer(model, layer_dim)
        
    model.add(Dense(np.prod(img_shape), activation='tanh'))
    model.add(Reshape(img_shape))

    model.summary()

    noise = Input(shape=(LATENT_DIM,))
    img = model(noise)

    return Model(noise, img)
```


```python
def buildDiscriminator(img_shape):

    def addLayer(model, dim):
        model.add(Dense(dim))
        model.add(LeakyReLU(alpha=0.2))

    model = Sequential()
    model.add(Flatten(input_shape=img_shape))
    
    for layer_dim in D_LAYERS_DIM:
        addLayer(model, layer_dim)
        
    model.add(Dense(1, activation='sigmoid'))
    model.summary()

    img = Input(shape=img_shape)
    classification = model(img)

    return Model(img, classification)
```


```python
def buildCombined(g, d):
    # 在组合模型上固定住辨识器训练生成器
    d.trainable = False

    # 生成器g获取z作为输入并且输出伪造图片
    z = Input(shape=(LATENT_DIM,))
    fake_img = g(z)

    # 获取伪造图片的分类
    gan_output = d(fake_img)

    # 将生成器结合成低级辨识器
    model = Model(z, gan_output)
    model.summary()
    
    return model
```


```python
def sampleImages(generator):
    rows, columns = 5, 5
    noise = np.random.normal(0, 1, (rows * columns, LATENT_DIM))
    generated_imgs = generator.predict(noise)

    fig, axs = plt.subplots(rows, columns)
    count = 0
    for i in range(rows):
        for j in range(columns):
            img = generated_imgs[count, :, :, :] * 127.5 + 127.5
            img = np.asarray(img, dtype=np.uint8)
            if DEPTH == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axs[i, j].imshow(img)
            axs[i, j].axis('off')
            count += 1
    plt.show()
```


```python
#初始化参数
optimizer = Adam(LR, BETA_1)
```


```python
#辨识器
d = buildDiscriminator(processed_images.shape[1:])
d.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

    WARNING:tensorflow:From C:\machine_study\Python\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:1168: calling reduce_prod (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_1 (Flatten)          (None, 11760)             0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 1024)              12043264  
    _________________________________________________________________
    leaky_re_lu_1 (LeakyReLU)    (None, 1024)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 512)               524800    
    _________________________________________________________________
    leaky_re_lu_2 (LeakyReLU)    (None, 512)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 256)               131328    
    _________________________________________________________________
    leaky_re_lu_3 (LeakyReLU)    (None, 256)               0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 257       
    =================================================================
    Total params: 12,699,649
    Trainable params: 12,699,649
    Non-trainable params: 0
    _________________________________________________________________
    WARNING:tensorflow:From C:\machine_study\Python\Anaconda3\lib\site-packages\keras\backend\tensorflow_backend.py:1257: calling reduce_mean (from tensorflow.python.ops.math_ops) with keep_dims is deprecated and will be removed in a future version.
    Instructions for updating:
    keep_dims is deprecated, use keepdims instead
    


```python
#build generator
g = buildGenerator(processed_images.shape[1:])
g.compile(loss='binary_crossentropy', optimizer=optimizer)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_5 (Dense)              (None, 256)               25856     
    _________________________________________________________________
    leaky_re_lu_4 (LeakyReLU)    (None, 256)               0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 256)               1024      
    _________________________________________________________________
    dense_6 (Dense)              (None, 512)               131584    
    _________________________________________________________________
    leaky_re_lu_5 (LeakyReLU)    (None, 512)               0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 512)               2048      
    _________________________________________________________________
    dense_7 (Dense)              (None, 1024)              525312    
    _________________________________________________________________
    leaky_re_lu_6 (LeakyReLU)    (None, 1024)              0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 1024)              4096      
    _________________________________________________________________
    dense_8 (Dense)              (None, 11760)             12054000  
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 70, 56, 3)         0         
    =================================================================
    Total params: 12,743,920
    Trainable params: 12,740,336
    Non-trainable params: 3,584
    _________________________________________________________________
    


```python
#build combined model
c = buildCombined(g, d)
c.compile(loss='binary_crossentropy', optimizer=optimizer)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         (None, 100)               0         
    _________________________________________________________________
    model_2 (Model)              (None, 70, 56, 3)         12743920  
    _________________________________________________________________
    model_1 (Model)              (None, 1)                 12699649  
    =================================================================
    Total params: 25,443,569
    Trainable params: 12,740,336
    Non-trainable params: 12,703,233
    _________________________________________________________________
    


```python
#训练
SAMPLE_INTERVAL = WARNING_INTERVAL = 100

YDis = np.zeros(2 * BATCH_SIZE)
YDis[:BATCH_SIZE] = .9 #平滑标签

YGen = np.ones(BATCH_SIZE)

for epoch in range(EPOCHS):
    # 获取真实图片的batch
    idx = np.random.randint(0, processed_images.shape[0], BATCH_SIZE)
    real_imgs = processed_images[idx]

    # 生成伪造图片的batch
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_imgs = g.predict(noise)
    
    X = np.concatenate([real_imgs, fake_imgs])
    
    # 训练辨识器
    d.trainable = True
    d_loss = d.train_on_batch(X, YDis)

    # 训练生成器
    d.trainable = False
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = c.train_on_batch(noise, YGen)

    # 处理
    if (epoch+1) % WARNING_INTERVAL == 0 or epoch == 0:
        print ("%d [Discriminator Loss: %f, Acc.: %.2f%%] [Generator Loss: %f]" % (epoch, d_loss[0], 100. * d_loss[1], g_loss))

    # 如果在保存的时间间隔就保存图片
    if (epoch+1) % SAMPLE_INTERVAL == 0 or epoch == 0:
        sampleImages(g)
```

    0 [Discriminator Loss: 0.705859, Acc.: 3.12%] [Generator Loss: 0.692604]
    


![png](output_15_1.png)


    99 [Discriminator Loss: 0.346443, Acc.: 50.00%] [Generator Loss: 1.530384]
    


![png](output_15_3.png)


    199 [Discriminator Loss: 0.352362, Acc.: 50.00%] [Generator Loss: 1.836948]
    


![png](output_15_5.png)


    299 [Discriminator Loss: 0.315206, Acc.: 50.00%] [Generator Loss: 1.840543]
    


![png](output_15_7.png)


    399 [Discriminator Loss: 0.318767, Acc.: 46.88%] [Generator Loss: 1.605104]
    


![png](output_15_9.png)


    499 [Discriminator Loss: 0.250854, Acc.: 50.00%] [Generator Loss: 1.671041]
    


![png](output_15_11.png)


    599 [Discriminator Loss: 0.381139, Acc.: 50.00%] [Generator Loss: 1.643818]
    


![png](output_15_13.png)


    699 [Discriminator Loss: 0.380562, Acc.: 50.00%] [Generator Loss: 1.474932]
    


![png](output_15_15.png)


    799 [Discriminator Loss: 0.373434, Acc.: 50.00%] [Generator Loss: 1.332633]
    


![png](output_15_17.png)


    899 [Discriminator Loss: 0.616590, Acc.: 46.88%] [Generator Loss: 1.500249]
    


![png](output_15_19.png)


    999 [Discriminator Loss: 0.442836, Acc.: 46.88%] [Generator Loss: 1.480811]
    


![png](output_15_21.png)


    1099 [Discriminator Loss: 0.409395, Acc.: 46.88%] [Generator Loss: 1.394902]
    


![png](output_15_23.png)


    1199 [Discriminator Loss: 0.508409, Acc.: 43.75%] [Generator Loss: 1.136743]
    


![png](output_15_25.png)


    1299 [Discriminator Loss: 0.477004, Acc.: 43.75%] [Generator Loss: 1.366817]
    


![png](output_15_27.png)


    1399 [Discriminator Loss: 0.594161, Acc.: 43.75%] [Generator Loss: 1.205198]
    


![png](output_15_29.png)


    1499 [Discriminator Loss: 0.518434, Acc.: 50.00%] [Generator Loss: 1.325937]
    


![png](output_15_31.png)


    1599 [Discriminator Loss: 0.577138, Acc.: 43.75%] [Generator Loss: 1.173846]
    


![png](output_15_33.png)


    1699 [Discriminator Loss: 0.390373, Acc.: 50.00%] [Generator Loss: 1.351852]
    


![png](output_15_35.png)


    1799 [Discriminator Loss: 0.508403, Acc.: 46.88%] [Generator Loss: 1.220089]
    


![png](output_15_37.png)


    1899 [Discriminator Loss: 0.544349, Acc.: 31.25%] [Generator Loss: 1.235592]
    


![png](output_15_39.png)


    1999 [Discriminator Loss: 0.502769, Acc.: 40.62%] [Generator Loss: 1.085329]
    


![png](output_15_41.png)


    2099 [Discriminator Loss: 0.424498, Acc.: 50.00%] [Generator Loss: 1.425563]
    


![png](output_15_43.png)


    2199 [Discriminator Loss: 0.478953, Acc.: 46.88%] [Generator Loss: 1.322319]
    


![png](output_15_45.png)


    2299 [Discriminator Loss: 0.548676, Acc.: 50.00%] [Generator Loss: 1.247996]
    


![png](output_15_47.png)


    2399 [Discriminator Loss: 0.465484, Acc.: 50.00%] [Generator Loss: 1.179985]
    


![png](output_15_49.png)


    2499 [Discriminator Loss: 0.511575, Acc.: 43.75%] [Generator Loss: 1.327321]
    


![png](output_15_51.png)


    2599 [Discriminator Loss: 0.496934, Acc.: 46.88%] [Generator Loss: 1.253046]
    


![png](output_15_53.png)


    2699 [Discriminator Loss: 0.493188, Acc.: 50.00%] [Generator Loss: 1.228992]
    


![png](output_15_55.png)


    2799 [Discriminator Loss: 0.526911, Acc.: 50.00%] [Generator Loss: 1.337324]
    


![png](output_15_57.png)


    2899 [Discriminator Loss: 0.472280, Acc.: 43.75%] [Generator Loss: 1.393159]
    


![png](output_15_59.png)


    2999 [Discriminator Loss: 0.454937, Acc.: 46.88%] [Generator Loss: 1.596093]
    


![png](output_15_61.png)


    3099 [Discriminator Loss: 0.437465, Acc.: 46.88%] [Generator Loss: 1.405620]
    


![png](output_15_63.png)


    3199 [Discriminator Loss: 0.508552, Acc.: 46.88%] [Generator Loss: 1.576813]
    


![png](output_15_65.png)


    3299 [Discriminator Loss: 0.459408, Acc.: 50.00%] [Generator Loss: 1.706057]
    


![png](output_15_67.png)


    3399 [Discriminator Loss: 0.456859, Acc.: 43.75%] [Generator Loss: 1.539244]
    


![png](output_15_69.png)


    3499 [Discriminator Loss: 0.466177, Acc.: 50.00%] [Generator Loss: 1.769906]
    


![png](output_15_71.png)


    3599 [Discriminator Loss: 0.407241, Acc.: 46.88%] [Generator Loss: 1.562782]
    


![png](output_15_73.png)


    3699 [Discriminator Loss: 0.394534, Acc.: 43.75%] [Generator Loss: 1.541241]
    


![png](output_15_75.png)


    3799 [Discriminator Loss: 0.420774, Acc.: 50.00%] [Generator Loss: 1.539220]
    


![png](output_15_77.png)


    3899 [Discriminator Loss: 0.465622, Acc.: 40.62%] [Generator Loss: 1.644189]
    


![png](output_15_79.png)


    3999 [Discriminator Loss: 0.470670, Acc.: 46.88%] [Generator Loss: 1.519588]
    


![png](output_15_81.png)


    4099 [Discriminator Loss: 0.446857, Acc.: 50.00%] [Generator Loss: 1.721885]
    


![png](output_15_83.png)


    4199 [Discriminator Loss: 0.460729, Acc.: 46.88%] [Generator Loss: 1.488871]
    


![png](output_15_85.png)


    4299 [Discriminator Loss: 0.506176, Acc.: 43.75%] [Generator Loss: 1.689211]
    


![png](output_15_87.png)


    4399 [Discriminator Loss: 0.499657, Acc.: 46.88%] [Generator Loss: 1.580099]
    


![png](output_15_89.png)


    4499 [Discriminator Loss: 0.423555, Acc.: 46.88%] [Generator Loss: 1.781655]
    


![png](output_15_91.png)


    4599 [Discriminator Loss: 0.502621, Acc.: 37.50%] [Generator Loss: 1.476544]
    


![png](output_15_93.png)


    4699 [Discriminator Loss: 0.447420, Acc.: 40.62%] [Generator Loss: 1.756811]
    


![png](output_15_95.png)


    4799 [Discriminator Loss: 0.395526, Acc.: 46.88%] [Generator Loss: 1.917628]
    


![png](output_15_97.png)


    4899 [Discriminator Loss: 0.357341, Acc.: 50.00%] [Generator Loss: 1.990626]
    


![png](output_15_99.png)


    4999 [Discriminator Loss: 0.348543, Acc.: 50.00%] [Generator Loss: 1.961836]
    


![png](output_15_101.png)


    5099 [Discriminator Loss: 0.405086, Acc.: 50.00%] [Generator Loss: 1.842877]
    


![png](output_15_103.png)


    5199 [Discriminator Loss: 0.379390, Acc.: 50.00%] [Generator Loss: 2.004472]
    


![png](output_15_105.png)


    5299 [Discriminator Loss: 0.374923, Acc.: 46.88%] [Generator Loss: 1.947821]
    


![png](output_15_107.png)


    5399 [Discriminator Loss: 0.357242, Acc.: 50.00%] [Generator Loss: 2.394717]
    


![png](output_15_109.png)


    5499 [Discriminator Loss: 0.391782, Acc.: 46.88%] [Generator Loss: 1.739019]
    


![png](output_15_111.png)


    5599 [Discriminator Loss: 0.348416, Acc.: 46.88%] [Generator Loss: 2.232237]
    


![png](output_15_113.png)


    5699 [Discriminator Loss: 0.336837, Acc.: 50.00%] [Generator Loss: 2.251399]
    


![png](output_15_115.png)


    5799 [Discriminator Loss: 0.361706, Acc.: 46.88%] [Generator Loss: 1.919540]
    


![png](output_15_117.png)


    5899 [Discriminator Loss: 0.394711, Acc.: 46.88%] [Generator Loss: 2.037126]
    


![png](output_15_119.png)


    5999 [Discriminator Loss: 0.340278, Acc.: 50.00%] [Generator Loss: 2.079579]
    


![png](output_15_121.png)


    6099 [Discriminator Loss: 0.408765, Acc.: 40.62%] [Generator Loss: 2.261632]
    


![png](output_15_123.png)


    6199 [Discriminator Loss: 0.323921, Acc.: 50.00%] [Generator Loss: 2.267210]
    


![png](output_15_125.png)


    6299 [Discriminator Loss: 0.459773, Acc.: 46.88%] [Generator Loss: 2.115924]
    


![png](output_15_127.png)


    6399 [Discriminator Loss: 0.403525, Acc.: 46.88%] [Generator Loss: 1.954991]
    


![png](output_15_129.png)


    6499 [Discriminator Loss: 0.390487, Acc.: 43.75%] [Generator Loss: 1.800906]
    


![png](output_15_131.png)


    6599 [Discriminator Loss: 0.365731, Acc.: 46.88%] [Generator Loss: 2.096528]
    


![png](output_15_133.png)


    6699 [Discriminator Loss: 0.318757, Acc.: 50.00%] [Generator Loss: 2.429365]
    


![png](output_15_135.png)


    6799 [Discriminator Loss: 0.452205, Acc.: 43.75%] [Generator Loss: 1.970767]
    


![png](output_15_137.png)


    6899 [Discriminator Loss: 0.298152, Acc.: 50.00%] [Generator Loss: 2.470682]
    


![png](output_15_139.png)


    6999 [Discriminator Loss: 0.347450, Acc.: 46.88%] [Generator Loss: 2.313902]
    


![png](output_15_141.png)


    7099 [Discriminator Loss: 0.355673, Acc.: 43.75%] [Generator Loss: 2.624203]
    


![png](output_15_143.png)


    7199 [Discriminator Loss: 0.330432, Acc.: 50.00%] [Generator Loss: 2.426723]
    


![png](output_15_145.png)


    7299 [Discriminator Loss: 0.331085, Acc.: 50.00%] [Generator Loss: 2.693770]
    


![png](output_15_147.png)


    7399 [Discriminator Loss: 0.332225, Acc.: 50.00%] [Generator Loss: 2.154036]
    


![png](output_15_149.png)


    7499 [Discriminator Loss: 0.317625, Acc.: 50.00%] [Generator Loss: 2.090562]
    


![png](output_15_151.png)


    7599 [Discriminator Loss: 0.333228, Acc.: 46.88%] [Generator Loss: 2.705907]
    


![png](output_15_153.png)


    7699 [Discriminator Loss: 0.259615, Acc.: 50.00%] [Generator Loss: 2.945856]
    


![png](output_15_155.png)


    7799 [Discriminator Loss: 0.294878, Acc.: 50.00%] [Generator Loss: 2.705618]
    


![png](output_15_157.png)


    7899 [Discriminator Loss: 0.290198, Acc.: 50.00%] [Generator Loss: 2.631955]
    


![png](output_15_159.png)


    7999 [Discriminator Loss: 0.266732, Acc.: 50.00%] [Generator Loss: 2.431810]
    


![png](output_15_161.png)


    8099 [Discriminator Loss: 0.283047, Acc.: 50.00%] [Generator Loss: 2.866724]
    


![png](output_15_163.png)


    8199 [Discriminator Loss: 0.313942, Acc.: 50.00%] [Generator Loss: 2.723072]
    


![png](output_15_165.png)


    8299 [Discriminator Loss: 0.270323, Acc.: 50.00%] [Generator Loss: 2.692429]
    


![png](output_15_167.png)


    8399 [Discriminator Loss: 0.263925, Acc.: 46.88%] [Generator Loss: 2.516573]
    


![png](output_15_169.png)


    8499 [Discriminator Loss: 0.352258, Acc.: 50.00%] [Generator Loss: 2.288093]
    


![png](output_15_171.png)


    8599 [Discriminator Loss: 0.298732, Acc.: 50.00%] [Generator Loss: 2.446842]
    


![png](output_15_173.png)


    8699 [Discriminator Loss: 0.248418, Acc.: 50.00%] [Generator Loss: 2.683459]
    


![png](output_15_175.png)


    8799 [Discriminator Loss: 0.261186, Acc.: 50.00%] [Generator Loss: 2.766473]
    


![png](output_15_177.png)


    8899 [Discriminator Loss: 0.303016, Acc.: 50.00%] [Generator Loss: 2.563261]
    


![png](output_15_179.png)


    8999 [Discriminator Loss: 0.311673, Acc.: 40.62%] [Generator Loss: 2.297503]
    


![png](output_15_181.png)


    9099 [Discriminator Loss: 0.251022, Acc.: 50.00%] [Generator Loss: 3.173155]
    


![png](output_15_183.png)


    9199 [Discriminator Loss: 0.263408, Acc.: 50.00%] [Generator Loss: 2.993392]
    


![png](output_15_185.png)


    9299 [Discriminator Loss: 0.286770, Acc.: 50.00%] [Generator Loss: 2.506930]
    


![png](output_15_187.png)


    9399 [Discriminator Loss: 0.266916, Acc.: 50.00%] [Generator Loss: 3.013502]
    


![png](output_15_189.png)


    9499 [Discriminator Loss: 0.334784, Acc.: 46.88%] [Generator Loss: 2.794177]
    


![png](output_15_191.png)


    9599 [Discriminator Loss: 0.262811, Acc.: 50.00%] [Generator Loss: 2.944830]
    


![png](output_15_193.png)


    9699 [Discriminator Loss: 0.240544, Acc.: 50.00%] [Generator Loss: 2.984646]
    


![png](output_15_195.png)


    9799 [Discriminator Loss: 0.329556, Acc.: 43.75%] [Generator Loss: 2.870439]
    


![png](output_15_197.png)


    9899 [Discriminator Loss: 0.274558, Acc.: 50.00%] [Generator Loss: 3.165918]
    


![png](output_15_199.png)


    9999 [Discriminator Loss: 0.230991, Acc.: 50.00%] [Generator Loss: 3.209969]
    


![png](output_15_201.png)


    10099 [Discriminator Loss: 0.270814, Acc.: 50.00%] [Generator Loss: 3.397029]
    


![png](output_15_203.png)


    10199 [Discriminator Loss: 0.247415, Acc.: 50.00%] [Generator Loss: 3.015153]
    


![png](output_15_205.png)


    10299 [Discriminator Loss: 0.221817, Acc.: 50.00%] [Generator Loss: 3.049962]
    


![png](output_15_207.png)


    10399 [Discriminator Loss: 0.267535, Acc.: 50.00%] [Generator Loss: 2.907621]
    


![png](output_15_209.png)


    10499 [Discriminator Loss: 0.254819, Acc.: 50.00%] [Generator Loss: 2.558258]
    


![png](output_15_211.png)


    10599 [Discriminator Loss: 0.246867, Acc.: 50.00%] [Generator Loss: 2.674755]
    


![png](output_15_213.png)


    10699 [Discriminator Loss: 0.254310, Acc.: 50.00%] [Generator Loss: 3.038120]
    


![png](output_15_215.png)


    10799 [Discriminator Loss: 0.249738, Acc.: 50.00%] [Generator Loss: 2.982536]
    


![png](output_15_217.png)


    10899 [Discriminator Loss: 0.231630, Acc.: 50.00%] [Generator Loss: 3.180937]
    


![png](output_15_219.png)


    10999 [Discriminator Loss: 0.238097, Acc.: 50.00%] [Generator Loss: 3.132089]
    


![png](output_15_221.png)


    11099 [Discriminator Loss: 0.247758, Acc.: 50.00%] [Generator Loss: 2.938409]
    


![png](output_15_223.png)


    11199 [Discriminator Loss: 0.256509, Acc.: 50.00%] [Generator Loss: 2.697111]
    


![png](output_15_225.png)


    11299 [Discriminator Loss: 0.214529, Acc.: 50.00%] [Generator Loss: 3.037398]
    


![png](output_15_227.png)


    11399 [Discriminator Loss: 0.236586, Acc.: 50.00%] [Generator Loss: 3.434495]
    


![png](output_15_229.png)


    11499 [Discriminator Loss: 0.249950, Acc.: 50.00%] [Generator Loss: 3.388303]
    


![png](output_15_231.png)


    11599 [Discriminator Loss: 0.221032, Acc.: 50.00%] [Generator Loss: 3.645879]
    


![png](output_15_233.png)


    11699 [Discriminator Loss: 0.253908, Acc.: 50.00%] [Generator Loss: 3.578416]
    


![png](output_15_235.png)


    11799 [Discriminator Loss: 0.215749, Acc.: 50.00%] [Generator Loss: 3.640763]
    


![png](output_15_237.png)


    11899 [Discriminator Loss: 0.224788, Acc.: 50.00%] [Generator Loss: 3.184854]
    


![png](output_15_239.png)


    11999 [Discriminator Loss: 0.221552, Acc.: 50.00%] [Generator Loss: 3.508756]
    


![png](output_15_241.png)


    12099 [Discriminator Loss: 0.273993, Acc.: 50.00%] [Generator Loss: 3.882975]
    


![png](output_15_243.png)


    12199 [Discriminator Loss: 0.240120, Acc.: 50.00%] [Generator Loss: 3.451721]
    


![png](output_15_245.png)


    12299 [Discriminator Loss: 0.219847, Acc.: 50.00%] [Generator Loss: 3.260644]
    


![png](output_15_247.png)


    12399 [Discriminator Loss: 0.245278, Acc.: 50.00%] [Generator Loss: 3.268765]
    


![png](output_15_249.png)


    12499 [Discriminator Loss: 0.275118, Acc.: 50.00%] [Generator Loss: 3.208629]
    


![png](output_15_251.png)


    12599 [Discriminator Loss: 0.208387, Acc.: 50.00%] [Generator Loss: 3.915851]
    


![png](output_15_253.png)


    12699 [Discriminator Loss: 0.195424, Acc.: 50.00%] [Generator Loss: 3.894679]
    


![png](output_15_255.png)


    12799 [Discriminator Loss: 0.216761, Acc.: 50.00%] [Generator Loss: 4.101087]
    


![png](output_15_257.png)


    12899 [Discriminator Loss: 0.233190, Acc.: 50.00%] [Generator Loss: 3.640838]
    


![png](output_15_259.png)


    12999 [Discriminator Loss: 0.207794, Acc.: 50.00%] [Generator Loss: 3.577243]
    


![png](output_15_261.png)


    13099 [Discriminator Loss: 0.238862, Acc.: 50.00%] [Generator Loss: 3.439309]
    


![png](output_15_263.png)


    13199 [Discriminator Loss: 0.268647, Acc.: 50.00%] [Generator Loss: 3.356547]
    


![png](output_15_265.png)


    13299 [Discriminator Loss: 0.213892, Acc.: 50.00%] [Generator Loss: 3.303556]
    


![png](output_15_267.png)


    13399 [Discriminator Loss: 0.197844, Acc.: 50.00%] [Generator Loss: 3.490476]
    


![png](output_15_269.png)


    13499 [Discriminator Loss: 0.220977, Acc.: 50.00%] [Generator Loss: 3.572332]
    


![png](output_15_271.png)


    13599 [Discriminator Loss: 0.236116, Acc.: 50.00%] [Generator Loss: 4.160029]
    


![png](output_15_273.png)


    13699 [Discriminator Loss: 0.216008, Acc.: 50.00%] [Generator Loss: 3.377432]
    


![png](output_15_275.png)


    13799 [Discriminator Loss: 0.222323, Acc.: 50.00%] [Generator Loss: 3.500231]
    


![png](output_15_277.png)


    13899 [Discriminator Loss: 0.216304, Acc.: 50.00%] [Generator Loss: 3.506346]
    


![png](output_15_279.png)


    13999 [Discriminator Loss: 0.218669, Acc.: 50.00%] [Generator Loss: 2.950146]
    


![png](output_15_281.png)


    14099 [Discriminator Loss: 0.224351, Acc.: 50.00%] [Generator Loss: 3.751628]
    


![png](output_15_283.png)


    14199 [Discriminator Loss: 0.224167, Acc.: 50.00%] [Generator Loss: 4.332609]
    


![png](output_15_285.png)


    14299 [Discriminator Loss: 0.264282, Acc.: 50.00%] [Generator Loss: 2.857003]
    


![png](output_15_287.png)


    14399 [Discriminator Loss: 0.241750, Acc.: 50.00%] [Generator Loss: 3.553103]
    


![png](output_15_289.png)


    14499 [Discriminator Loss: 0.210200, Acc.: 50.00%] [Generator Loss: 3.780151]
    


![png](output_15_291.png)


    14599 [Discriminator Loss: 0.209401, Acc.: 50.00%] [Generator Loss: 3.820736]
    


![png](output_15_293.png)


    14699 [Discriminator Loss: 0.231932, Acc.: 50.00%] [Generator Loss: 3.340911]
    


![png](output_15_295.png)


    14799 [Discriminator Loss: 0.206411, Acc.: 50.00%] [Generator Loss: 3.863117]
    


![png](output_15_297.png)


    14899 [Discriminator Loss: 0.209673, Acc.: 50.00%] [Generator Loss: 3.901859]
    


![png](output_15_299.png)


    14999 [Discriminator Loss: 0.244808, Acc.: 50.00%] [Generator Loss: 3.133154]
    


![png](output_15_301.png)


    15099 [Discriminator Loss: 0.231370, Acc.: 50.00%] [Generator Loss: 3.288863]
    


![png](output_15_303.png)


    15199 [Discriminator Loss: 0.199042, Acc.: 50.00%] [Generator Loss: 3.826611]
    


![png](output_15_305.png)


    15299 [Discriminator Loss: 0.226835, Acc.: 50.00%] [Generator Loss: 4.070142]
    


![png](output_15_307.png)


    15399 [Discriminator Loss: 0.226069, Acc.: 50.00%] [Generator Loss: 2.911292]
    


![png](output_15_309.png)


    15499 [Discriminator Loss: 0.214761, Acc.: 50.00%] [Generator Loss: 4.187685]
    


![png](output_15_311.png)


    15599 [Discriminator Loss: 0.202474, Acc.: 50.00%] [Generator Loss: 3.508649]
    


![png](output_15_313.png)


    15699 [Discriminator Loss: 0.228289, Acc.: 50.00%] [Generator Loss: 3.569915]
    


![png](output_15_315.png)


    15799 [Discriminator Loss: 0.216797, Acc.: 50.00%] [Generator Loss: 3.572399]
    


![png](output_15_317.png)


    15899 [Discriminator Loss: 0.198746, Acc.: 50.00%] [Generator Loss: 4.286937]
    


![png](output_15_319.png)


    15999 [Discriminator Loss: 0.242967, Acc.: 50.00%] [Generator Loss: 3.612521]
    


![png](output_15_321.png)


    16099 [Discriminator Loss: 0.206245, Acc.: 50.00%] [Generator Loss: 3.902646]
    


![png](output_15_323.png)


    16199 [Discriminator Loss: 0.180580, Acc.: 50.00%] [Generator Loss: 4.548311]
    


![png](output_15_325.png)


    16299 [Discriminator Loss: 0.235338, Acc.: 50.00%] [Generator Loss: 3.413727]
    


![png](output_15_327.png)


    16399 [Discriminator Loss: 0.292041, Acc.: 50.00%] [Generator Loss: 2.866558]
    


![png](output_15_329.png)


    16499 [Discriminator Loss: 0.220754, Acc.: 50.00%] [Generator Loss: 4.401505]
    


![png](output_15_331.png)


    16599 [Discriminator Loss: 0.231438, Acc.: 50.00%] [Generator Loss: 3.863136]
    


![png](output_15_333.png)


    16699 [Discriminator Loss: 0.187335, Acc.: 50.00%] [Generator Loss: 4.329034]
    


![png](output_15_335.png)


    16799 [Discriminator Loss: 0.201010, Acc.: 50.00%] [Generator Loss: 3.346109]
    


![png](output_15_337.png)


    16899 [Discriminator Loss: 0.215801, Acc.: 50.00%] [Generator Loss: 3.488194]
    


![png](output_15_339.png)


    16999 [Discriminator Loss: 0.206622, Acc.: 50.00%] [Generator Loss: 3.867380]
    


![png](output_15_341.png)


    17099 [Discriminator Loss: 0.270315, Acc.: 46.88%] [Generator Loss: 3.463565]
    


![png](output_15_343.png)


    17199 [Discriminator Loss: 0.286918, Acc.: 50.00%] [Generator Loss: 3.118623]
    


![png](output_15_345.png)


    17299 [Discriminator Loss: 0.211575, Acc.: 50.00%] [Generator Loss: 3.692233]
    


![png](output_15_347.png)


    17399 [Discriminator Loss: 0.194165, Acc.: 50.00%] [Generator Loss: 4.144120]
    


![png](output_15_349.png)


    17499 [Discriminator Loss: 0.203366, Acc.: 50.00%] [Generator Loss: 4.427969]
    


![png](output_15_351.png)


    17599 [Discriminator Loss: 0.199592, Acc.: 50.00%] [Generator Loss: 3.723123]
    


![png](output_15_353.png)


    17699 [Discriminator Loss: 0.185506, Acc.: 50.00%] [Generator Loss: 4.647007]
    


![png](output_15_355.png)


    17799 [Discriminator Loss: 0.206298, Acc.: 50.00%] [Generator Loss: 4.435544]
    


![png](output_15_357.png)


    17899 [Discriminator Loss: 0.185807, Acc.: 50.00%] [Generator Loss: 4.616975]
    


![png](output_15_359.png)


    17999 [Discriminator Loss: 0.226260, Acc.: 50.00%] [Generator Loss: 3.611944]
    


![png](output_15_361.png)


    18099 [Discriminator Loss: 0.192843, Acc.: 50.00%] [Generator Loss: 4.490035]
    


![png](output_15_363.png)


    18199 [Discriminator Loss: 0.207687, Acc.: 50.00%] [Generator Loss: 4.160069]
    


![png](output_15_365.png)


    18299 [Discriminator Loss: 0.192843, Acc.: 50.00%] [Generator Loss: 4.267022]
    


![png](output_15_367.png)


    18399 [Discriminator Loss: 0.204442, Acc.: 50.00%] [Generator Loss: 3.245438]
    


![png](output_15_369.png)


    18499 [Discriminator Loss: 0.245554, Acc.: 50.00%] [Generator Loss: 4.157769]
    


![png](output_15_371.png)


    18599 [Discriminator Loss: 0.198838, Acc.: 50.00%] [Generator Loss: 3.599926]
    


![png](output_15_373.png)


    18699 [Discriminator Loss: 0.197155, Acc.: 50.00%] [Generator Loss: 4.485633]
    


![png](output_15_375.png)


    18799 [Discriminator Loss: 0.222223, Acc.: 50.00%] [Generator Loss: 4.615891]
    


![png](output_15_377.png)


    18899 [Discriminator Loss: 0.199631, Acc.: 50.00%] [Generator Loss: 3.855389]
    


![png](output_15_379.png)


    18999 [Discriminator Loss: 0.199488, Acc.: 50.00%] [Generator Loss: 4.179087]
    


![png](output_15_381.png)


    19099 [Discriminator Loss: 0.196709, Acc.: 50.00%] [Generator Loss: 5.134875]
    


![png](output_15_383.png)


    19199 [Discriminator Loss: 0.213260, Acc.: 50.00%] [Generator Loss: 4.183051]
    


![png](output_15_385.png)


    19299 [Discriminator Loss: 0.275807, Acc.: 50.00%] [Generator Loss: 4.370052]
    


![png](output_15_387.png)


    19399 [Discriminator Loss: 0.190310, Acc.: 50.00%] [Generator Loss: 4.434796]
    


![png](output_15_389.png)


    19499 [Discriminator Loss: 0.203334, Acc.: 50.00%] [Generator Loss: 3.944854]
    


![png](output_15_391.png)


    19599 [Discriminator Loss: 0.218442, Acc.: 50.00%] [Generator Loss: 4.823578]
    


![png](output_15_393.png)


    19699 [Discriminator Loss: 0.256380, Acc.: 50.00%] [Generator Loss: 4.390144]
    


![png](output_15_395.png)


    19799 [Discriminator Loss: 0.177840, Acc.: 50.00%] [Generator Loss: 4.659359]
    


![png](output_15_397.png)


    19899 [Discriminator Loss: 0.184137, Acc.: 50.00%] [Generator Loss: 4.996747]
    


![png](output_15_399.png)


    19999 [Discriminator Loss: 0.202941, Acc.: 50.00%] [Generator Loss: 4.638603]
    


![png](output_15_401.png)


    20099 [Discriminator Loss: 0.189119, Acc.: 50.00%] [Generator Loss: 4.726477]
    


![png](output_15_403.png)


    20199 [Discriminator Loss: 0.211787, Acc.: 50.00%] [Generator Loss: 4.203971]
    


![png](output_15_405.png)


    20299 [Discriminator Loss: 0.225055, Acc.: 46.88%] [Generator Loss: 4.764792]
    


![png](output_15_407.png)


    20399 [Discriminator Loss: 0.220626, Acc.: 50.00%] [Generator Loss: 3.849586]
    


![png](output_15_409.png)


    20499 [Discriminator Loss: 0.201288, Acc.: 50.00%] [Generator Loss: 4.659321]
    


![png](output_15_411.png)


    20599 [Discriminator Loss: 0.198873, Acc.: 50.00%] [Generator Loss: 4.843485]
    


![png](output_15_413.png)


    20699 [Discriminator Loss: 0.233455, Acc.: 50.00%] [Generator Loss: 3.674802]
    


![png](output_15_415.png)


    20799 [Discriminator Loss: 0.177701, Acc.: 50.00%] [Generator Loss: 5.522161]
    


![png](output_15_417.png)


    20899 [Discriminator Loss: 0.193361, Acc.: 50.00%] [Generator Loss: 4.830045]
    


![png](output_15_419.png)


    20999 [Discriminator Loss: 0.203715, Acc.: 50.00%] [Generator Loss: 4.772823]
    


![png](output_15_421.png)


    21099 [Discriminator Loss: 0.191167, Acc.: 50.00%] [Generator Loss: 5.752180]
    


![png](output_15_423.png)


    21199 [Discriminator Loss: 0.197812, Acc.: 50.00%] [Generator Loss: 4.150517]
    


![png](output_15_425.png)


    21299 [Discriminator Loss: 0.188951, Acc.: 50.00%] [Generator Loss: 4.375514]
    


![png](output_15_427.png)


    21399 [Discriminator Loss: 0.246873, Acc.: 46.88%] [Generator Loss: 5.401443]
    


![png](output_15_429.png)


    21499 [Discriminator Loss: 0.213696, Acc.: 50.00%] [Generator Loss: 4.633869]
    


![png](output_15_431.png)


    21599 [Discriminator Loss: 0.201305, Acc.: 50.00%] [Generator Loss: 4.289634]
    


![png](output_15_433.png)


    21699 [Discriminator Loss: 0.215610, Acc.: 50.00%] [Generator Loss: 4.645699]
    


![png](output_15_435.png)


    21799 [Discriminator Loss: 0.190359, Acc.: 50.00%] [Generator Loss: 4.535028]
    


![png](output_15_437.png)


    21899 [Discriminator Loss: 0.232339, Acc.: 50.00%] [Generator Loss: 4.778258]
    


![png](output_15_439.png)


    21999 [Discriminator Loss: 0.185421, Acc.: 50.00%] [Generator Loss: 5.345923]
    


![png](output_15_441.png)


    22099 [Discriminator Loss: 0.191622, Acc.: 50.00%] [Generator Loss: 5.098998]
    


![png](output_15_443.png)


    22199 [Discriminator Loss: 0.201439, Acc.: 50.00%] [Generator Loss: 4.778505]
    


![png](output_15_445.png)


    22299 [Discriminator Loss: 0.181242, Acc.: 50.00%] [Generator Loss: 4.593478]
    


![png](output_15_447.png)


    22399 [Discriminator Loss: 0.209659, Acc.: 50.00%] [Generator Loss: 4.705475]
    


![png](output_15_449.png)


    22499 [Discriminator Loss: 0.198298, Acc.: 50.00%] [Generator Loss: 5.131633]
    


![png](output_15_451.png)


    22599 [Discriminator Loss: 0.200207, Acc.: 50.00%] [Generator Loss: 4.750843]
    


![png](output_15_453.png)


    22699 [Discriminator Loss: 0.213704, Acc.: 50.00%] [Generator Loss: 4.453341]
    


![png](output_15_455.png)


    22799 [Discriminator Loss: 0.232338, Acc.: 50.00%] [Generator Loss: 4.298847]
    


![png](output_15_457.png)


    22899 [Discriminator Loss: 0.198115, Acc.: 50.00%] [Generator Loss: 4.578321]
    


![png](output_15_459.png)


    22999 [Discriminator Loss: 0.194718, Acc.: 50.00%] [Generator Loss: 4.222215]
    


![png](output_15_461.png)


    23099 [Discriminator Loss: 0.194774, Acc.: 50.00%] [Generator Loss: 5.660826]
    


![png](output_15_463.png)


    23199 [Discriminator Loss: 0.208415, Acc.: 50.00%] [Generator Loss: 5.179228]
    


![png](output_15_465.png)


    23299 [Discriminator Loss: 0.206255, Acc.: 50.00%] [Generator Loss: 5.425950]
    


![png](output_15_467.png)


    23399 [Discriminator Loss: 0.216427, Acc.: 50.00%] [Generator Loss: 4.779734]
    


![png](output_15_469.png)


    23499 [Discriminator Loss: 0.211743, Acc.: 50.00%] [Generator Loss: 4.798205]
    


![png](output_15_471.png)


    23599 [Discriminator Loss: 0.180542, Acc.: 50.00%] [Generator Loss: 5.452457]
    


![png](output_15_473.png)


    23699 [Discriminator Loss: 0.228408, Acc.: 50.00%] [Generator Loss: 4.792343]
    


![png](output_15_475.png)


    23799 [Discriminator Loss: 0.179203, Acc.: 50.00%] [Generator Loss: 4.947266]
    


![png](output_15_477.png)


    23899 [Discriminator Loss: 0.201749, Acc.: 50.00%] [Generator Loss: 4.708875]
    


![png](output_15_479.png)


    23999 [Discriminator Loss: 0.199291, Acc.: 50.00%] [Generator Loss: 5.109673]
    


![png](output_15_481.png)


    24099 [Discriminator Loss: 0.190386, Acc.: 50.00%] [Generator Loss: 5.272611]
    


![png](output_15_483.png)


    24199 [Discriminator Loss: 0.180752, Acc.: 50.00%] [Generator Loss: 4.765657]
    


![png](output_15_485.png)


    24299 [Discriminator Loss: 0.243882, Acc.: 50.00%] [Generator Loss: 6.156512]
    


![png](output_15_487.png)


    24399 [Discriminator Loss: 0.272706, Acc.: 50.00%] [Generator Loss: 4.510758]
    


![png](output_15_489.png)


    24499 [Discriminator Loss: 0.191528, Acc.: 50.00%] [Generator Loss: 3.890719]
    


![png](output_15_491.png)


    24599 [Discriminator Loss: 0.197887, Acc.: 50.00%] [Generator Loss: 4.072302]
    


![png](output_15_493.png)


    24699 [Discriminator Loss: 0.194958, Acc.: 50.00%] [Generator Loss: 4.445964]
    


![png](output_15_495.png)


    24799 [Discriminator Loss: 0.232779, Acc.: 50.00%] [Generator Loss: 5.405078]
    


![png](output_15_497.png)


    24899 [Discriminator Loss: 0.227079, Acc.: 50.00%] [Generator Loss: 4.927087]
    


![png](output_15_499.png)


    24999 [Discriminator Loss: 0.227547, Acc.: 50.00%] [Generator Loss: 5.198724]
    


![png](output_15_501.png)


    25099 [Discriminator Loss: 0.213896, Acc.: 50.00%] [Generator Loss: 5.226925]
    


![png](output_15_503.png)


    25199 [Discriminator Loss: 0.192550, Acc.: 50.00%] [Generator Loss: 4.716604]
    


![png](output_15_505.png)


    25299 [Discriminator Loss: 0.197439, Acc.: 50.00%] [Generator Loss: 5.542611]
    


![png](output_15_507.png)


    25399 [Discriminator Loss: 0.200147, Acc.: 50.00%] [Generator Loss: 4.853840]
    


![png](output_15_509.png)


    25499 [Discriminator Loss: 0.196321, Acc.: 50.00%] [Generator Loss: 4.562839]
    


![png](output_15_511.png)


    25599 [Discriminator Loss: 0.207211, Acc.: 50.00%] [Generator Loss: 4.528983]
    


![png](output_15_513.png)


    25699 [Discriminator Loss: 0.266717, Acc.: 50.00%] [Generator Loss: 3.624197]
    


![png](output_15_515.png)


    25799 [Discriminator Loss: 0.244886, Acc.: 50.00%] [Generator Loss: 4.616258]
    


![png](output_15_517.png)


    25899 [Discriminator Loss: 0.198208, Acc.: 50.00%] [Generator Loss: 4.439633]
    


![png](output_15_519.png)


    25999 [Discriminator Loss: 0.245468, Acc.: 50.00%] [Generator Loss: 4.658411]
    


![png](output_15_521.png)


    26099 [Discriminator Loss: 0.196523, Acc.: 50.00%] [Generator Loss: 4.459651]
    


![png](output_15_523.png)


    26199 [Discriminator Loss: 0.212240, Acc.: 46.88%] [Generator Loss: 4.864231]
    


![png](output_15_525.png)


    26299 [Discriminator Loss: 0.218585, Acc.: 50.00%] [Generator Loss: 5.239383]
    


![png](output_15_527.png)


    26399 [Discriminator Loss: 0.245140, Acc.: 50.00%] [Generator Loss: 4.893253]
    


![png](output_15_529.png)


    26499 [Discriminator Loss: 0.227162, Acc.: 50.00%] [Generator Loss: 4.716321]
    


![png](output_15_531.png)


    26599 [Discriminator Loss: 0.288852, Acc.: 50.00%] [Generator Loss: 3.378398]
    


![png](output_15_533.png)


    26699 [Discriminator Loss: 0.227221, Acc.: 50.00%] [Generator Loss: 4.703102]
    


![png](output_15_535.png)


    26799 [Discriminator Loss: 0.206523, Acc.: 50.00%] [Generator Loss: 3.986953]
    


![png](output_15_537.png)


    26899 [Discriminator Loss: 0.253822, Acc.: 50.00%] [Generator Loss: 4.563589]
    


![png](output_15_539.png)


    26999 [Discriminator Loss: 0.174230, Acc.: 50.00%] [Generator Loss: 6.440557]
    


![png](output_15_541.png)


    27099 [Discriminator Loss: 0.246005, Acc.: 50.00%] [Generator Loss: 4.068841]
    


![png](output_15_543.png)


    27199 [Discriminator Loss: 0.201281, Acc.: 50.00%] [Generator Loss: 4.284819]
    


![png](output_15_545.png)


    27299 [Discriminator Loss: 0.428430, Acc.: 50.00%] [Generator Loss: 3.714534]
    


![png](output_15_547.png)


    27399 [Discriminator Loss: 0.303394, Acc.: 50.00%] [Generator Loss: 3.937818]
    


![png](output_15_549.png)


    27499 [Discriminator Loss: 0.203092, Acc.: 50.00%] [Generator Loss: 4.367931]
    


![png](output_15_551.png)


    27599 [Discriminator Loss: 0.249478, Acc.: 50.00%] [Generator Loss: 3.030441]
    


![png](output_15_553.png)


    27699 [Discriminator Loss: 0.211508, Acc.: 50.00%] [Generator Loss: 3.936856]
    


![png](output_15_555.png)


    27799 [Discriminator Loss: 0.225003, Acc.: 50.00%] [Generator Loss: 3.744143]
    


![png](output_15_557.png)


    27899 [Discriminator Loss: 0.205079, Acc.: 50.00%] [Generator Loss: 4.494799]
    


![png](output_15_559.png)


    27999 [Discriminator Loss: 0.201561, Acc.: 50.00%] [Generator Loss: 4.659624]
    


![png](output_15_561.png)


    28099 [Discriminator Loss: 0.224247, Acc.: 50.00%] [Generator Loss: 3.417658]
    


![png](output_15_563.png)


    28199 [Discriminator Loss: 0.187141, Acc.: 50.00%] [Generator Loss: 5.016285]
    


![png](output_15_565.png)


    28299 [Discriminator Loss: 0.233293, Acc.: 50.00%] [Generator Loss: 3.602603]
    


![png](output_15_567.png)


    28399 [Discriminator Loss: 0.195834, Acc.: 50.00%] [Generator Loss: 4.704080]
    


![png](output_15_569.png)


    28499 [Discriminator Loss: 0.190982, Acc.: 50.00%] [Generator Loss: 4.694103]
    


![png](output_15_571.png)


    28599 [Discriminator Loss: 0.205614, Acc.: 50.00%] [Generator Loss: 4.792244]
    


![png](output_15_573.png)


    28699 [Discriminator Loss: 0.189974, Acc.: 50.00%] [Generator Loss: 4.458751]
    


![png](output_15_575.png)


    28799 [Discriminator Loss: 0.228381, Acc.: 50.00%] [Generator Loss: 4.094924]
    


![png](output_15_577.png)


    28899 [Discriminator Loss: 0.194014, Acc.: 50.00%] [Generator Loss: 4.165119]
    


![png](output_15_579.png)


    28999 [Discriminator Loss: 0.198695, Acc.: 50.00%] [Generator Loss: 4.423394]
    


![png](output_15_581.png)


    29099 [Discriminator Loss: 0.191437, Acc.: 50.00%] [Generator Loss: 4.928921]
    


![png](output_15_583.png)


    29199 [Discriminator Loss: 0.227450, Acc.: 50.00%] [Generator Loss: 4.077593]
    


![png](output_15_585.png)


    29299 [Discriminator Loss: 0.254490, Acc.: 50.00%] [Generator Loss: 4.073380]
    


![png](output_15_587.png)


    29399 [Discriminator Loss: 0.232988, Acc.: 50.00%] [Generator Loss: 3.254632]
    


![png](output_15_589.png)


    29499 [Discriminator Loss: 0.197651, Acc.: 50.00%] [Generator Loss: 4.944232]
    


![png](output_15_591.png)


    29599 [Discriminator Loss: 0.192195, Acc.: 50.00%] [Generator Loss: 4.564324]
    


![png](output_15_593.png)


    29699 [Discriminator Loss: 0.250641, Acc.: 50.00%] [Generator Loss: 4.026864]
    


![png](output_15_595.png)


    29799 [Discriminator Loss: 0.187160, Acc.: 50.00%] [Generator Loss: 4.863910]
    


![png](output_15_597.png)


    29899 [Discriminator Loss: 0.207581, Acc.: 50.00%] [Generator Loss: 5.083276]
    


![png](output_15_599.png)


    29999 [Discriminator Loss: 0.179740, Acc.: 50.00%] [Generator Loss: 4.817950]
    


![png](output_15_601.png)


    30099 [Discriminator Loss: 0.206254, Acc.: 50.00%] [Generator Loss: 4.544405]
    


![png](output_15_603.png)


    30199 [Discriminator Loss: 0.193486, Acc.: 50.00%] [Generator Loss: 4.692171]
    


![png](output_15_605.png)


    30299 [Discriminator Loss: 0.290577, Acc.: 50.00%] [Generator Loss: 3.415726]
    


![png](output_15_607.png)


    30399 [Discriminator Loss: 0.190372, Acc.: 50.00%] [Generator Loss: 4.674561]
    


![png](output_15_609.png)


    30499 [Discriminator Loss: 0.186977, Acc.: 50.00%] [Generator Loss: 5.228439]
    


![png](output_15_611.png)


    30599 [Discriminator Loss: 0.187123, Acc.: 50.00%] [Generator Loss: 5.297214]
    


![png](output_15_613.png)


    30699 [Discriminator Loss: 0.180491, Acc.: 50.00%] [Generator Loss: 5.399487]
    


![png](output_15_615.png)


    30799 [Discriminator Loss: 0.226262, Acc.: 50.00%] [Generator Loss: 4.262157]
    


![png](output_15_617.png)


    30899 [Discriminator Loss: 0.184731, Acc.: 50.00%] [Generator Loss: 4.959188]
    


![png](output_15_619.png)


    30999 [Discriminator Loss: 0.187379, Acc.: 50.00%] [Generator Loss: 5.014345]
    


![png](output_15_621.png)


    31099 [Discriminator Loss: 0.248171, Acc.: 43.75%] [Generator Loss: 5.245766]
    


![png](output_15_623.png)


    31199 [Discriminator Loss: 0.238861, Acc.: 50.00%] [Generator Loss: 4.697570]
    


![png](output_15_625.png)


    31299 [Discriminator Loss: 0.204324, Acc.: 50.00%] [Generator Loss: 5.709866]
    


![png](output_15_627.png)


    31399 [Discriminator Loss: 0.227242, Acc.: 50.00%] [Generator Loss: 5.194775]
    


![png](output_15_629.png)


    31499 [Discriminator Loss: 0.185364, Acc.: 50.00%] [Generator Loss: 6.288341]
    


![png](output_15_631.png)


    31599 [Discriminator Loss: 0.193201, Acc.: 50.00%] [Generator Loss: 5.355061]
    


![png](output_15_633.png)


    31699 [Discriminator Loss: 0.221814, Acc.: 50.00%] [Generator Loss: 4.799547]
    


![png](output_15_635.png)


    31799 [Discriminator Loss: 0.204434, Acc.: 50.00%] [Generator Loss: 5.241780]
    


![png](output_15_637.png)


    31899 [Discriminator Loss: 0.268361, Acc.: 50.00%] [Generator Loss: 3.726466]
    


![png](output_15_639.png)


    31999 [Discriminator Loss: 0.201867, Acc.: 50.00%] [Generator Loss: 5.096879]
    


![png](output_15_641.png)


    32099 [Discriminator Loss: 0.320904, Acc.: 50.00%] [Generator Loss: 4.657047]
    


![png](output_15_643.png)


    32199 [Discriminator Loss: 0.179598, Acc.: 50.00%] [Generator Loss: 5.443740]
    


![png](output_15_645.png)


    32299 [Discriminator Loss: 0.235149, Acc.: 50.00%] [Generator Loss: 4.168787]
    


![png](output_15_647.png)


    32399 [Discriminator Loss: 0.197031, Acc.: 50.00%] [Generator Loss: 4.795702]
    


![png](output_15_649.png)


    32499 [Discriminator Loss: 0.219637, Acc.: 46.88%] [Generator Loss: 4.804122]
    


![png](output_15_651.png)


    32599 [Discriminator Loss: 0.232339, Acc.: 50.00%] [Generator Loss: 5.863583]
    


![png](output_15_653.png)


    32699 [Discriminator Loss: 0.227569, Acc.: 50.00%] [Generator Loss: 5.173666]
    


![png](output_15_655.png)


    32799 [Discriminator Loss: 0.209477, Acc.: 50.00%] [Generator Loss: 4.414763]
    


![png](output_15_657.png)


    32899 [Discriminator Loss: 0.223731, Acc.: 50.00%] [Generator Loss: 4.672632]
    


![png](output_15_659.png)


    32999 [Discriminator Loss: 0.212680, Acc.: 50.00%] [Generator Loss: 4.817073]
    


![png](output_15_661.png)


    33099 [Discriminator Loss: 0.211989, Acc.: 50.00%] [Generator Loss: 4.165356]
    


![png](output_15_663.png)


    33199 [Discriminator Loss: 0.203803, Acc.: 50.00%] [Generator Loss: 5.158484]
    


![png](output_15_665.png)


    33299 [Discriminator Loss: 0.219528, Acc.: 46.88%] [Generator Loss: 5.252950]
    


![png](output_15_667.png)


    33399 [Discriminator Loss: 0.217414, Acc.: 50.00%] [Generator Loss: 4.507004]
    


![png](output_15_669.png)


    33499 [Discriminator Loss: 0.191622, Acc.: 50.00%] [Generator Loss: 4.523240]
    


![png](output_15_671.png)


    33599 [Discriminator Loss: 0.237810, Acc.: 50.00%] [Generator Loss: 3.667682]
    


![png](output_15_673.png)


    33699 [Discriminator Loss: 0.229195, Acc.: 50.00%] [Generator Loss: 3.791856]
    


![png](output_15_675.png)


    33799 [Discriminator Loss: 0.235371, Acc.: 50.00%] [Generator Loss: 3.938595]
    


![png](output_15_677.png)


    33899 [Discriminator Loss: 0.203519, Acc.: 50.00%] [Generator Loss: 3.787020]
    


![png](output_15_679.png)


    33999 [Discriminator Loss: 0.192925, Acc.: 50.00%] [Generator Loss: 5.803423]
    


![png](output_15_681.png)


    34099 [Discriminator Loss: 0.199673, Acc.: 50.00%] [Generator Loss: 4.949805]
    


![png](output_15_683.png)


    34199 [Discriminator Loss: 0.185881, Acc.: 50.00%] [Generator Loss: 5.055093]
    


![png](output_15_685.png)


    34299 [Discriminator Loss: 0.194823, Acc.: 50.00%] [Generator Loss: 5.012200]
    


![png](output_15_687.png)


    34399 [Discriminator Loss: 0.221535, Acc.: 50.00%] [Generator Loss: 4.045897]
    


![png](output_15_689.png)


    34499 [Discriminator Loss: 0.258534, Acc.: 50.00%] [Generator Loss: 3.881058]
    


![png](output_15_691.png)


    34599 [Discriminator Loss: 0.219488, Acc.: 50.00%] [Generator Loss: 4.537618]
    


![png](output_15_693.png)


    34699 [Discriminator Loss: 0.208456, Acc.: 50.00%] [Generator Loss: 4.781159]
    


![png](output_15_695.png)


    34799 [Discriminator Loss: 0.196974, Acc.: 50.00%] [Generator Loss: 4.270106]
    


![png](output_15_697.png)


    34899 [Discriminator Loss: 0.200924, Acc.: 50.00%] [Generator Loss: 4.575688]
    


![png](output_15_699.png)


    34999 [Discriminator Loss: 0.209164, Acc.: 46.88%] [Generator Loss: 5.279465]
    


![png](output_15_701.png)


    35099 [Discriminator Loss: 0.192432, Acc.: 50.00%] [Generator Loss: 4.871521]
    


![png](output_15_703.png)


    35199 [Discriminator Loss: 0.189767, Acc.: 50.00%] [Generator Loss: 5.368334]
    


![png](output_15_705.png)


    35299 [Discriminator Loss: 0.190972, Acc.: 50.00%] [Generator Loss: 4.812129]
    


![png](output_15_707.png)


    35399 [Discriminator Loss: 0.200310, Acc.: 50.00%] [Generator Loss: 4.384276]
    


![png](output_15_709.png)


    35499 [Discriminator Loss: 0.217048, Acc.: 50.00%] [Generator Loss: 5.226769]
    


![png](output_15_711.png)


    35599 [Discriminator Loss: 0.208398, Acc.: 50.00%] [Generator Loss: 4.682844]
    


![png](output_15_713.png)


    35699 [Discriminator Loss: 0.196955, Acc.: 50.00%] [Generator Loss: 4.887307]
    


![png](output_15_715.png)


    35799 [Discriminator Loss: 0.197530, Acc.: 50.00%] [Generator Loss: 4.802269]
    


![png](output_15_717.png)


    35899 [Discriminator Loss: 0.206539, Acc.: 50.00%] [Generator Loss: 4.463434]
    


![png](output_15_719.png)


    35999 [Discriminator Loss: 0.208142, Acc.: 50.00%] [Generator Loss: 5.039951]
    


![png](output_15_721.png)


    36099 [Discriminator Loss: 0.207445, Acc.: 50.00%] [Generator Loss: 3.993265]
    


![png](output_15_723.png)


    36199 [Discriminator Loss: 0.203096, Acc.: 50.00%] [Generator Loss: 5.376816]
    


![png](output_15_725.png)


    36299 [Discriminator Loss: 0.223095, Acc.: 50.00%] [Generator Loss: 4.248487]
    


![png](output_15_727.png)


    36399 [Discriminator Loss: 0.211814, Acc.: 50.00%] [Generator Loss: 4.816875]
    


![png](output_15_729.png)


    36499 [Discriminator Loss: 0.263102, Acc.: 50.00%] [Generator Loss: 4.233924]
    


![png](output_15_731.png)


    36599 [Discriminator Loss: 0.216920, Acc.: 50.00%] [Generator Loss: 5.123439]
    


![png](output_15_733.png)


    36699 [Discriminator Loss: 0.203219, Acc.: 50.00%] [Generator Loss: 4.297037]
    


![png](output_15_735.png)


    36799 [Discriminator Loss: 0.181492, Acc.: 50.00%] [Generator Loss: 5.009568]
    


![png](output_15_737.png)


    36899 [Discriminator Loss: 0.208900, Acc.: 50.00%] [Generator Loss: 4.826907]
    


![png](output_15_739.png)


    36999 [Discriminator Loss: 0.207333, Acc.: 50.00%] [Generator Loss: 4.451300]
    


![png](output_15_741.png)


    37099 [Discriminator Loss: 0.187324, Acc.: 50.00%] [Generator Loss: 5.002056]
    


![png](output_15_743.png)


    37199 [Discriminator Loss: 0.216076, Acc.: 50.00%] [Generator Loss: 4.617548]
    


![png](output_15_745.png)


    37299 [Discriminator Loss: 0.195275, Acc.: 50.00%] [Generator Loss: 4.283769]
    


![png](output_15_747.png)


    37399 [Discriminator Loss: 0.269336, Acc.: 50.00%] [Generator Loss: 4.847116]
    


![png](output_15_749.png)


    37499 [Discriminator Loss: 0.238956, Acc.: 46.88%] [Generator Loss: 4.568676]
    


![png](output_15_751.png)


    37599 [Discriminator Loss: 0.213219, Acc.: 50.00%] [Generator Loss: 4.868883]
    


![png](output_15_753.png)


    37699 [Discriminator Loss: 0.229524, Acc.: 50.00%] [Generator Loss: 4.727124]
    


![png](output_15_755.png)


    37799 [Discriminator Loss: 0.231402, Acc.: 50.00%] [Generator Loss: 3.934648]
    


![png](output_15_757.png)


    37899 [Discriminator Loss: 0.201499, Acc.: 50.00%] [Generator Loss: 4.475463]
    


![png](output_15_759.png)


    37999 [Discriminator Loss: 0.206401, Acc.: 50.00%] [Generator Loss: 4.741904]
    


![png](output_15_761.png)


    38099 [Discriminator Loss: 0.248853, Acc.: 50.00%] [Generator Loss: 4.486191]
    


![png](output_15_763.png)


    38199 [Discriminator Loss: 0.226674, Acc.: 50.00%] [Generator Loss: 4.152228]
    


![png](output_15_765.png)


    38299 [Discriminator Loss: 0.186768, Acc.: 50.00%] [Generator Loss: 4.459595]
    


![png](output_15_767.png)


    38399 [Discriminator Loss: 0.217625, Acc.: 50.00%] [Generator Loss: 5.312999]
    


![png](output_15_769.png)


    38499 [Discriminator Loss: 0.244429, Acc.: 50.00%] [Generator Loss: 5.075632]
    


![png](output_15_771.png)


    38599 [Discriminator Loss: 0.188185, Acc.: 50.00%] [Generator Loss: 4.792708]
    


![png](output_15_773.png)


    38699 [Discriminator Loss: 0.182508, Acc.: 50.00%] [Generator Loss: 4.583794]
    


![png](output_15_775.png)


    38799 [Discriminator Loss: 0.180235, Acc.: 50.00%] [Generator Loss: 5.452372]
    


![png](output_15_777.png)


    38899 [Discriminator Loss: 0.201882, Acc.: 50.00%] [Generator Loss: 4.441399]
    


![png](output_15_779.png)


    38999 [Discriminator Loss: 0.190872, Acc.: 50.00%] [Generator Loss: 4.346539]
    


![png](output_15_781.png)


    39099 [Discriminator Loss: 0.197275, Acc.: 50.00%] [Generator Loss: 4.514620]
    


![png](output_15_783.png)


    39199 [Discriminator Loss: 0.203939, Acc.: 50.00%] [Generator Loss: 5.047291]
    


![png](output_15_785.png)


    39299 [Discriminator Loss: 0.174501, Acc.: 50.00%] [Generator Loss: 5.367162]
    


![png](output_15_787.png)


    39399 [Discriminator Loss: 0.216457, Acc.: 50.00%] [Generator Loss: 3.922556]
    


![png](output_15_789.png)


    39499 [Discriminator Loss: 0.237919, Acc.: 50.00%] [Generator Loss: 5.282147]
    


![png](output_15_791.png)


    39599 [Discriminator Loss: 0.187351, Acc.: 50.00%] [Generator Loss: 5.655998]
    


![png](output_15_793.png)


    39699 [Discriminator Loss: 0.187151, Acc.: 50.00%] [Generator Loss: 5.259088]
    


![png](output_15_795.png)


    39799 [Discriminator Loss: 0.196218, Acc.: 50.00%] [Generator Loss: 5.127178]
    


![png](output_15_797.png)


    39899 [Discriminator Loss: 0.222332, Acc.: 50.00%] [Generator Loss: 5.078945]
    


![png](output_15_799.png)


    39999 [Discriminator Loss: 0.201935, Acc.: 50.00%] [Generator Loss: 4.790720]
    


![png](output_15_801.png)


    40099 [Discriminator Loss: 0.198137, Acc.: 50.00%] [Generator Loss: 4.997955]
    


![png](output_15_803.png)


    40199 [Discriminator Loss: 0.247666, Acc.: 50.00%] [Generator Loss: 6.073701]
    


![png](output_15_805.png)


    40299 [Discriminator Loss: 0.222516, Acc.: 50.00%] [Generator Loss: 4.675921]
    


![png](output_15_807.png)


    40399 [Discriminator Loss: 0.200556, Acc.: 50.00%] [Generator Loss: 4.607599]
    


![png](output_15_809.png)


    40499 [Discriminator Loss: 0.192173, Acc.: 50.00%] [Generator Loss: 5.141148]
    


![png](output_15_811.png)


    40599 [Discriminator Loss: 0.204808, Acc.: 50.00%] [Generator Loss: 5.108368]
    


![png](output_15_813.png)


    40699 [Discriminator Loss: 0.222247, Acc.: 50.00%] [Generator Loss: 4.488996]
    


![png](output_15_815.png)


    40799 [Discriminator Loss: 0.187329, Acc.: 50.00%] [Generator Loss: 6.106511]
    


![png](output_15_817.png)


    40899 [Discriminator Loss: 0.185916, Acc.: 50.00%] [Generator Loss: 5.529009]
    


![png](output_15_819.png)


    40999 [Discriminator Loss: 0.183521, Acc.: 50.00%] [Generator Loss: 5.972779]
    


![png](output_15_821.png)


    41099 [Discriminator Loss: 0.199856, Acc.: 50.00%] [Generator Loss: 4.945886]
    


![png](output_15_823.png)


    41199 [Discriminator Loss: 0.382985, Acc.: 50.00%] [Generator Loss: 4.656253]
    


![png](output_15_825.png)


    41299 [Discriminator Loss: 0.198661, Acc.: 50.00%] [Generator Loss: 5.063895]
    


![png](output_15_827.png)


    41399 [Discriminator Loss: 0.194914, Acc.: 50.00%] [Generator Loss: 4.457287]
    


![png](output_15_829.png)


    41499 [Discriminator Loss: 0.236169, Acc.: 50.00%] [Generator Loss: 4.631660]
    


![png](output_15_831.png)


    41599 [Discriminator Loss: 0.269300, Acc.: 46.88%] [Generator Loss: 5.269789]
    


![png](output_15_833.png)


    41699 [Discriminator Loss: 0.185568, Acc.: 50.00%] [Generator Loss: 4.515068]
    


![png](output_15_835.png)


    41799 [Discriminator Loss: 0.197894, Acc.: 50.00%] [Generator Loss: 4.157397]
    


![png](output_15_837.png)


    41899 [Discriminator Loss: 0.195463, Acc.: 50.00%] [Generator Loss: 4.355138]
    


![png](output_15_839.png)


    41999 [Discriminator Loss: 0.193579, Acc.: 50.00%] [Generator Loss: 3.714145]
    


![png](output_15_841.png)


    42099 [Discriminator Loss: 0.295977, Acc.: 46.88%] [Generator Loss: 4.129004]
    


![png](output_15_843.png)


    42199 [Discriminator Loss: 0.264228, Acc.: 50.00%] [Generator Loss: 4.270266]
    


![png](output_15_845.png)


    42299 [Discriminator Loss: 0.201707, Acc.: 50.00%] [Generator Loss: 4.148371]
    


![png](output_15_847.png)


    42399 [Discriminator Loss: 0.206453, Acc.: 50.00%] [Generator Loss: 4.386889]
    


![png](output_15_849.png)


    42499 [Discriminator Loss: 0.195182, Acc.: 50.00%] [Generator Loss: 4.571601]
    


![png](output_15_851.png)


    42599 [Discriminator Loss: 0.214281, Acc.: 50.00%] [Generator Loss: 4.182468]
    


![png](output_15_853.png)


    42699 [Discriminator Loss: 0.199033, Acc.: 50.00%] [Generator Loss: 3.984059]
    


![png](output_15_855.png)


    42799 [Discriminator Loss: 0.178203, Acc.: 50.00%] [Generator Loss: 4.689199]
    


![png](output_15_857.png)


    42899 [Discriminator Loss: 0.209975, Acc.: 50.00%] [Generator Loss: 4.312890]
    


![png](output_15_859.png)


    42999 [Discriminator Loss: 0.230848, Acc.: 50.00%] [Generator Loss: 4.152761]
    


![png](output_15_861.png)


    43099 [Discriminator Loss: 0.248115, Acc.: 50.00%] [Generator Loss: 5.235784]
    


![png](output_15_863.png)


    43199 [Discriminator Loss: 0.189673, Acc.: 50.00%] [Generator Loss: 4.238753]
    


![png](output_15_865.png)


    43299 [Discriminator Loss: 0.227421, Acc.: 50.00%] [Generator Loss: 4.260222]
    


![png](output_15_867.png)


    43399 [Discriminator Loss: 0.270353, Acc.: 46.88%] [Generator Loss: 4.969910]
    


![png](output_15_869.png)


    43499 [Discriminator Loss: 0.197525, Acc.: 50.00%] [Generator Loss: 5.055779]
    


![png](output_15_871.png)


    43599 [Discriminator Loss: 0.219022, Acc.: 50.00%] [Generator Loss: 4.795819]
    


![png](output_15_873.png)


    43699 [Discriminator Loss: 0.267247, Acc.: 46.88%] [Generator Loss: 4.997390]
    


![png](output_15_875.png)


    43799 [Discriminator Loss: 0.183771, Acc.: 50.00%] [Generator Loss: 4.936062]
    


![png](output_15_877.png)


    43899 [Discriminator Loss: 0.198572, Acc.: 50.00%] [Generator Loss: 4.890919]
    


![png](output_15_879.png)


    43999 [Discriminator Loss: 0.188926, Acc.: 50.00%] [Generator Loss: 4.631191]
    


![png](output_15_881.png)


    44099 [Discriminator Loss: 0.258400, Acc.: 50.00%] [Generator Loss: 4.578922]
    


![png](output_15_883.png)


    44199 [Discriminator Loss: 0.202339, Acc.: 50.00%] [Generator Loss: 4.820295]
    


![png](output_15_885.png)


    44299 [Discriminator Loss: 0.219480, Acc.: 50.00%] [Generator Loss: 5.261392]
    


![png](output_15_887.png)


    44399 [Discriminator Loss: 0.202030, Acc.: 50.00%] [Generator Loss: 4.742162]
    


![png](output_15_889.png)


    44499 [Discriminator Loss: 0.184633, Acc.: 50.00%] [Generator Loss: 4.777944]
    


![png](output_15_891.png)


    44599 [Discriminator Loss: 0.187896, Acc.: 50.00%] [Generator Loss: 4.569323]
    


![png](output_15_893.png)


    44699 [Discriminator Loss: 0.191894, Acc.: 50.00%] [Generator Loss: 4.564614]
    


![png](output_15_895.png)


    44799 [Discriminator Loss: 0.183938, Acc.: 50.00%] [Generator Loss: 5.153484]
    


![png](output_15_897.png)


    44899 [Discriminator Loss: 0.182459, Acc.: 50.00%] [Generator Loss: 5.234219]
    


![png](output_15_899.png)


    44999 [Discriminator Loss: 0.184048, Acc.: 50.00%] [Generator Loss: 4.647976]
    


![png](output_15_901.png)


    45099 [Discriminator Loss: 0.181868, Acc.: 50.00%] [Generator Loss: 4.935896]
    


![png](output_15_903.png)


    45199 [Discriminator Loss: 0.189769, Acc.: 50.00%] [Generator Loss: 4.864957]
    


![png](output_15_905.png)


    45299 [Discriminator Loss: 0.217386, Acc.: 50.00%] [Generator Loss: 4.676645]
    


![png](output_15_907.png)


    45399 [Discriminator Loss: 0.183221, Acc.: 50.00%] [Generator Loss: 4.902539]
    


![png](output_15_909.png)


    45499 [Discriminator Loss: 0.270350, Acc.: 50.00%] [Generator Loss: 4.408203]
    


![png](output_15_911.png)


    45599 [Discriminator Loss: 0.277515, Acc.: 50.00%] [Generator Loss: 3.780618]
    


![png](output_15_913.png)


    45699 [Discriminator Loss: 0.196942, Acc.: 50.00%] [Generator Loss: 3.944170]
    


![png](output_15_915.png)


    45799 [Discriminator Loss: 0.226841, Acc.: 46.88%] [Generator Loss: 5.657649]
    


![png](output_15_917.png)


    45899 [Discriminator Loss: 0.194010, Acc.: 50.00%] [Generator Loss: 5.080309]
    


![png](output_15_919.png)


    45999 [Discriminator Loss: 0.187543, Acc.: 50.00%] [Generator Loss: 4.514233]
    


![png](output_15_921.png)


    46099 [Discriminator Loss: 0.197805, Acc.: 50.00%] [Generator Loss: 4.830222]
    


![png](output_15_923.png)


    46199 [Discriminator Loss: 0.192573, Acc.: 50.00%] [Generator Loss: 4.512919]
    


![png](output_15_925.png)


    46299 [Discriminator Loss: 0.194586, Acc.: 50.00%] [Generator Loss: 4.216307]
    


![png](output_15_927.png)


    46399 [Discriminator Loss: 0.195019, Acc.: 50.00%] [Generator Loss: 5.254764]
    


![png](output_15_929.png)


    46499 [Discriminator Loss: 0.288776, Acc.: 50.00%] [Generator Loss: 3.806346]
    


![png](output_15_931.png)


    46599 [Discriminator Loss: 0.220983, Acc.: 50.00%] [Generator Loss: 3.519623]
    


![png](output_15_933.png)


    46699 [Discriminator Loss: 0.212168, Acc.: 50.00%] [Generator Loss: 4.259376]
    


![png](output_15_935.png)


    46799 [Discriminator Loss: 0.197801, Acc.: 50.00%] [Generator Loss: 4.825039]
    


![png](output_15_937.png)


    46899 [Discriminator Loss: 0.195418, Acc.: 50.00%] [Generator Loss: 4.860214]
    


![png](output_15_939.png)


    46999 [Discriminator Loss: 0.192568, Acc.: 50.00%] [Generator Loss: 5.311077]
    


![png](output_15_941.png)


    47099 [Discriminator Loss: 0.206550, Acc.: 50.00%] [Generator Loss: 4.599804]
    


![png](output_15_943.png)


    47199 [Discriminator Loss: 0.215904, Acc.: 50.00%] [Generator Loss: 4.582682]
    


![png](output_15_945.png)


    47299 [Discriminator Loss: 0.200076, Acc.: 50.00%] [Generator Loss: 4.744141]
    


![png](output_15_947.png)


    47399 [Discriminator Loss: 0.209359, Acc.: 50.00%] [Generator Loss: 4.145015]
    


![png](output_15_949.png)


    47499 [Discriminator Loss: 0.214215, Acc.: 50.00%] [Generator Loss: 5.992319]
    


![png](output_15_951.png)


    47599 [Discriminator Loss: 0.189115, Acc.: 50.00%] [Generator Loss: 5.818109]
    


![png](output_15_953.png)


    47699 [Discriminator Loss: 0.198146, Acc.: 50.00%] [Generator Loss: 5.606324]
    


![png](output_15_955.png)


    47799 [Discriminator Loss: 0.244122, Acc.: 50.00%] [Generator Loss: 4.835163]
    


![png](output_15_957.png)


    47899 [Discriminator Loss: 0.221908, Acc.: 50.00%] [Generator Loss: 4.699048]
    


![png](output_15_959.png)


    47999 [Discriminator Loss: 0.264137, Acc.: 50.00%] [Generator Loss: 4.911812]
    


![png](output_15_961.png)


    48099 [Discriminator Loss: 0.186679, Acc.: 50.00%] [Generator Loss: 4.643673]
    


![png](output_15_963.png)


    48199 [Discriminator Loss: 0.198521, Acc.: 50.00%] [Generator Loss: 4.421924]
    


![png](output_15_965.png)


    48299 [Discriminator Loss: 0.233221, Acc.: 50.00%] [Generator Loss: 3.664936]
    


![png](output_15_967.png)


    48399 [Discriminator Loss: 0.241129, Acc.: 50.00%] [Generator Loss: 3.808108]
    


![png](output_15_969.png)


    48499 [Discriminator Loss: 0.252381, Acc.: 50.00%] [Generator Loss: 4.325066]
    


![png](output_15_971.png)


    48599 [Discriminator Loss: 0.200090, Acc.: 50.00%] [Generator Loss: 4.485260]
    


![png](output_15_973.png)


    48699 [Discriminator Loss: 0.196284, Acc.: 50.00%] [Generator Loss: 4.768801]
    


![png](output_15_975.png)


    48799 [Discriminator Loss: 0.182787, Acc.: 50.00%] [Generator Loss: 4.749042]
    


![png](output_15_977.png)


    48899 [Discriminator Loss: 0.186637, Acc.: 50.00%] [Generator Loss: 4.716693]
    


![png](output_15_979.png)


    48999 [Discriminator Loss: 0.219343, Acc.: 50.00%] [Generator Loss: 4.173238]
    


![png](output_15_981.png)


    49099 [Discriminator Loss: 0.182187, Acc.: 50.00%] [Generator Loss: 5.382607]
    


![png](output_15_983.png)


    49199 [Discriminator Loss: 0.213579, Acc.: 50.00%] [Generator Loss: 5.877297]
    


![png](output_15_985.png)


    49299 [Discriminator Loss: 0.194653, Acc.: 50.00%] [Generator Loss: 4.873508]
    


![png](output_15_987.png)


    49399 [Discriminator Loss: 0.222862, Acc.: 50.00%] [Generator Loss: 4.647823]
    


![png](output_15_989.png)


    49499 [Discriminator Loss: 0.271454, Acc.: 50.00%] [Generator Loss: 4.518676]
    


![png](output_15_991.png)


    49599 [Discriminator Loss: 0.221793, Acc.: 50.00%] [Generator Loss: 4.566399]
    


![png](output_15_993.png)


    49699 [Discriminator Loss: 0.203305, Acc.: 50.00%] [Generator Loss: 4.377223]
    


![png](output_15_995.png)


    49799 [Discriminator Loss: 0.196093, Acc.: 50.00%] [Generator Loss: 4.235477]
    


![png](output_15_997.png)


    49899 [Discriminator Loss: 0.206301, Acc.: 50.00%] [Generator Loss: 4.684705]
    


![png](output_15_999.png)


    49999 [Discriminator Loss: 0.229077, Acc.: 50.00%] [Generator Loss: 4.194158]
    


![png](output_15_1001.png)


    50099 [Discriminator Loss: 0.178364, Acc.: 50.00%] [Generator Loss: 4.784287]
    


![png](output_15_1003.png)


    50199 [Discriminator Loss: 0.214011, Acc.: 50.00%] [Generator Loss: 4.263295]
    


![png](output_15_1005.png)


    50299 [Discriminator Loss: 0.211252, Acc.: 50.00%] [Generator Loss: 4.731246]
    


![png](output_15_1007.png)


    50399 [Discriminator Loss: 0.212568, Acc.: 50.00%] [Generator Loss: 4.773615]
    


![png](output_15_1009.png)


    50499 [Discriminator Loss: 0.267102, Acc.: 46.88%] [Generator Loss: 5.081255]
    


![png](output_15_1011.png)


    50599 [Discriminator Loss: 0.186589, Acc.: 50.00%] [Generator Loss: 4.928342]
    


![png](output_15_1013.png)


    50699 [Discriminator Loss: 0.214353, Acc.: 50.00%] [Generator Loss: 4.531277]
    


![png](output_15_1015.png)


    50799 [Discriminator Loss: 0.200183, Acc.: 50.00%] [Generator Loss: 4.545346]
    


![png](output_15_1017.png)


    50899 [Discriminator Loss: 0.190668, Acc.: 50.00%] [Generator Loss: 5.335259]
    


![png](output_15_1019.png)


    50999 [Discriminator Loss: 0.199720, Acc.: 50.00%] [Generator Loss: 4.432771]
    


![png](output_15_1021.png)


    51099 [Discriminator Loss: 0.199689, Acc.: 50.00%] [Generator Loss: 4.377468]
    


![png](output_15_1023.png)


    51199 [Discriminator Loss: 0.190598, Acc.: 50.00%] [Generator Loss: 4.809628]
    


![png](output_15_1025.png)


    51299 [Discriminator Loss: 0.182594, Acc.: 50.00%] [Generator Loss: 4.832110]
    


![png](output_15_1027.png)


    51399 [Discriminator Loss: 0.203315, Acc.: 50.00%] [Generator Loss: 5.814363]
    


![png](output_15_1029.png)


    51499 [Discriminator Loss: 0.193157, Acc.: 50.00%] [Generator Loss: 4.646562]
    


![png](output_15_1031.png)


    51599 [Discriminator Loss: 0.197068, Acc.: 50.00%] [Generator Loss: 4.788333]
    


![png](output_15_1033.png)


    51699 [Discriminator Loss: 0.183994, Acc.: 50.00%] [Generator Loss: 5.183094]
    


![png](output_15_1035.png)


    51799 [Discriminator Loss: 0.190907, Acc.: 50.00%] [Generator Loss: 4.909018]
    


![png](output_15_1037.png)


    51899 [Discriminator Loss: 0.192448, Acc.: 50.00%] [Generator Loss: 5.325535]
    


![png](output_15_1039.png)


    51999 [Discriminator Loss: 0.178601, Acc.: 50.00%] [Generator Loss: 4.882923]
    


![png](output_15_1041.png)


    52099 [Discriminator Loss: 0.191751, Acc.: 50.00%] [Generator Loss: 6.020660]
    


![png](output_15_1043.png)


    52199 [Discriminator Loss: 0.214555, Acc.: 50.00%] [Generator Loss: 4.095078]
    


![png](output_15_1045.png)


    52299 [Discriminator Loss: 0.183027, Acc.: 50.00%] [Generator Loss: 5.451503]
    


![png](output_15_1047.png)


    52399 [Discriminator Loss: 0.186559, Acc.: 50.00%] [Generator Loss: 4.496674]
    


![png](output_15_1049.png)


    52499 [Discriminator Loss: 0.185106, Acc.: 50.00%] [Generator Loss: 5.530549]
    


![png](output_15_1051.png)


    52599 [Discriminator Loss: 0.218030, Acc.: 50.00%] [Generator Loss: 4.451962]
    


![png](output_15_1053.png)


    52699 [Discriminator Loss: 0.183295, Acc.: 50.00%] [Generator Loss: 4.419615]
    


![png](output_15_1055.png)


    52799 [Discriminator Loss: 0.188824, Acc.: 50.00%] [Generator Loss: 4.863238]
    


![png](output_15_1057.png)


    52899 [Discriminator Loss: 0.205919, Acc.: 50.00%] [Generator Loss: 5.058976]
    


![png](output_15_1059.png)


    52999 [Discriminator Loss: 0.178464, Acc.: 50.00%] [Generator Loss: 6.067551]
    


![png](output_15_1061.png)


    53099 [Discriminator Loss: 0.218218, Acc.: 50.00%] [Generator Loss: 4.772855]
    


![png](output_15_1063.png)


    53199 [Discriminator Loss: 0.189647, Acc.: 50.00%] [Generator Loss: 4.830702]
    


![png](output_15_1065.png)


    53299 [Discriminator Loss: 0.199454, Acc.: 50.00%] [Generator Loss: 4.724833]
    


![png](output_15_1067.png)


    53399 [Discriminator Loss: 0.213154, Acc.: 50.00%] [Generator Loss: 5.212846]
    


![png](output_15_1069.png)


    53499 [Discriminator Loss: 0.209156, Acc.: 50.00%] [Generator Loss: 5.028785]
    


![png](output_15_1071.png)


    53599 [Discriminator Loss: 0.182448, Acc.: 50.00%] [Generator Loss: 5.060333]
    


![png](output_15_1073.png)


    53699 [Discriminator Loss: 0.215559, Acc.: 50.00%] [Generator Loss: 5.206966]
    


![png](output_15_1075.png)


    53799 [Discriminator Loss: 0.192472, Acc.: 50.00%] [Generator Loss: 4.482029]
    


![png](output_15_1077.png)


    53899 [Discriminator Loss: 0.182998, Acc.: 50.00%] [Generator Loss: 5.336737]
    


![png](output_15_1079.png)


    53999 [Discriminator Loss: 0.196121, Acc.: 50.00%] [Generator Loss: 4.435163]
    


![png](output_15_1081.png)


    54099 [Discriminator Loss: 0.182324, Acc.: 50.00%] [Generator Loss: 5.101933]
    


![png](output_15_1083.png)


    54199 [Discriminator Loss: 0.190712, Acc.: 50.00%] [Generator Loss: 4.420429]
    


![png](output_15_1085.png)


    54299 [Discriminator Loss: 0.253974, Acc.: 50.00%] [Generator Loss: 3.919936]
    


![png](output_15_1087.png)


    54399 [Discriminator Loss: 0.192055, Acc.: 50.00%] [Generator Loss: 5.076313]
    


![png](output_15_1089.png)


    54499 [Discriminator Loss: 0.250814, Acc.: 50.00%] [Generator Loss: 4.087386]
    


![png](output_15_1091.png)


    54599 [Discriminator Loss: 0.228189, Acc.: 50.00%] [Generator Loss: 4.810251]
    


![png](output_15_1093.png)


    54699 [Discriminator Loss: 0.196840, Acc.: 50.00%] [Generator Loss: 4.675594]
    


![png](output_15_1095.png)


    54799 [Discriminator Loss: 0.187098, Acc.: 50.00%] [Generator Loss: 5.214050]
    


![png](output_15_1097.png)


    54899 [Discriminator Loss: 0.198232, Acc.: 50.00%] [Generator Loss: 4.807789]
    


![png](output_15_1099.png)


    54999 [Discriminator Loss: 0.228319, Acc.: 50.00%] [Generator Loss: 4.589862]
    


![png](output_15_1101.png)


    55099 [Discriminator Loss: 0.202460, Acc.: 50.00%] [Generator Loss: 4.838396]
    


![png](output_15_1103.png)


    55199 [Discriminator Loss: 0.199011, Acc.: 50.00%] [Generator Loss: 4.399462]
    


![png](output_15_1105.png)


    55299 [Discriminator Loss: 0.263919, Acc.: 50.00%] [Generator Loss: 4.073527]
    


![png](output_15_1107.png)


    55399 [Discriminator Loss: 0.201177, Acc.: 50.00%] [Generator Loss: 4.796829]
    


![png](output_15_1109.png)


    55499 [Discriminator Loss: 0.186942, Acc.: 50.00%] [Generator Loss: 4.754351]
    


![png](output_15_1111.png)


    55599 [Discriminator Loss: 0.191728, Acc.: 50.00%] [Generator Loss: 5.010515]
    


![png](output_15_1113.png)


    55699 [Discriminator Loss: 0.190959, Acc.: 50.00%] [Generator Loss: 4.857539]
    


![png](output_15_1115.png)


    55799 [Discriminator Loss: 0.186594, Acc.: 50.00%] [Generator Loss: 4.933507]
    


![png](output_15_1117.png)


    55899 [Discriminator Loss: 0.182021, Acc.: 50.00%] [Generator Loss: 4.871453]
    


![png](output_15_1119.png)


    55999 [Discriminator Loss: 0.186372, Acc.: 50.00%] [Generator Loss: 4.899321]
    


![png](output_15_1121.png)


    56099 [Discriminator Loss: 0.202391, Acc.: 50.00%] [Generator Loss: 4.795869]
    


![png](output_15_1123.png)


    56199 [Discriminator Loss: 0.187756, Acc.: 50.00%] [Generator Loss: 5.517591]
    


![png](output_15_1125.png)


    56299 [Discriminator Loss: 0.194306, Acc.: 50.00%] [Generator Loss: 5.315543]
    


![png](output_15_1127.png)


    56399 [Discriminator Loss: 0.181106, Acc.: 50.00%] [Generator Loss: 5.530529]
    


![png](output_15_1129.png)


    56499 [Discriminator Loss: 0.181522, Acc.: 50.00%] [Generator Loss: 5.492394]
    


![png](output_15_1131.png)


    56599 [Discriminator Loss: 0.219526, Acc.: 50.00%] [Generator Loss: 4.908090]
    


![png](output_15_1133.png)


    56699 [Discriminator Loss: 0.187782, Acc.: 50.00%] [Generator Loss: 6.191876]
    


![png](output_15_1135.png)


    56799 [Discriminator Loss: 0.236013, Acc.: 50.00%] [Generator Loss: 5.156234]
    


![png](output_15_1137.png)


    56899 [Discriminator Loss: 0.203991, Acc.: 50.00%] [Generator Loss: 4.995097]
    


![png](output_15_1139.png)


    56999 [Discriminator Loss: 0.184986, Acc.: 50.00%] [Generator Loss: 6.180265]
    


![png](output_15_1141.png)


    57099 [Discriminator Loss: 0.220356, Acc.: 50.00%] [Generator Loss: 5.178768]
    


![png](output_15_1143.png)


    57199 [Discriminator Loss: 0.182020, Acc.: 50.00%] [Generator Loss: 5.939548]
    


![png](output_15_1145.png)


    57299 [Discriminator Loss: 0.205454, Acc.: 50.00%] [Generator Loss: 5.067138]
    


![png](output_15_1147.png)


    57399 [Discriminator Loss: 0.188331, Acc.: 50.00%] [Generator Loss: 6.195132]
    


![png](output_15_1149.png)


    57499 [Discriminator Loss: 0.216902, Acc.: 50.00%] [Generator Loss: 4.143826]
    


![png](output_15_1151.png)


    57599 [Discriminator Loss: 0.199616, Acc.: 50.00%] [Generator Loss: 6.037016]
    


![png](output_15_1153.png)


    57699 [Discriminator Loss: 0.213470, Acc.: 50.00%] [Generator Loss: 4.559368]
    


![png](output_15_1155.png)


    57799 [Discriminator Loss: 0.221367, Acc.: 50.00%] [Generator Loss: 5.392152]
    


![png](output_15_1157.png)


    57899 [Discriminator Loss: 0.190275, Acc.: 50.00%] [Generator Loss: 4.863247]
    


![png](output_15_1159.png)


    57999 [Discriminator Loss: 0.189381, Acc.: 50.00%] [Generator Loss: 6.032892]
    


![png](output_15_1161.png)


    58099 [Discriminator Loss: 0.195924, Acc.: 50.00%] [Generator Loss: 5.184461]
    


![png](output_15_1163.png)


    58199 [Discriminator Loss: 0.178104, Acc.: 50.00%] [Generator Loss: 5.409165]
    


![png](output_15_1165.png)


    58299 [Discriminator Loss: 0.181832, Acc.: 50.00%] [Generator Loss: 5.632461]
    


![png](output_15_1167.png)


    58399 [Discriminator Loss: 0.187249, Acc.: 50.00%] [Generator Loss: 5.718653]
    


![png](output_15_1169.png)


    58499 [Discriminator Loss: 0.184930, Acc.: 50.00%] [Generator Loss: 5.417265]
    


![png](output_15_1171.png)


    58599 [Discriminator Loss: 0.205114, Acc.: 50.00%] [Generator Loss: 4.402874]
    


![png](output_15_1173.png)


    58699 [Discriminator Loss: 0.189650, Acc.: 50.00%] [Generator Loss: 4.892275]
    


![png](output_15_1175.png)


    58799 [Discriminator Loss: 0.195843, Acc.: 50.00%] [Generator Loss: 4.204408]
    


![png](output_15_1177.png)


    58899 [Discriminator Loss: 0.201929, Acc.: 50.00%] [Generator Loss: 4.023510]
    


![png](output_15_1179.png)


    58999 [Discriminator Loss: 0.194585, Acc.: 50.00%] [Generator Loss: 4.337784]
    


![png](output_15_1181.png)


    59099 [Discriminator Loss: 0.209366, Acc.: 50.00%] [Generator Loss: 4.330474]
    


![png](output_15_1183.png)


    59199 [Discriminator Loss: 0.194538, Acc.: 50.00%] [Generator Loss: 4.642085]
    


![png](output_15_1185.png)


    59299 [Discriminator Loss: 0.300206, Acc.: 50.00%] [Generator Loss: 4.544250]
    


![png](output_15_1187.png)


    59399 [Discriminator Loss: 0.202710, Acc.: 50.00%] [Generator Loss: 5.088209]
    


![png](output_15_1189.png)


    59499 [Discriminator Loss: 0.238408, Acc.: 50.00%] [Generator Loss: 4.741205]
    


![png](output_15_1191.png)


    59599 [Discriminator Loss: 0.195561, Acc.: 50.00%] [Generator Loss: 5.478787]
    


![png](output_15_1193.png)


    59699 [Discriminator Loss: 0.232108, Acc.: 50.00%] [Generator Loss: 5.293727]
    


![png](output_15_1195.png)


    59799 [Discriminator Loss: 0.206062, Acc.: 50.00%] [Generator Loss: 5.529275]
    


![png](output_15_1197.png)


    59899 [Discriminator Loss: 0.195104, Acc.: 50.00%] [Generator Loss: 4.946084]
    


![png](output_15_1199.png)


    59999 [Discriminator Loss: 0.280329, Acc.: 50.00%] [Generator Loss: 4.998095]
    


![png](output_15_1201.png)


    60099 [Discriminator Loss: 0.205403, Acc.: 50.00%] [Generator Loss: 4.333411]
    


![png](output_15_1203.png)


    60199 [Discriminator Loss: 0.180072, Acc.: 50.00%] [Generator Loss: 5.469588]
    


![png](output_15_1205.png)


    60299 [Discriminator Loss: 0.180707, Acc.: 50.00%] [Generator Loss: 5.819433]
    


![png](output_15_1207.png)


    60399 [Discriminator Loss: 0.260948, Acc.: 50.00%] [Generator Loss: 4.311582]
    


![png](output_15_1209.png)


    60499 [Discriminator Loss: 0.183116, Acc.: 50.00%] [Generator Loss: 6.044875]
    


![png](output_15_1211.png)


    60599 [Discriminator Loss: 0.185712, Acc.: 50.00%] [Generator Loss: 5.034066]
    


![png](output_15_1213.png)


    60699 [Discriminator Loss: 0.190475, Acc.: 50.00%] [Generator Loss: 5.531404]
    


![png](output_15_1215.png)


    60799 [Discriminator Loss: 0.190574, Acc.: 50.00%] [Generator Loss: 4.878501]
    


![png](output_15_1217.png)


    60899 [Discriminator Loss: 0.187152, Acc.: 50.00%] [Generator Loss: 5.246289]
    


![png](output_15_1219.png)


    60999 [Discriminator Loss: 0.186976, Acc.: 50.00%] [Generator Loss: 5.431002]
    


![png](output_15_1221.png)


    61099 [Discriminator Loss: 0.205112, Acc.: 50.00%] [Generator Loss: 5.196317]
    


![png](output_15_1223.png)


    61199 [Discriminator Loss: 0.204427, Acc.: 50.00%] [Generator Loss: 4.238957]
    


![png](output_15_1225.png)


    61299 [Discriminator Loss: 0.249433, Acc.: 50.00%] [Generator Loss: 3.710929]
    


![png](output_15_1227.png)


    61399 [Discriminator Loss: 0.200049, Acc.: 50.00%] [Generator Loss: 3.889353]
    


![png](output_15_1229.png)


    61499 [Discriminator Loss: 0.204248, Acc.: 50.00%] [Generator Loss: 5.963184]
    


![png](output_15_1231.png)


    61599 [Discriminator Loss: 0.186252, Acc.: 50.00%] [Generator Loss: 5.298292]
    


![png](output_15_1233.png)


    61699 [Discriminator Loss: 0.204698, Acc.: 50.00%] [Generator Loss: 4.269163]
    


![png](output_15_1235.png)


    61799 [Discriminator Loss: 0.179246, Acc.: 50.00%] [Generator Loss: 5.275495]
    


![png](output_15_1237.png)


    61899 [Discriminator Loss: 0.225754, Acc.: 50.00%] [Generator Loss: 4.389694]
    


![png](output_15_1239.png)


    61999 [Discriminator Loss: 0.171725, Acc.: 50.00%] [Generator Loss: 5.391454]
    


![png](output_15_1241.png)


    62099 [Discriminator Loss: 0.276629, Acc.: 50.00%] [Generator Loss: 3.989589]
    


![png](output_15_1243.png)


    62199 [Discriminator Loss: 0.180416, Acc.: 50.00%] [Generator Loss: 5.838820]
    


![png](output_15_1245.png)


    62299 [Discriminator Loss: 0.181093, Acc.: 50.00%] [Generator Loss: 5.323298]
    


![png](output_15_1247.png)


    62399 [Discriminator Loss: 0.174563, Acc.: 50.00%] [Generator Loss: 6.093063]
    


![png](output_15_1249.png)


    62499 [Discriminator Loss: 0.197901, Acc.: 50.00%] [Generator Loss: 4.846997]
    


![png](output_15_1251.png)


    62599 [Discriminator Loss: 0.196533, Acc.: 50.00%] [Generator Loss: 4.538827]
    


![png](output_15_1253.png)


    62699 [Discriminator Loss: 0.199873, Acc.: 50.00%] [Generator Loss: 5.208162]
    


![png](output_15_1255.png)


    62799 [Discriminator Loss: 0.236580, Acc.: 50.00%] [Generator Loss: 5.395514]
    


![png](output_15_1257.png)


    62899 [Discriminator Loss: 0.177799, Acc.: 50.00%] [Generator Loss: 5.722693]
    


![png](output_15_1259.png)


    62999 [Discriminator Loss: 0.201468, Acc.: 50.00%] [Generator Loss: 4.721533]
    


![png](output_15_1261.png)


    63099 [Discriminator Loss: 0.332784, Acc.: 50.00%] [Generator Loss: 4.741244]
    


![png](output_15_1263.png)


    63199 [Discriminator Loss: 0.180618, Acc.: 50.00%] [Generator Loss: 6.840654]
    


![png](output_15_1265.png)


    63299 [Discriminator Loss: 0.185739, Acc.: 50.00%] [Generator Loss: 4.992869]
    


![png](output_15_1267.png)


    63399 [Discriminator Loss: 0.206500, Acc.: 50.00%] [Generator Loss: 5.535220]
    


![png](output_15_1269.png)


    63499 [Discriminator Loss: 0.184753, Acc.: 50.00%] [Generator Loss: 6.077363]
    


![png](output_15_1271.png)


    63599 [Discriminator Loss: 0.184139, Acc.: 50.00%] [Generator Loss: 5.251460]
    


![png](output_15_1273.png)


    63699 [Discriminator Loss: 0.215179, Acc.: 50.00%] [Generator Loss: 5.553777]
    


![png](output_15_1275.png)


    63799 [Discriminator Loss: 0.221210, Acc.: 50.00%] [Generator Loss: 4.681421]
    


![png](output_15_1277.png)


    63899 [Discriminator Loss: 0.193411, Acc.: 50.00%] [Generator Loss: 4.586529]
    


![png](output_15_1279.png)


    63999 [Discriminator Loss: 0.200119, Acc.: 50.00%] [Generator Loss: 5.517238]
    


![png](output_15_1281.png)


    64099 [Discriminator Loss: 0.201872, Acc.: 50.00%] [Generator Loss: 4.866325]
    


![png](output_15_1283.png)


    64199 [Discriminator Loss: 0.192110, Acc.: 50.00%] [Generator Loss: 4.714128]
    


![png](output_15_1285.png)


    64299 [Discriminator Loss: 0.222563, Acc.: 50.00%] [Generator Loss: 5.021951]
    


![png](output_15_1287.png)


    64399 [Discriminator Loss: 0.181120, Acc.: 50.00%] [Generator Loss: 5.096130]
    


![png](output_15_1289.png)


    64499 [Discriminator Loss: 0.203182, Acc.: 50.00%] [Generator Loss: 5.378765]
    


![png](output_15_1291.png)


    64599 [Discriminator Loss: 0.184732, Acc.: 50.00%] [Generator Loss: 6.419049]
    


![png](output_15_1293.png)


    64699 [Discriminator Loss: 0.244322, Acc.: 50.00%] [Generator Loss: 6.768348]
    


![png](output_15_1295.png)


    64799 [Discriminator Loss: 0.191159, Acc.: 50.00%] [Generator Loss: 4.815443]
    


![png](output_15_1297.png)


    64899 [Discriminator Loss: 0.178483, Acc.: 50.00%] [Generator Loss: 6.122252]
    


![png](output_15_1299.png)


    64999 [Discriminator Loss: 0.242836, Acc.: 50.00%] [Generator Loss: 4.709971]
    


![png](output_15_1301.png)


    65099 [Discriminator Loss: 0.184658, Acc.: 50.00%] [Generator Loss: 5.406112]
    


![png](output_15_1303.png)


    65199 [Discriminator Loss: 0.177210, Acc.: 50.00%] [Generator Loss: 5.939996]
    


![png](output_15_1305.png)


    65299 [Discriminator Loss: 0.246686, Acc.: 50.00%] [Generator Loss: 5.736398]
    


![png](output_15_1307.png)


    65399 [Discriminator Loss: 0.206372, Acc.: 50.00%] [Generator Loss: 4.912695]
    


![png](output_15_1309.png)


    65499 [Discriminator Loss: 0.195167, Acc.: 50.00%] [Generator Loss: 5.043265]
    


![png](output_15_1311.png)


    65599 [Discriminator Loss: 0.231317, Acc.: 50.00%] [Generator Loss: 3.908410]
    


![png](output_15_1313.png)


    65699 [Discriminator Loss: 0.197282, Acc.: 50.00%] [Generator Loss: 4.957705]
    


![png](output_15_1315.png)


    65799 [Discriminator Loss: 0.199209, Acc.: 50.00%] [Generator Loss: 4.996157]
    


![png](output_15_1317.png)


    65899 [Discriminator Loss: 0.193164, Acc.: 50.00%] [Generator Loss: 5.228665]
    


![png](output_15_1319.png)


    65999 [Discriminator Loss: 0.213876, Acc.: 50.00%] [Generator Loss: 5.611640]
    


![png](output_15_1321.png)


    66099 [Discriminator Loss: 0.193245, Acc.: 50.00%] [Generator Loss: 6.263458]
    


![png](output_15_1323.png)


    66199 [Discriminator Loss: 0.182455, Acc.: 50.00%] [Generator Loss: 5.489256]
    


![png](output_15_1325.png)


    66299 [Discriminator Loss: 0.177602, Acc.: 50.00%] [Generator Loss: 6.900891]
    


![png](output_15_1327.png)


    66399 [Discriminator Loss: 0.183418, Acc.: 50.00%] [Generator Loss: 4.769927]
    


![png](output_15_1329.png)


    66499 [Discriminator Loss: 0.246876, Acc.: 50.00%] [Generator Loss: 4.543059]
    


![png](output_15_1331.png)


    66599 [Discriminator Loss: 0.178365, Acc.: 50.00%] [Generator Loss: 5.668295]
    


![png](output_15_1333.png)


    66699 [Discriminator Loss: 0.195530, Acc.: 50.00%] [Generator Loss: 5.127567]
    


![png](output_15_1335.png)


    66799 [Discriminator Loss: 0.177011, Acc.: 50.00%] [Generator Loss: 6.585788]
    


![png](output_15_1337.png)


    66899 [Discriminator Loss: 0.185580, Acc.: 50.00%] [Generator Loss: 6.434483]
    


![png](output_15_1339.png)


    66999 [Discriminator Loss: 0.271415, Acc.: 50.00%] [Generator Loss: 5.595269]
    


![png](output_15_1341.png)


    67099 [Discriminator Loss: 0.331598, Acc.: 50.00%] [Generator Loss: 4.187827]
    


![png](output_15_1343.png)


    67199 [Discriminator Loss: 0.207127, Acc.: 50.00%] [Generator Loss: 5.087336]
    


![png](output_15_1345.png)


    67299 [Discriminator Loss: 0.236338, Acc.: 50.00%] [Generator Loss: 4.311087]
    


![png](output_15_1347.png)


    67399 [Discriminator Loss: 0.210430, Acc.: 50.00%] [Generator Loss: 5.601530]
    


![png](output_15_1349.png)


    67499 [Discriminator Loss: 0.189927, Acc.: 50.00%] [Generator Loss: 5.513662]
    


![png](output_15_1351.png)


    67599 [Discriminator Loss: 0.185745, Acc.: 50.00%] [Generator Loss: 5.841928]
    


![png](output_15_1353.png)


    67699 [Discriminator Loss: 0.225558, Acc.: 50.00%] [Generator Loss: 4.650061]
    


![png](output_15_1355.png)


    67799 [Discriminator Loss: 0.187923, Acc.: 50.00%] [Generator Loss: 5.644242]
    


![png](output_15_1357.png)


    67899 [Discriminator Loss: 0.240230, Acc.: 50.00%] [Generator Loss: 5.672023]
    


![png](output_15_1359.png)


    67999 [Discriminator Loss: 0.183978, Acc.: 50.00%] [Generator Loss: 5.018893]
    


![png](output_15_1361.png)


    68099 [Discriminator Loss: 0.184871, Acc.: 50.00%] [Generator Loss: 4.616078]
    


![png](output_15_1363.png)


    68199 [Discriminator Loss: 0.217865, Acc.: 50.00%] [Generator Loss: 4.619178]
    


![png](output_15_1365.png)


    68299 [Discriminator Loss: 0.197257, Acc.: 50.00%] [Generator Loss: 6.013505]
    


![png](output_15_1367.png)


    68399 [Discriminator Loss: 0.186427, Acc.: 50.00%] [Generator Loss: 4.609657]
    


![png](output_15_1369.png)


    68499 [Discriminator Loss: 0.211482, Acc.: 50.00%] [Generator Loss: 5.137599]
    


![png](output_15_1371.png)


    68599 [Discriminator Loss: 0.271062, Acc.: 50.00%] [Generator Loss: 4.057114]
    


![png](output_15_1373.png)


    68699 [Discriminator Loss: 0.217663, Acc.: 46.88%] [Generator Loss: 4.923107]
    


![png](output_15_1375.png)


    68799 [Discriminator Loss: 0.217642, Acc.: 50.00%] [Generator Loss: 4.825864]
    


![png](output_15_1377.png)


    68899 [Discriminator Loss: 0.178450, Acc.: 50.00%] [Generator Loss: 5.676708]
    


![png](output_15_1379.png)


    68999 [Discriminator Loss: 0.177318, Acc.: 50.00%] [Generator Loss: 5.063901]
    


![png](output_15_1381.png)


    69099 [Discriminator Loss: 0.278356, Acc.: 50.00%] [Generator Loss: 5.334432]
    


![png](output_15_1383.png)


    69199 [Discriminator Loss: 0.197847, Acc.: 50.00%] [Generator Loss: 5.553760]
    


![png](output_15_1385.png)


    69299 [Discriminator Loss: 0.235828, Acc.: 50.00%] [Generator Loss: 4.398150]
    


![png](output_15_1387.png)


    69399 [Discriminator Loss: 0.235571, Acc.: 46.88%] [Generator Loss: 6.278265]
    


![png](output_15_1389.png)


    69499 [Discriminator Loss: 0.181432, Acc.: 50.00%] [Generator Loss: 5.105444]
    


![png](output_15_1391.png)


    69599 [Discriminator Loss: 0.172923, Acc.: 50.00%] [Generator Loss: 5.433429]
    


![png](output_15_1393.png)


    69699 [Discriminator Loss: 0.358500, Acc.: 50.00%] [Generator Loss: 4.933774]
    


![png](output_15_1395.png)


    69799 [Discriminator Loss: 0.180337, Acc.: 50.00%] [Generator Loss: 4.449615]
    


![png](output_15_1397.png)


    69899 [Discriminator Loss: 0.194160, Acc.: 50.00%] [Generator Loss: 4.986193]
    


![png](output_15_1399.png)


    69999 [Discriminator Loss: 0.192391, Acc.: 50.00%] [Generator Loss: 5.612138]
    


![png](output_15_1401.png)


    70099 [Discriminator Loss: 0.184093, Acc.: 50.00%] [Generator Loss: 5.044040]
    


![png](output_15_1403.png)


    70199 [Discriminator Loss: 0.219886, Acc.: 50.00%] [Generator Loss: 4.495447]
    


![png](output_15_1405.png)


    70299 [Discriminator Loss: 0.195690, Acc.: 50.00%] [Generator Loss: 4.516527]
    


![png](output_15_1407.png)


    70399 [Discriminator Loss: 0.180870, Acc.: 50.00%] [Generator Loss: 7.104033]
    


![png](output_15_1409.png)


    70499 [Discriminator Loss: 0.210519, Acc.: 50.00%] [Generator Loss: 4.474022]
    


![png](output_15_1411.png)


    70599 [Discriminator Loss: 0.235010, Acc.: 50.00%] [Generator Loss: 5.351053]
    


![png](output_15_1413.png)


    70699 [Discriminator Loss: 0.188221, Acc.: 50.00%] [Generator Loss: 4.784234]
    


![png](output_15_1415.png)


    70799 [Discriminator Loss: 0.200834, Acc.: 50.00%] [Generator Loss: 5.913697]
    


![png](output_15_1417.png)


    70899 [Discriminator Loss: 0.259967, Acc.: 50.00%] [Generator Loss: 3.603112]
    


![png](output_15_1419.png)


    70999 [Discriminator Loss: 0.276907, Acc.: 50.00%] [Generator Loss: 4.408277]
    


![png](output_15_1421.png)


    71099 [Discriminator Loss: 0.183955, Acc.: 50.00%] [Generator Loss: 6.014843]
    


![png](output_15_1423.png)


    71199 [Discriminator Loss: 0.214543, Acc.: 50.00%] [Generator Loss: 5.421687]
    


![png](output_15_1425.png)


    71299 [Discriminator Loss: 0.186939, Acc.: 50.00%] [Generator Loss: 5.623305]
    


![png](output_15_1427.png)


    71399 [Discriminator Loss: 0.185686, Acc.: 50.00%] [Generator Loss: 5.676881]
    


![png](output_15_1429.png)


    71499 [Discriminator Loss: 0.175385, Acc.: 50.00%] [Generator Loss: 6.311950]
    


![png](output_15_1431.png)


    71599 [Discriminator Loss: 0.262590, Acc.: 50.00%] [Generator Loss: 4.399297]
    


![png](output_15_1433.png)


    71699 [Discriminator Loss: 0.190695, Acc.: 50.00%] [Generator Loss: 5.254608]
    


![png](output_15_1435.png)


    71799 [Discriminator Loss: 0.210929, Acc.: 50.00%] [Generator Loss: 5.484785]
    


![png](output_15_1437.png)


    71899 [Discriminator Loss: 0.205130, Acc.: 46.88%] [Generator Loss: 5.220483]
    


![png](output_15_1439.png)


    71999 [Discriminator Loss: 0.186341, Acc.: 50.00%] [Generator Loss: 4.686289]
    


![png](output_15_1441.png)


    72099 [Discriminator Loss: 0.219898, Acc.: 50.00%] [Generator Loss: 4.994912]
    


![png](output_15_1443.png)


    72199 [Discriminator Loss: 0.218162, Acc.: 50.00%] [Generator Loss: 4.656605]
    


![png](output_15_1445.png)


    72299 [Discriminator Loss: 0.201746, Acc.: 50.00%] [Generator Loss: 5.492301]
    


![png](output_15_1447.png)


    72399 [Discriminator Loss: 0.226556, Acc.: 50.00%] [Generator Loss: 4.427594]
    


![png](output_15_1449.png)


    72499 [Discriminator Loss: 0.193031, Acc.: 50.00%] [Generator Loss: 5.790577]
    


![png](output_15_1451.png)


    72599 [Discriminator Loss: 0.182541, Acc.: 50.00%] [Generator Loss: 5.025278]
    


![png](output_15_1453.png)


    72699 [Discriminator Loss: 0.202549, Acc.: 50.00%] [Generator Loss: 5.635297]
    


![png](output_15_1455.png)


    72799 [Discriminator Loss: 0.202491, Acc.: 50.00%] [Generator Loss: 4.566397]
    


![png](output_15_1457.png)


    72899 [Discriminator Loss: 0.182731, Acc.: 50.00%] [Generator Loss: 5.855147]
    


![png](output_15_1459.png)


    72999 [Discriminator Loss: 0.227399, Acc.: 50.00%] [Generator Loss: 4.524060]
    


![png](output_15_1461.png)


    73099 [Discriminator Loss: 0.173836, Acc.: 50.00%] [Generator Loss: 5.319587]
    


![png](output_15_1463.png)


    73199 [Discriminator Loss: 0.181813, Acc.: 50.00%] [Generator Loss: 4.832836]
    


![png](output_15_1465.png)


    73299 [Discriminator Loss: 0.253412, Acc.: 50.00%] [Generator Loss: 4.982766]
    


![png](output_15_1467.png)


    73399 [Discriminator Loss: 0.195623, Acc.: 50.00%] [Generator Loss: 5.624368]
    


![png](output_15_1469.png)


    73499 [Discriminator Loss: 0.176952, Acc.: 50.00%] [Generator Loss: 5.292355]
    


![png](output_15_1471.png)


    73599 [Discriminator Loss: 0.192069, Acc.: 50.00%] [Generator Loss: 4.460250]
    


![png](output_15_1473.png)


    73699 [Discriminator Loss: 0.192851, Acc.: 50.00%] [Generator Loss: 5.330287]
    


![png](output_15_1475.png)


    73799 [Discriminator Loss: 0.176417, Acc.: 50.00%] [Generator Loss: 5.310887]
    


![png](output_15_1477.png)


    73899 [Discriminator Loss: 0.206063, Acc.: 50.00%] [Generator Loss: 5.797566]
    


![png](output_15_1479.png)


    73999 [Discriminator Loss: 0.206735, Acc.: 50.00%] [Generator Loss: 5.481078]
    


![png](output_15_1481.png)


    74099 [Discriminator Loss: 0.191702, Acc.: 50.00%] [Generator Loss: 5.016667]
    


![png](output_15_1483.png)


    74199 [Discriminator Loss: 0.193883, Acc.: 50.00%] [Generator Loss: 5.303905]
    


![png](output_15_1485.png)


    74299 [Discriminator Loss: 0.329542, Acc.: 50.00%] [Generator Loss: 4.775288]
    


![png](output_15_1487.png)


    74399 [Discriminator Loss: 0.264532, Acc.: 50.00%] [Generator Loss: 6.243183]
    


![png](output_15_1489.png)


    74499 [Discriminator Loss: 0.262400, Acc.: 50.00%] [Generator Loss: 4.459170]
    


![png](output_15_1491.png)


    74599 [Discriminator Loss: 0.184254, Acc.: 50.00%] [Generator Loss: 5.305845]
    


![png](output_15_1493.png)


    74699 [Discriminator Loss: 0.197634, Acc.: 50.00%] [Generator Loss: 5.059115]
    


![png](output_15_1495.png)


    74799 [Discriminator Loss: 0.190912, Acc.: 50.00%] [Generator Loss: 5.330902]
    


![png](output_15_1497.png)


    74899 [Discriminator Loss: 0.209972, Acc.: 50.00%] [Generator Loss: 5.083656]
    


![png](output_15_1499.png)


    74999 [Discriminator Loss: 0.268242, Acc.: 50.00%] [Generator Loss: 5.574942]
    


![png](output_15_1501.png)


    75099 [Discriminator Loss: 0.191298, Acc.: 50.00%] [Generator Loss: 5.257819]
    


![png](output_15_1503.png)


    75199 [Discriminator Loss: 0.263448, Acc.: 50.00%] [Generator Loss: 4.276958]
    


![png](output_15_1505.png)

