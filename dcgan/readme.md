

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
#指定数据所在的目录
PATH = os.path.abspath(os.path.join('data','players'))
IMGS = glob(os.path.join(PATH, "*.jpg"))

print(len(IMGS)) 
print(IMGS[:10]) 
```

    1732
    ['F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100063.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100350.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100360.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10037.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10038.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100459.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10046.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-100557.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10056.jpg', 'F:\\Kaggle\\generateSP\\data\\players\\static.weltsport.net-10060.jpg']
    


```python
WIDTH = 28
HEIGHT = 28
DEPTH = 3
```


```python
def procImages(images):
    processed_images = []
    
    # set depth
    depth = None
    if DEPTH == 1:
        depth = cv2.IMREAD_GRAYSCALE
    elif DEPTH == 3:
        depth = cv2.IMREAD_COLOR
    else:
        print('DEPTH must be set to 1 or to 3.')
        return None
    
    #resize images
    for img in images:
        base = os.path.basename(img)
        full_size_image = cv2.imread(img, depth)
        processed_images.append(cv2.resize(full_size_image, (WIDTH, HEIGHT), interpolation=cv2.INTER_CUBIC))
    processed_images = np.asarray(processed_images)
    
    # rescale images to [-1, 1]
    processed_images = np.divide(processed_images, 127.5) - 1

    return processed_images
```


```python
processed_images = procImages(IMGS[:30])
processed_images.shape
```




    (30, 28, 28, 3)




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


![png](out/output_5_0.png)



```python
# GAN parameters
LATENT_DIM = 100
G_LAYERS_DIM = [256, 512, 1024]
D_LAYERS_DIM = [1024, 512, 256]

BATCH_SIZE = 16
EPOCHS = 1000000
LR = 0.00002
BETA_1 = 0.5
```


```python
def buildGeneratorDC(img_shape):
    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=LATENT_DIM))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(DEPTH, kernel_size=3, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(LATENT_DIM,))
    img = model(noise)

    return Model(noise, img)
```


```python
def buildDiscriminatorDC(img_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dropout(0.25))
    model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    
    model.add(Dropout(0.25))
    model.add(Flatten())
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
#build the discriminator
dDC = buildDiscriminatorDC(processed_images.shape[1:])
dDC.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_19 (Conv2D)           (None, 14, 14, 32)        896       
    _________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)   (None, 14, 14, 32)        0         
    _________________________________________________________________
    dropout_13 (Dropout)         (None, 14, 14, 32)        0         
    _________________________________________________________________
    conv2d_20 (Conv2D)           (None, 7, 7, 64)          18496     
    _________________________________________________________________
    zero_padding2d_4 (ZeroPaddin (None, 8, 8, 64)          0         
    _________________________________________________________________
    batch_normalization_14 (Batc (None, 8, 8, 64)          256       
    _________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)   (None, 8, 8, 64)          0         
    _________________________________________________________________
    dropout_14 (Dropout)         (None, 8, 8, 64)          0         
    _________________________________________________________________
    conv2d_21 (Conv2D)           (None, 4, 4, 128)         73856     
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 4, 4, 128)         512       
    _________________________________________________________________
    leaky_re_lu_15 (LeakyReLU)   (None, 4, 4, 128)         0         
    _________________________________________________________________
    dropout_15 (Dropout)         (None, 4, 4, 128)         0         
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 4, 4, 256)         295168    
    _________________________________________________________________
    batch_normalization_16 (Batc (None, 4, 4, 256)         1024      
    _________________________________________________________________
    leaky_re_lu_16 (LeakyReLU)   (None, 4, 4, 256)         0         
    _________________________________________________________________
    dropout_16 (Dropout)         (None, 4, 4, 256)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 4096)              0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 1)                 4097      
    =================================================================
    Total params: 394,305
    Trainable params: 393,409
    Non-trainable params: 896
    _________________________________________________________________
    


```python
processed_images.shape[1:]
```




    (28, 28, 3)




```python
#build generator
gDC = buildGeneratorDC(processed_images.shape[1:])
gDC.compile(loss='binary_crossentropy', optimizer=optimizer)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_7 (Dense)              (None, 6272)              633472    
    _________________________________________________________________
    reshape_3 (Reshape)          (None, 7, 7, 128)         0         
    _________________________________________________________________
    up_sampling2d_5 (UpSampling2 (None, 14, 14, 128)       0         
    _________________________________________________________________
    conv2d_23 (Conv2D)           (None, 14, 14, 128)       147584    
    _________________________________________________________________
    batch_normalization_17 (Batc (None, 14, 14, 128)       512       
    _________________________________________________________________
    activation_7 (Activation)    (None, 14, 14, 128)       0         
    _________________________________________________________________
    up_sampling2d_6 (UpSampling2 (None, 28, 28, 128)       0         
    _________________________________________________________________
    conv2d_24 (Conv2D)           (None, 28, 28, 64)        73792     
    _________________________________________________________________
    batch_normalization_18 (Batc (None, 28, 28, 64)        256       
    _________________________________________________________________
    activation_8 (Activation)    (None, 28, 28, 64)        0         
    _________________________________________________________________
    conv2d_25 (Conv2D)           (None, 28, 28, 3)         1731      
    _________________________________________________________________
    activation_9 (Activation)    (None, 28, 28, 3)         0         
    =================================================================
    Total params: 857,347
    Trainable params: 856,963
    Non-trainable params: 384
    _________________________________________________________________
    


```python
#build combined model
cDC = buildCombined(gDC, dDC)
cDC.compile(loss='binary_crossentropy', optimizer=optimizer)
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_10 (InputLayer)        (None, 100)               0         
    _________________________________________________________________
    model_9 (Model)              (None, 28, 28, 3)         857347    
    _________________________________________________________________
    model_8 (Model)              (None, 1)                 394305    
    =================================================================
    Total params: 1,251,652
    Trainable params: 856,963
    Non-trainable params: 394,689
    _________________________________________________________________
    


```python
#training DC GAN
SAMPLE_INTERVAL = WARNING_INTERVAL = 100

YDis = np.zeros(2 * BATCH_SIZE)
YDis[:BATCH_SIZE] = .9 #Label smoothing

YGen = np.ones(BATCH_SIZE)

for epoch in range(EPOCHS):
    # get a batch of real images
    idx = np.random.randint(0, processed_images.shape[0], BATCH_SIZE)
    real_imgs = processed_images[idx]

    # generate a batch of fake images
    noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    fake_imgs = gDC.predict(noise)
    X = np.concatenate([real_imgs, fake_imgs])
    
    # Train discriminator
    dDC.trainable = True
    d_loss = dDC.train_on_batch(X, YDis)

    # Train the generator
    dDC.trainable = False
    #noise = np.random.normal(0, 1, (BATCH_SIZE, LATENT_DIM))
    g_loss = cDC.train_on_batch(noise, YGen)

    # Progress
    if (epoch+1) % WARNING_INTERVAL == 0 or epoch == 0:
        print ("%d [Discriminator Loss: %f, Acc.: %.2f%%] [Generator Loss: %f]" % (epoch, d_loss[0], 100. * d_loss[1], g_loss))

    # If at save interval => save generated image samples
    if (epoch+1) % SAMPLE_INTERVAL == 0 or epoch == 0:
        sampleImages(gDC)
```

    0 [Discriminator Loss: 0.844155, Acc.: 28.12%] [Generator Loss: 0.826777]
    


![png](out/output_16_1.png)


    99 [Discriminator Loss: 0.407898, Acc.: 46.88%] [Generator Loss: 1.572817]
    


![png](out/output_16_3.png)


    199 [Discriminator Loss: 0.392555, Acc.: 50.00%] [Generator Loss: 1.654232]
    


![png](out/output_16_5.png)


    299 [Discriminator Loss: 0.344601, Acc.: 50.00%] [Generator Loss: 1.435650]
    


![png](out/output_16_7.png)


    399 [Discriminator Loss: 0.215505, Acc.: 50.00%] [Generator Loss: 1.719332]
    


![png](out/output_16_9.png)


    499 [Discriminator Loss: 0.231912, Acc.: 46.88%] [Generator Loss: 1.743370]
    


![png](out/output_16_11.png)


    599 [Discriminator Loss: 0.223142, Acc.: 50.00%] [Generator Loss: 2.019858]
    


![png](out/output_16_13.png)


    699 [Discriminator Loss: 0.226920, Acc.: 50.00%] [Generator Loss: 1.785352]
    


![png](out/output_16_15.png)


    799 [Discriminator Loss: 0.218211, Acc.: 50.00%] [Generator Loss: 2.323272]
    


![png](out/output_16_17.png)


    899 [Discriminator Loss: 0.236664, Acc.: 46.88%] [Generator Loss: 1.860659]
    


![png](out/output_16_19.png)


    999 [Discriminator Loss: 0.193473, Acc.: 50.00%] [Generator Loss: 1.456210]
    


![png](out/output_16_21.png)


    1099 [Discriminator Loss: 0.205847, Acc.: 50.00%] [Generator Loss: 1.928674]
    


![png](out/output_16_23.png)


    1199 [Discriminator Loss: 0.205343, Acc.: 50.00%] [Generator Loss: 1.844219]
    


![png](out/output_16_25.png)


    1299 [Discriminator Loss: 0.181888, Acc.: 50.00%] [Generator Loss: 1.615909]
    


![png](out/output_16_27.png)


    1399 [Discriminator Loss: 0.191934, Acc.: 50.00%] [Generator Loss: 2.082539]
    


![png](out/output_16_29.png)


    1499 [Discriminator Loss: 0.187280, Acc.: 50.00%] [Generator Loss: 2.139929]
    


![png](out/output_16_31.png)


    1599 [Discriminator Loss: 0.180116, Acc.: 50.00%] [Generator Loss: 2.216020]
    


![png](out/output_16_33.png)


    1699 [Discriminator Loss: 0.197188, Acc.: 50.00%] [Generator Loss: 2.016933]
    


![png](out/output_16_35.png)


    1799 [Discriminator Loss: 0.229933, Acc.: 50.00%] [Generator Loss: 2.288685]
    


![png](out/output_16_37.png)


    1899 [Discriminator Loss: 0.180052, Acc.: 50.00%] [Generator Loss: 2.468081]
    


![png](out/output_16_39.png)


    1999 [Discriminator Loss: 0.185192, Acc.: 50.00%] [Generator Loss: 2.332157]
    


![png](out/output_16_41.png)


    2099 [Discriminator Loss: 0.177470, Acc.: 50.00%] [Generator Loss: 2.370375]
    


![png](out/output_16_43.png)


    2199 [Discriminator Loss: 0.179667, Acc.: 50.00%] [Generator Loss: 2.422594]
    


![png](out/output_16_45.png)


    2299 [Discriminator Loss: 0.201787, Acc.: 50.00%] [Generator Loss: 2.448705]
    


![png](out/output_16_47.png)


    2399 [Discriminator Loss: 0.178702, Acc.: 50.00%] [Generator Loss: 2.417077]
    


![png](out/output_16_49.png)


    2499 [Discriminator Loss: 0.187985, Acc.: 50.00%] [Generator Loss: 2.542654]
    


![png](out/output_16_51.png)


    2599 [Discriminator Loss: 0.188354, Acc.: 50.00%] [Generator Loss: 2.114798]
    


![png](out/output_16_53.png)


    2699 [Discriminator Loss: 0.174401, Acc.: 50.00%] [Generator Loss: 2.417632]
    


![png](out/output_16_55.png)


    2799 [Discriminator Loss: 0.181192, Acc.: 50.00%] [Generator Loss: 2.718800]
    


![png](out/output_16_57.png)


    2899 [Discriminator Loss: 0.181175, Acc.: 50.00%] [Generator Loss: 2.601932]
    


![png](out/output_16_59.png)


    2999 [Discriminator Loss: 0.190987, Acc.: 50.00%] [Generator Loss: 2.671049]
    


![png](out/output_16_61.png)


    3099 [Discriminator Loss: 0.172865, Acc.: 50.00%] [Generator Loss: 2.441411]
    


![png](out/output_16_63.png)


    3199 [Discriminator Loss: 0.171602, Acc.: 50.00%] [Generator Loss: 2.745640]
    


![png](out/output_16_65.png)


    3299 [Discriminator Loss: 0.174345, Acc.: 50.00%] [Generator Loss: 2.708027]
    


![png](out/output_16_67.png)


    3399 [Discriminator Loss: 0.168419, Acc.: 50.00%] [Generator Loss: 2.791888]
    


![png](out/output_16_69.png)


    3499 [Discriminator Loss: 0.170034, Acc.: 50.00%] [Generator Loss: 2.784906]
    


![png](out/output_16_71.png)


    3599 [Discriminator Loss: 0.171397, Acc.: 50.00%] [Generator Loss: 2.666945]
    


![png](out/output_16_73.png)


    3699 [Discriminator Loss: 0.174490, Acc.: 50.00%] [Generator Loss: 2.754630]
    


![png](out/output_16_75.png)


    3799 [Discriminator Loss: 0.174342, Acc.: 50.00%] [Generator Loss: 2.839196]
    


![png](out/output_16_77.png)


    3899 [Discriminator Loss: 0.175982, Acc.: 50.00%] [Generator Loss: 2.721471]
    


![png](out/output_16_79.png)


    3999 [Discriminator Loss: 0.183088, Acc.: 50.00%] [Generator Loss: 2.871280]
    


![png](out/output_16_81.png)


    4099 [Discriminator Loss: 0.168094, Acc.: 50.00%] [Generator Loss: 3.067040]
    


![png](out/output_16_83.png)


    4199 [Discriminator Loss: 0.169364, Acc.: 50.00%] [Generator Loss: 2.678401]
    


![png](out/output_16_85.png)


    4299 [Discriminator Loss: 0.167602, Acc.: 50.00%] [Generator Loss: 2.969089]
    


![png](out/output_16_87.png)


    4399 [Discriminator Loss: 0.170394, Acc.: 50.00%] [Generator Loss: 3.229683]
    


![png](out/output_16_89.png)


    4499 [Discriminator Loss: 0.182281, Acc.: 50.00%] [Generator Loss: 2.974152]
    


![png](out/output_16_91.png)


    4599 [Discriminator Loss: 0.171698, Acc.: 50.00%] [Generator Loss: 2.962282]
    


![png](out/output_16_93.png)


    4699 [Discriminator Loss: 0.170160, Acc.: 50.00%] [Generator Loss: 3.357580]
    


![png](out/output_16_95.png)


    4799 [Discriminator Loss: 0.171497, Acc.: 50.00%] [Generator Loss: 3.113940]
    


![png](out/output_16_97.png)


    4899 [Discriminator Loss: 0.172531, Acc.: 50.00%] [Generator Loss: 2.787012]
    


![png](out/output_16_99.png)


    4999 [Discriminator Loss: 0.165555, Acc.: 50.00%] [Generator Loss: 3.126535]
    


![png](out/output_16_101.png)


    5099 [Discriminator Loss: 0.169283, Acc.: 50.00%] [Generator Loss: 2.676387]
    


![png](out/output_16_103.png)


    5199 [Discriminator Loss: 0.168535, Acc.: 50.00%] [Generator Loss: 2.945904]
    


![png](out/output_16_105.png)


    5299 [Discriminator Loss: 0.166396, Acc.: 50.00%] [Generator Loss: 3.134063]
    


![png](out/output_16_107.png)


    5399 [Discriminator Loss: 0.175315, Acc.: 50.00%] [Generator Loss: 3.076426]
    


![png](out/output_16_109.png)


    5499 [Discriminator Loss: 0.164203, Acc.: 50.00%] [Generator Loss: 3.000234]
    


![png](out/output_16_111.png)


    5599 [Discriminator Loss: 0.169932, Acc.: 50.00%] [Generator Loss: 3.065523]
    


![png](out/output_16_113.png)


    5699 [Discriminator Loss: 0.166291, Acc.: 50.00%] [Generator Loss: 3.180311]
    


![png](out/output_16_115.png)


    5799 [Discriminator Loss: 0.166865, Acc.: 50.00%] [Generator Loss: 3.216035]
    


![png](out/output_16_117.png)


    5899 [Discriminator Loss: 0.166161, Acc.: 50.00%] [Generator Loss: 3.465611]
    


![png](out/output_16_119.png)


    5999 [Discriminator Loss: 0.166497, Acc.: 50.00%] [Generator Loss: 3.353391]
    


![png](out/output_16_121.png)


    6099 [Discriminator Loss: 0.166010, Acc.: 50.00%] [Generator Loss: 3.530622]
    


![png](out/output_16_123.png)


    6199 [Discriminator Loss: 0.166607, Acc.: 50.00%] [Generator Loss: 3.807276]
    


![png](out/output_16_125.png)


    6299 [Discriminator Loss: 0.164617, Acc.: 50.00%] [Generator Loss: 3.783701]
    


![png](out/output_16_127.png)


    6399 [Discriminator Loss: 0.166444, Acc.: 50.00%] [Generator Loss: 3.965499]
    


![png](out/output_16_129.png)


    6499 [Discriminator Loss: 0.166291, Acc.: 50.00%] [Generator Loss: 3.227392]
    


![png](out/output_16_131.png)


    6599 [Discriminator Loss: 0.166888, Acc.: 50.00%] [Generator Loss: 3.232172]
    


![png](out/output_16_133.png)


    6699 [Discriminator Loss: 0.167127, Acc.: 50.00%] [Generator Loss: 3.417498]
    


![png](out/output_16_135.png)


    6799 [Discriminator Loss: 0.164165, Acc.: 50.00%] [Generator Loss: 3.653866]
    


![png](out/output_16_137.png)


    6899 [Discriminator Loss: 0.164310, Acc.: 50.00%] [Generator Loss: 3.393194]
    


![png](out/output_16_139.png)


    6999 [Discriminator Loss: 0.170011, Acc.: 50.00%] [Generator Loss: 3.759209]
    


![png](out/output_16_141.png)


    7099 [Discriminator Loss: 0.165111, Acc.: 50.00%] [Generator Loss: 3.561116]
    


![png](out/output_16_143.png)


    7199 [Discriminator Loss: 0.165492, Acc.: 50.00%] [Generator Loss: 3.787984]
    


![png](out/output_16_145.png)


    7299 [Discriminator Loss: 0.168918, Acc.: 50.00%] [Generator Loss: 3.800504]
    


![png](out/output_16_147.png)


    7399 [Discriminator Loss: 0.164390, Acc.: 50.00%] [Generator Loss: 3.692609]
    


![png](out/output_16_149.png)


    7499 [Discriminator Loss: 0.164769, Acc.: 50.00%] [Generator Loss: 3.894603]
    


![png](out/output_16_151.png)


    7599 [Discriminator Loss: 0.166966, Acc.: 50.00%] [Generator Loss: 3.472956]
    


![png](out/output_16_153.png)


    7699 [Discriminator Loss: 0.167576, Acc.: 50.00%] [Generator Loss: 4.231601]
    


![png](out/output_16_155.png)


    7799 [Discriminator Loss: 0.165259, Acc.: 50.00%] [Generator Loss: 4.020025]
    


![png](out/output_16_157.png)


    7899 [Discriminator Loss: 0.163634, Acc.: 50.00%] [Generator Loss: 3.904343]
    


![png](out/output_16_159.png)


    7999 [Discriminator Loss: 0.164920, Acc.: 50.00%] [Generator Loss: 4.121094]
    


![png](out/output_16_161.png)


    8099 [Discriminator Loss: 0.164082, Acc.: 50.00%] [Generator Loss: 3.963915]
    


![png](out/output_16_163.png)


    8199 [Discriminator Loss: 0.164858, Acc.: 50.00%] [Generator Loss: 3.904504]
    


![png](out/output_16_165.png)


    8299 [Discriminator Loss: 0.165369, Acc.: 50.00%] [Generator Loss: 4.264998]
    


![png](out/output_16_167.png)


    8399 [Discriminator Loss: 0.167401, Acc.: 50.00%] [Generator Loss: 4.107815]
    


![png](out/output_16_169.png)


    8499 [Discriminator Loss: 0.168767, Acc.: 50.00%] [Generator Loss: 3.872203]
    


![png](out/output_16_171.png)


    8599 [Discriminator Loss: 0.163559, Acc.: 50.00%] [Generator Loss: 4.160940]
    


![png](out/output_16_173.png)


    8699 [Discriminator Loss: 0.165668, Acc.: 50.00%] [Generator Loss: 4.356079]
    


![png](out/output_16_175.png)


    8799 [Discriminator Loss: 0.164344, Acc.: 50.00%] [Generator Loss: 4.497719]
    


![png](out/output_16_177.png)


    8899 [Discriminator Loss: 0.170619, Acc.: 50.00%] [Generator Loss: 4.361531]
    


![png](out/output_16_179.png)


    8999 [Discriminator Loss: 0.164063, Acc.: 50.00%] [Generator Loss: 4.303911]
    


![png](out/output_16_181.png)


    9099 [Discriminator Loss: 0.164115, Acc.: 50.00%] [Generator Loss: 4.149350]
    


![png](out/output_16_183.png)


    9199 [Discriminator Loss: 0.164378, Acc.: 50.00%] [Generator Loss: 3.895847]
    


![png](out/output_16_185.png)


    9299 [Discriminator Loss: 0.164353, Acc.: 50.00%] [Generator Loss: 4.538137]
    


![png](out/output_16_187.png)


    9399 [Discriminator Loss: 0.165014, Acc.: 50.00%] [Generator Loss: 4.222649]
    


![png](out/output_16_189.png)


    9499 [Discriminator Loss: 0.163168, Acc.: 50.00%] [Generator Loss: 4.554295]
    


![png](out/output_16_191.png)


    9599 [Discriminator Loss: 0.165034, Acc.: 50.00%] [Generator Loss: 4.167037]
    


![png](out/output_16_193.png)


    9699 [Discriminator Loss: 0.165121, Acc.: 50.00%] [Generator Loss: 4.607861]
    


![png](out/output_16_195.png)


    9799 [Discriminator Loss: 0.163719, Acc.: 50.00%] [Generator Loss: 4.456860]
    


![png](out/output_16_197.png)


    9899 [Discriminator Loss: 0.163881, Acc.: 50.00%] [Generator Loss: 4.363617]
    


![png](out/output_16_199.png)


    9999 [Discriminator Loss: 0.164095, Acc.: 50.00%] [Generator Loss: 4.322974]
    


![png](out/output_16_201.png)


    10099 [Discriminator Loss: 0.163847, Acc.: 50.00%] [Generator Loss: 4.614496]
    


![png](out/output_16_203.png)


    10199 [Discriminator Loss: 0.163551, Acc.: 50.00%] [Generator Loss: 4.432217]
    


![png](out/output_16_205.png)


    10299 [Discriminator Loss: 0.163839, Acc.: 50.00%] [Generator Loss: 4.667894]
    


![png](out/output_16_207.png)


    10399 [Discriminator Loss: 0.163856, Acc.: 50.00%] [Generator Loss: 4.419616]
    


![png](out/output_16_209.png)


    10499 [Discriminator Loss: 0.163711, Acc.: 50.00%] [Generator Loss: 4.793807]
    


![png](out/output_16_211.png)


    10599 [Discriminator Loss: 0.163349, Acc.: 50.00%] [Generator Loss: 4.823689]
    


![png](out/output_16_213.png)


    10699 [Discriminator Loss: 0.163946, Acc.: 50.00%] [Generator Loss: 5.013738]
    


![png](out/output_16_215.png)


    10799 [Discriminator Loss: 0.162983, Acc.: 50.00%] [Generator Loss: 5.233280]
    


![png](out/output_16_217.png)


    10899 [Discriminator Loss: 0.163450, Acc.: 50.00%] [Generator Loss: 4.805761]
    


![png](out/output_16_219.png)


    10999 [Discriminator Loss: 0.163911, Acc.: 50.00%] [Generator Loss: 5.360293]
    


![png](out/output_16_221.png)


    11099 [Discriminator Loss: 0.163568, Acc.: 50.00%] [Generator Loss: 4.837401]
    


![png](out/output_16_223.png)


    11199 [Discriminator Loss: 0.163613, Acc.: 50.00%] [Generator Loss: 5.202855]
    


![png](out/output_16_225.png)


    11299 [Discriminator Loss: 0.163489, Acc.: 50.00%] [Generator Loss: 5.088096]
    


![png](out/output_16_227.png)


    11399 [Discriminator Loss: 0.163211, Acc.: 50.00%] [Generator Loss: 5.054580]
    


![png](out/output_16_229.png)


    11499 [Discriminator Loss: 0.163542, Acc.: 50.00%] [Generator Loss: 5.489645]
    


![png](out/output_16_231.png)


    11599 [Discriminator Loss: 0.163629, Acc.: 50.00%] [Generator Loss: 5.174011]
    


![png](out/output_16_233.png)


    11699 [Discriminator Loss: 0.163200, Acc.: 50.00%] [Generator Loss: 5.304090]
    


![png](out/output_16_235.png)


    11799 [Discriminator Loss: 0.163497, Acc.: 50.00%] [Generator Loss: 6.036710]
    


![png](out/output_16_237.png)


    11899 [Discriminator Loss: 0.164480, Acc.: 50.00%] [Generator Loss: 4.868309]
    


![png](out/output_16_239.png)


    11999 [Discriminator Loss: 0.166022, Acc.: 50.00%] [Generator Loss: 5.446295]
    


![png](out/output_16_241.png)


    12099 [Discriminator Loss: 0.166419, Acc.: 50.00%] [Generator Loss: 5.187495]
    


![png](out/output_16_243.png)


    12199 [Discriminator Loss: 0.166046, Acc.: 50.00%] [Generator Loss: 6.349765]
    


![png](out/output_16_245.png)


    12299 [Discriminator Loss: 0.166465, Acc.: 50.00%] [Generator Loss: 5.353168]
    


![png](out/output_16_247.png)


    12399 [Discriminator Loss: 0.164429, Acc.: 50.00%] [Generator Loss: 5.094367]
    


![png](out/output_16_249.png)


    12499 [Discriminator Loss: 0.167961, Acc.: 50.00%] [Generator Loss: 4.526375]
    


![png](out/output_16_251.png)


    12599 [Discriminator Loss: 0.165415, Acc.: 50.00%] [Generator Loss: 5.303524]
    


![png](out/output_16_253.png)


    12699 [Discriminator Loss: 0.165136, Acc.: 50.00%] [Generator Loss: 4.277478]
    


![png](out/output_16_255.png)


    12799 [Discriminator Loss: 0.163793, Acc.: 50.00%] [Generator Loss: 4.249160]
    


![png](out/output_16_257.png)


    12899 [Discriminator Loss: 0.164696, Acc.: 50.00%] [Generator Loss: 3.460105]
    


![png](out/output_16_259.png)


    12999 [Discriminator Loss: 0.163742, Acc.: 50.00%] [Generator Loss: 4.206430]
    


![png](out/output_16_261.png)


    13099 [Discriminator Loss: 0.163903, Acc.: 50.00%] [Generator Loss: 5.397884]
    


![png](out/output_16_263.png)


    13199 [Discriminator Loss: 0.163445, Acc.: 50.00%] [Generator Loss: 4.717357]
    


![png](out/output_16_265.png)


    13299 [Discriminator Loss: 0.163475, Acc.: 50.00%] [Generator Loss: 4.639860]
    


![png](out/output_16_267.png)


    13399 [Discriminator Loss: 0.165054, Acc.: 50.00%] [Generator Loss: 6.719573]
    


![png](out/output_16_269.png)


    13499 [Discriminator Loss: 0.163514, Acc.: 50.00%] [Generator Loss: 3.196826]
    


![png](out/output_16_271.png)


    13599 [Discriminator Loss: 0.162887, Acc.: 50.00%] [Generator Loss: 4.421573]
    


![png](out/output_16_273.png)


    13699 [Discriminator Loss: 0.163997, Acc.: 50.00%] [Generator Loss: 3.764066]
    


![png](out/output_16_275.png)


    13799 [Discriminator Loss: 0.163524, Acc.: 50.00%] [Generator Loss: 4.129004]
    


![png](out/output_16_277.png)


    13899 [Discriminator Loss: 0.162997, Acc.: 50.00%] [Generator Loss: 4.564364]
    


![png](out/output_16_279.png)


    13999 [Discriminator Loss: 0.163428, Acc.: 50.00%] [Generator Loss: 5.587677]
    


![png](out/output_16_281.png)


    14099 [Discriminator Loss: 0.163991, Acc.: 50.00%] [Generator Loss: 5.980223]
    


![png](out/output_16_283.png)


    14199 [Discriminator Loss: 0.163300, Acc.: 50.00%] [Generator Loss: 4.467700]
    


![png](out/output_16_285.png)


    14299 [Discriminator Loss: 0.163456, Acc.: 50.00%] [Generator Loss: 3.979326]
    


![png](out/output_16_287.png)


    14399 [Discriminator Loss: 0.162935, Acc.: 50.00%] [Generator Loss: 4.273056]
    


![png](out/output_16_289.png)


    14499 [Discriminator Loss: 0.163044, Acc.: 50.00%] [Generator Loss: 3.399909]
    


![png](out/output_16_291.png)


    14599 [Discriminator Loss: 0.163223, Acc.: 50.00%] [Generator Loss: 4.322097]
    


![png](out/output_16_293.png)


    14699 [Discriminator Loss: 0.164062, Acc.: 50.00%] [Generator Loss: 6.130485]
    


![png](out/output_16_295.png)


    14799 [Discriminator Loss: 0.163480, Acc.: 50.00%] [Generator Loss: 3.850237]
    


![png](out/output_16_297.png)


    14899 [Discriminator Loss: 0.162894, Acc.: 50.00%] [Generator Loss: 3.873600]
    


![png](out/output_16_299.png)


    14999 [Discriminator Loss: 0.163241, Acc.: 50.00%] [Generator Loss: 4.483905]
    


![png](out/output_16_301.png)


    15099 [Discriminator Loss: 0.163139, Acc.: 50.00%] [Generator Loss: 4.702290]
    


![png](out/output_16_303.png)


    15199 [Discriminator Loss: 0.163091, Acc.: 50.00%] [Generator Loss: 3.819625]
    


![png](out/output_16_305.png)


    15299 [Discriminator Loss: 0.163173, Acc.: 50.00%] [Generator Loss: 3.879886]
    


![png](out/output_16_307.png)


    15399 [Discriminator Loss: 0.163371, Acc.: 50.00%] [Generator Loss: 3.226943]
    


![png](out/output_16_309.png)


    15499 [Discriminator Loss: 0.163197, Acc.: 50.00%] [Generator Loss: 2.831904]
    


![png](out/output_16_311.png)


    15599 [Discriminator Loss: 0.164184, Acc.: 50.00%] [Generator Loss: 4.674219]
    


![png](out/output_16_313.png)


    15699 [Discriminator Loss: 0.163540, Acc.: 50.00%] [Generator Loss: 3.034998]
    


![png](out/output_16_315.png)


    15799 [Discriminator Loss: 0.165972, Acc.: 50.00%] [Generator Loss: 3.864181]
    


![png](out/output_16_317.png)


    15899 [Discriminator Loss: 0.164222, Acc.: 50.00%] [Generator Loss: 4.686877]
    


![png](out/output_16_319.png)


    15999 [Discriminator Loss: 0.163951, Acc.: 50.00%] [Generator Loss: 5.080579]
    


![png](out/output_16_321.png)


    16099 [Discriminator Loss: 0.163321, Acc.: 50.00%] [Generator Loss: 4.400635]
    


![png](out/output_16_323.png)


    16199 [Discriminator Loss: 0.163969, Acc.: 50.00%] [Generator Loss: 4.134634]
    


![png](out/output_16_325.png)


    16299 [Discriminator Loss: 0.164070, Acc.: 50.00%] [Generator Loss: 3.630612]
    


![png](out/output_16_327.png)


    16399 [Discriminator Loss: 0.165311, Acc.: 50.00%] [Generator Loss: 2.884573]
    


![png](out/output_16_329.png)


    16499 [Discriminator Loss: 0.175783, Acc.: 50.00%] [Generator Loss: 5.858651]
    


![png](out/output_16_331.png)


    16599 [Discriminator Loss: 0.163626, Acc.: 50.00%] [Generator Loss: 3.700930]
    


![png](out/output_16_333.png)


    16699 [Discriminator Loss: 0.163501, Acc.: 50.00%] [Generator Loss: 3.463303]
    


![png](out/output_16_335.png)


    16799 [Discriminator Loss: 0.165142, Acc.: 50.00%] [Generator Loss: 3.553228]
    


![png](out/output_16_337.png)


    16899 [Discriminator Loss: 0.163814, Acc.: 50.00%] [Generator Loss: 5.047354]
    


![png](out/output_16_339.png)


    16999 [Discriminator Loss: 0.167221, Acc.: 50.00%] [Generator Loss: 4.486086]
    


![png](out/output_16_341.png)


    17099 [Discriminator Loss: 0.164202, Acc.: 50.00%] [Generator Loss: 4.734048]
    


![png](out/output_16_343.png)


    17199 [Discriminator Loss: 0.170863, Acc.: 50.00%] [Generator Loss: 3.785838]
    


![png](out/output_16_345.png)


    17299 [Discriminator Loss: 0.164825, Acc.: 50.00%] [Generator Loss: 3.914680]
    


![png](out/output_16_347.png)


    17399 [Discriminator Loss: 0.164102, Acc.: 50.00%] [Generator Loss: 4.711103]
    


![png](out/output_16_349.png)


    17499 [Discriminator Loss: 0.164806, Acc.: 50.00%] [Generator Loss: 4.788031]
    


![png](out/output_16_351.png)


    17599 [Discriminator Loss: 0.172254, Acc.: 50.00%] [Generator Loss: 4.797960]
    


![png](out/output_16_353.png)


    17699 [Discriminator Loss: 0.166445, Acc.: 50.00%] [Generator Loss: 4.837053]
    


![png](out/output_16_355.png)


    17799 [Discriminator Loss: 0.168085, Acc.: 50.00%] [Generator Loss: 3.923819]
    


![png](out/output_16_357.png)


    17899 [Discriminator Loss: 0.175679, Acc.: 50.00%] [Generator Loss: 5.441113]
    


![png](out/output_16_359.png)


    17999 [Discriminator Loss: 0.174659, Acc.: 50.00%] [Generator Loss: 4.982611]
    


![png](out/output_16_361.png)


    18099 [Discriminator Loss: 0.173796, Acc.: 50.00%] [Generator Loss: 4.197525]
    


![png](out/output_16_363.png)


    18199 [Discriminator Loss: 0.177296, Acc.: 50.00%] [Generator Loss: 5.055236]
    


![png](out/output_16_365.png)


    18299 [Discriminator Loss: 0.191721, Acc.: 50.00%] [Generator Loss: 5.725544]
    


![png](out/output_16_367.png)


    18399 [Discriminator Loss: 0.212938, Acc.: 50.00%] [Generator Loss: 4.060614]
    


![png](out/output_16_369.png)


    18499 [Discriminator Loss: 0.185095, Acc.: 50.00%] [Generator Loss: 3.913157]
    


![png](out/output_16_371.png)


    18599 [Discriminator Loss: 0.175867, Acc.: 50.00%] [Generator Loss: 4.161339]
    


![png](out/output_16_373.png)


    18699 [Discriminator Loss: 0.165617, Acc.: 50.00%] [Generator Loss: 4.416111]
    


![png](out/output_16_375.png)


    18799 [Discriminator Loss: 0.171632, Acc.: 50.00%] [Generator Loss: 3.485172]
    


![png](out/output_16_377.png)


    18899 [Discriminator Loss: 0.206316, Acc.: 50.00%] [Generator Loss: 3.000452]
    


![png](out/output_16_379.png)


    18999 [Discriminator Loss: 0.183648, Acc.: 50.00%] [Generator Loss: 3.069248]
    


![png](out/output_16_381.png)


    19099 [Discriminator Loss: 0.194006, Acc.: 50.00%] [Generator Loss: 2.839571]
    


![png](out/output_16_383.png)


    19199 [Discriminator Loss: 0.190284, Acc.: 50.00%] [Generator Loss: 3.145810]
    


![png](out/output_16_385.png)


    19299 [Discriminator Loss: 0.181723, Acc.: 50.00%] [Generator Loss: 3.530956]
    


![png](out/output_16_387.png)


    19399 [Discriminator Loss: 0.185315, Acc.: 50.00%] [Generator Loss: 3.531356]
    


![png](out/output_16_389.png)


    19499 [Discriminator Loss: 0.173950, Acc.: 50.00%] [Generator Loss: 3.083139]
    


![png](out/output_16_391.png)


    19599 [Discriminator Loss: 0.183985, Acc.: 50.00%] [Generator Loss: 2.954408]
    


![png](out/output_16_393.png)


    19699 [Discriminator Loss: 0.177703, Acc.: 50.00%] [Generator Loss: 3.287370]
    


![png](out/output_16_395.png)


    19799 [Discriminator Loss: 0.194013, Acc.: 50.00%] [Generator Loss: 3.117704]
    


![png](out/output_16_397.png)


    19899 [Discriminator Loss: 0.180540, Acc.: 50.00%] [Generator Loss: 3.479836]
    


![png](out/output_16_399.png)


    19999 [Discriminator Loss: 0.179108, Acc.: 50.00%] [Generator Loss: 3.104191]
    


![png](out/output_16_401.png)


    20099 [Discriminator Loss: 0.180282, Acc.: 50.00%] [Generator Loss: 3.252121]
    


![png](out/output_16_403.png)


    20199 [Discriminator Loss: 0.181182, Acc.: 50.00%] [Generator Loss: 3.987833]
    


![png](out/output_16_405.png)


    20299 [Discriminator Loss: 0.169096, Acc.: 50.00%] [Generator Loss: 3.563093]
    


![png](out/output_16_407.png)


    20399 [Discriminator Loss: 0.179558, Acc.: 50.00%] [Generator Loss: 4.130210]
    


![png](out/output_16_409.png)


    20499 [Discriminator Loss: 0.181644, Acc.: 50.00%] [Generator Loss: 4.097416]
    


![png](out/output_16_411.png)


    20599 [Discriminator Loss: 0.177457, Acc.: 50.00%] [Generator Loss: 4.035939]
    


![png](out/output_16_413.png)


    20699 [Discriminator Loss: 0.180489, Acc.: 50.00%] [Generator Loss: 3.010058]
    


![png](out/output_16_415.png)


    20799 [Discriminator Loss: 0.188143, Acc.: 50.00%] [Generator Loss: 3.899626]
    


![png](out/output_16_417.png)


    20899 [Discriminator Loss: 0.173459, Acc.: 50.00%] [Generator Loss: 3.234814]
    


![png](out/output_16_419.png)


    20999 [Discriminator Loss: 0.204743, Acc.: 50.00%] [Generator Loss: 3.007739]
    


![png](out/output_16_421.png)


    21099 [Discriminator Loss: 0.248684, Acc.: 50.00%] [Generator Loss: 3.457678]
    


![png](out/output_16_423.png)


    21199 [Discriminator Loss: 0.190630, Acc.: 50.00%] [Generator Loss: 3.183964]
    


![png](out/output_16_425.png)


    21299 [Discriminator Loss: 0.197533, Acc.: 50.00%] [Generator Loss: 2.974371]
    


![png](out/output_16_427.png)


    21399 [Discriminator Loss: 0.206756, Acc.: 50.00%] [Generator Loss: 3.086124]
    


![png](out/output_16_429.png)


    21499 [Discriminator Loss: 0.187478, Acc.: 50.00%] [Generator Loss: 3.378042]
    


![png](out/output_16_431.png)


    21599 [Discriminator Loss: 0.218492, Acc.: 50.00%] [Generator Loss: 3.259659]
    


![png](out/output_16_433.png)


    21699 [Discriminator Loss: 0.181754, Acc.: 50.00%] [Generator Loss: 3.323876]
    


![png](out/output_16_435.png)


    21799 [Discriminator Loss: 0.175912, Acc.: 50.00%] [Generator Loss: 3.225142]
    


![png](out/output_16_437.png)


    21899 [Discriminator Loss: 0.175055, Acc.: 50.00%] [Generator Loss: 3.307461]
    


![png](out/output_16_439.png)


    21999 [Discriminator Loss: 0.275602, Acc.: 50.00%] [Generator Loss: 3.431754]
    


![png](out/output_16_441.png)


    22099 [Discriminator Loss: 0.189198, Acc.: 50.00%] [Generator Loss: 3.024430]
    


![png](out/output_16_443.png)


    22199 [Discriminator Loss: 0.175268, Acc.: 50.00%] [Generator Loss: 3.419267]
    


![png](out/output_16_445.png)


    22299 [Discriminator Loss: 0.180851, Acc.: 50.00%] [Generator Loss: 3.656663]
    


![png](out/output_16_447.png)


    22399 [Discriminator Loss: 0.186721, Acc.: 50.00%] [Generator Loss: 3.325585]
    


![png](out/output_16_449.png)


    22499 [Discriminator Loss: 0.275564, Acc.: 50.00%] [Generator Loss: 3.414080]
    


![png](out/output_16_451.png)


    22599 [Discriminator Loss: 0.201933, Acc.: 50.00%] [Generator Loss: 3.322978]
    


![png](out/output_16_453.png)


    22699 [Discriminator Loss: 0.192896, Acc.: 50.00%] [Generator Loss: 3.296208]
    


![png](out/output_16_455.png)


    22799 [Discriminator Loss: 0.172153, Acc.: 50.00%] [Generator Loss: 3.332891]
    


![png](out/output_16_457.png)


    22899 [Discriminator Loss: 0.174940, Acc.: 50.00%] [Generator Loss: 3.819083]
    


![png](out/output_16_459.png)


    22999 [Discriminator Loss: 0.185824, Acc.: 50.00%] [Generator Loss: 3.017099]
    


![png](out/output_16_461.png)


    23099 [Discriminator Loss: 0.196248, Acc.: 50.00%] [Generator Loss: 3.654293]
    


![png](out/output_16_463.png)


    23199 [Discriminator Loss: 0.174691, Acc.: 50.00%] [Generator Loss: 3.068156]
    


![png](out/output_16_465.png)


    23299 [Discriminator Loss: 0.172277, Acc.: 50.00%] [Generator Loss: 2.863069]
    


![png](out/output_16_467.png)


    23399 [Discriminator Loss: 0.173163, Acc.: 50.00%] [Generator Loss: 3.362727]
    


![png](out/output_16_469.png)


    23499 [Discriminator Loss: 0.170800, Acc.: 50.00%] [Generator Loss: 3.416552]
    


![png](out/output_16_471.png)


    23599 [Discriminator Loss: 0.190734, Acc.: 50.00%] [Generator Loss: 3.703921]
    


![png](out/output_16_473.png)


    23699 [Discriminator Loss: 0.174262, Acc.: 50.00%] [Generator Loss: 2.951439]
    


![png](out/output_16_475.png)


    23799 [Discriminator Loss: 0.174097, Acc.: 50.00%] [Generator Loss: 3.556358]
    


![png](out/output_16_477.png)


    23899 [Discriminator Loss: 0.178382, Acc.: 50.00%] [Generator Loss: 2.811726]
    


![png](out/output_16_479.png)


    23999 [Discriminator Loss: 0.184790, Acc.: 50.00%] [Generator Loss: 3.349203]
    


![png](out/output_16_481.png)


    24099 [Discriminator Loss: 0.192419, Acc.: 50.00%] [Generator Loss: 2.999490]
    


![png](out/output_16_483.png)


    24199 [Discriminator Loss: 0.174995, Acc.: 50.00%] [Generator Loss: 3.054445]
    


![png](out/output_16_485.png)


    24299 [Discriminator Loss: 0.198403, Acc.: 50.00%] [Generator Loss: 3.318672]
    


![png](out/output_16_487.png)


    24399 [Discriminator Loss: 0.178071, Acc.: 50.00%] [Generator Loss: 3.556854]
    


![png](out/output_16_489.png)


    24499 [Discriminator Loss: 0.168913, Acc.: 50.00%] [Generator Loss: 3.148281]
    


![png](out/output_16_491.png)


    24599 [Discriminator Loss: 0.180978, Acc.: 50.00%] [Generator Loss: 3.430113]
    


![png](out/output_16_493.png)


    24699 [Discriminator Loss: 0.195998, Acc.: 50.00%] [Generator Loss: 3.377835]
    


![png](out/output_16_495.png)


    24799 [Discriminator Loss: 0.191919, Acc.: 50.00%] [Generator Loss: 3.200593]
    


![png](out/output_16_497.png)


    24899 [Discriminator Loss: 0.174799, Acc.: 50.00%] [Generator Loss: 2.710456]
    


![png](out/output_16_499.png)


    24999 [Discriminator Loss: 0.211071, Acc.: 50.00%] [Generator Loss: 3.687290]
    


![png](out/output_16_501.png)


    25099 [Discriminator Loss: 0.191421, Acc.: 50.00%] [Generator Loss: 3.671376]
    


![png](out/output_16_503.png)


    25199 [Discriminator Loss: 0.207824, Acc.: 50.00%] [Generator Loss: 2.658327]
    


![png](out/output_16_505.png)


    25299 [Discriminator Loss: 0.179031, Acc.: 50.00%] [Generator Loss: 3.276677]
    


![png](out/output_16_507.png)


    25399 [Discriminator Loss: 0.179298, Acc.: 50.00%] [Generator Loss: 3.046302]
    


![png](out/output_16_509.png)


    25499 [Discriminator Loss: 0.186204, Acc.: 50.00%] [Generator Loss: 3.208229]
    


![png](out/output_16_511.png)


    25599 [Discriminator Loss: 0.187736, Acc.: 50.00%] [Generator Loss: 3.767744]
    


![png](out/output_16_513.png)


    25699 [Discriminator Loss: 0.174522, Acc.: 50.00%] [Generator Loss: 3.367731]
    


![png](out/output_16_515.png)


    25799 [Discriminator Loss: 0.190563, Acc.: 50.00%] [Generator Loss: 3.195689]
    


![png](out/output_16_517.png)


    25899 [Discriminator Loss: 0.185956, Acc.: 50.00%] [Generator Loss: 2.764851]
    


![png](out/output_16_519.png)


    25999 [Discriminator Loss: 0.196530, Acc.: 50.00%] [Generator Loss: 3.319268]
    


![png](out/output_16_521.png)


    26099 [Discriminator Loss: 0.179310, Acc.: 50.00%] [Generator Loss: 2.800326]
    


![png](out/output_16_523.png)


    26199 [Discriminator Loss: 0.185178, Acc.: 50.00%] [Generator Loss: 3.234683]
    


![png](out/output_16_525.png)


    26299 [Discriminator Loss: 0.180176, Acc.: 50.00%] [Generator Loss: 3.381241]
    


![png](out/output_16_527.png)


    26399 [Discriminator Loss: 0.190603, Acc.: 50.00%] [Generator Loss: 2.637016]
    


![png](out/output_16_529.png)


    26499 [Discriminator Loss: 0.178230, Acc.: 50.00%] [Generator Loss: 2.899812]
    


![png](out/output_16_531.png)


    26599 [Discriminator Loss: 0.192819, Acc.: 50.00%] [Generator Loss: 2.663577]
    


![png](out/output_16_533.png)


    26699 [Discriminator Loss: 0.178325, Acc.: 50.00%] [Generator Loss: 3.001212]
    


![png](out/output_16_535.png)


    26799 [Discriminator Loss: 0.175465, Acc.: 50.00%] [Generator Loss: 3.275593]
    


![png](out/output_16_537.png)


    26899 [Discriminator Loss: 0.175463, Acc.: 50.00%] [Generator Loss: 3.399632]
    


![png](out/output_16_539.png)


    26999 [Discriminator Loss: 0.173098, Acc.: 50.00%] [Generator Loss: 2.671695]
    


![png](out/output_16_541.png)


    27099 [Discriminator Loss: 0.169915, Acc.: 50.00%] [Generator Loss: 2.983675]
    


![png](out/output_16_543.png)


    27199 [Discriminator Loss: 0.227659, Acc.: 50.00%] [Generator Loss: 3.459003]
    


![png](out/output_16_545.png)


    27299 [Discriminator Loss: 0.175289, Acc.: 50.00%] [Generator Loss: 3.377780]
    


![png](out/output_16_547.png)


    27399 [Discriminator Loss: 0.180916, Acc.: 50.00%] [Generator Loss: 2.978377]
    


![png](out/output_16_549.png)


    27499 [Discriminator Loss: 0.175977, Acc.: 50.00%] [Generator Loss: 3.551251]
    


![png](out/output_16_551.png)


    27599 [Discriminator Loss: 0.181909, Acc.: 50.00%] [Generator Loss: 3.802594]
    


![png](out/output_16_553.png)


    27699 [Discriminator Loss: 0.186735, Acc.: 50.00%] [Generator Loss: 3.201486]
    


![png](out/output_16_555.png)


    27799 [Discriminator Loss: 0.184225, Acc.: 50.00%] [Generator Loss: 3.388523]
    


![png](out/output_16_557.png)


    27899 [Discriminator Loss: 0.179035, Acc.: 50.00%] [Generator Loss: 3.190830]
    


![png](out/output_16_559.png)


    27999 [Discriminator Loss: 0.173475, Acc.: 50.00%] [Generator Loss: 3.272876]
    


![png](out/output_16_561.png)


    28099 [Discriminator Loss: 0.172411, Acc.: 50.00%] [Generator Loss: 3.634982]
    


![png](out/output_16_563.png)


    28199 [Discriminator Loss: 0.179247, Acc.: 50.00%] [Generator Loss: 3.793573]
    


![png](out/output_16_565.png)


    28299 [Discriminator Loss: 0.179083, Acc.: 50.00%] [Generator Loss: 3.553690]
    


![png](out/output_16_567.png)


    28399 [Discriminator Loss: 0.167671, Acc.: 50.00%] [Generator Loss: 3.714415]
    


![png](out/output_16_569.png)


    28499 [Discriminator Loss: 0.174568, Acc.: 50.00%] [Generator Loss: 2.759072]
    


![png](out/output_16_571.png)


    28599 [Discriminator Loss: 0.172342, Acc.: 50.00%] [Generator Loss: 3.419189]
    


![png](out/output_16_573.png)


    28699 [Discriminator Loss: 0.171954, Acc.: 50.00%] [Generator Loss: 3.795979]
    


![png](out/output_16_575.png)


    28799 [Discriminator Loss: 0.184800, Acc.: 50.00%] [Generator Loss: 3.200469]
    


![png](out/output_16_577.png)


    28899 [Discriminator Loss: 0.169114, Acc.: 50.00%] [Generator Loss: 3.480252]
    


![png](out/output_16_579.png)


    28999 [Discriminator Loss: 0.179613, Acc.: 50.00%] [Generator Loss: 3.358675]
    


![png](out/output_16_581.png)


    29099 [Discriminator Loss: 0.178805, Acc.: 50.00%] [Generator Loss: 3.719623]
    


![png](out/output_16_583.png)


    29199 [Discriminator Loss: 0.175302, Acc.: 50.00%] [Generator Loss: 3.307360]
    


![png](out/output_16_585.png)


    29299 [Discriminator Loss: 0.173843, Acc.: 50.00%] [Generator Loss: 3.320118]
    


![png](out/output_16_587.png)


    29399 [Discriminator Loss: 0.178849, Acc.: 50.00%] [Generator Loss: 3.475919]
    


![png](out/output_16_589.png)


    29499 [Discriminator Loss: 0.178065, Acc.: 50.00%] [Generator Loss: 3.379612]
    


![png](out/output_16_591.png)


    29599 [Discriminator Loss: 0.176411, Acc.: 50.00%] [Generator Loss: 3.264698]
    


![png](out/output_16_593.png)


    29699 [Discriminator Loss: 0.171358, Acc.: 50.00%] [Generator Loss: 3.721380]
    


![png](out/output_16_595.png)


    29799 [Discriminator Loss: 0.169068, Acc.: 50.00%] [Generator Loss: 3.385775]
    


![png](out/output_16_597.png)


    29899 [Discriminator Loss: 0.194207, Acc.: 50.00%] [Generator Loss: 3.243869]
    


![png](out/output_16_599.png)


    29999 [Discriminator Loss: 0.167243, Acc.: 50.00%] [Generator Loss: 3.430578]
    


![png](out/output_16_601.png)


    30099 [Discriminator Loss: 0.169677, Acc.: 50.00%] [Generator Loss: 3.618952]
    


![png](out/output_16_603.png)


    30199 [Discriminator Loss: 0.179337, Acc.: 50.00%] [Generator Loss: 3.607438]
    


![png](out/output_16_605.png)


    30299 [Discriminator Loss: 0.183733, Acc.: 50.00%] [Generator Loss: 2.892971]
    


![png](out/output_16_607.png)


    30399 [Discriminator Loss: 0.198554, Acc.: 50.00%] [Generator Loss: 3.666486]
    


![png](out/output_16_609.png)


    30499 [Discriminator Loss: 0.209374, Acc.: 50.00%] [Generator Loss: 3.171605]
    


![png](out/output_16_611.png)


    30599 [Discriminator Loss: 0.169426, Acc.: 50.00%] [Generator Loss: 3.194911]
    


![png](out/output_16_613.png)


    30699 [Discriminator Loss: 0.176000, Acc.: 50.00%] [Generator Loss: 3.285362]
    


![png](out/output_16_615.png)


    30799 [Discriminator Loss: 0.172040, Acc.: 50.00%] [Generator Loss: 4.213117]
    


![png](out/output_16_617.png)


    30899 [Discriminator Loss: 0.182331, Acc.: 50.00%] [Generator Loss: 3.122890]
    


![png](out/output_16_619.png)


    30999 [Discriminator Loss: 0.168655, Acc.: 50.00%] [Generator Loss: 3.220893]
    


![png](out/output_16_621.png)


    31099 [Discriminator Loss: 0.181049, Acc.: 50.00%] [Generator Loss: 3.353928]
    


![png](out/output_16_623.png)


    31199 [Discriminator Loss: 0.183722, Acc.: 50.00%] [Generator Loss: 3.731144]
    


![png](out/output_16_625.png)


    31299 [Discriminator Loss: 0.181758, Acc.: 50.00%] [Generator Loss: 3.858708]
    


![png](out/output_16_627.png)


    31399 [Discriminator Loss: 0.178635, Acc.: 50.00%] [Generator Loss: 3.843863]
    


![png](out/output_16_629.png)


    31499 [Discriminator Loss: 0.171826, Acc.: 50.00%] [Generator Loss: 3.424491]
    


![png](out/output_16_631.png)


    31599 [Discriminator Loss: 0.215972, Acc.: 50.00%] [Generator Loss: 3.683404]
    


![png](out/output_16_633.png)


    31699 [Discriminator Loss: 0.183153, Acc.: 50.00%] [Generator Loss: 3.422256]
    


![png](out/output_16_635.png)


    31799 [Discriminator Loss: 0.231785, Acc.: 50.00%] [Generator Loss: 3.461879]
    


![png](out/output_16_637.png)


    31899 [Discriminator Loss: 0.171637, Acc.: 50.00%] [Generator Loss: 3.271887]
    


![png](out/output_16_639.png)


    31999 [Discriminator Loss: 0.198110, Acc.: 50.00%] [Generator Loss: 3.914008]
    


![png](out/output_16_641.png)


    32099 [Discriminator Loss: 0.171804, Acc.: 50.00%] [Generator Loss: 3.248473]
    


![png](out/output_16_643.png)


    32199 [Discriminator Loss: 0.168496, Acc.: 50.00%] [Generator Loss: 3.138527]
    


![png](out/output_16_645.png)


    32299 [Discriminator Loss: 0.172077, Acc.: 50.00%] [Generator Loss: 4.198875]
    


![png](out/output_16_647.png)


    32399 [Discriminator Loss: 0.192111, Acc.: 50.00%] [Generator Loss: 3.491166]
    


![png](out/output_16_649.png)


    32499 [Discriminator Loss: 0.184913, Acc.: 50.00%] [Generator Loss: 3.553454]
    


![png](out/output_16_651.png)


    32599 [Discriminator Loss: 0.186362, Acc.: 50.00%] [Generator Loss: 3.233759]
    


![png](out/output_16_653.png)


    32699 [Discriminator Loss: 0.176219, Acc.: 50.00%] [Generator Loss: 3.722950]
    


![png](out/output_16_655.png)


    32799 [Discriminator Loss: 0.169706, Acc.: 50.00%] [Generator Loss: 3.246888]
    


![png](out/output_16_657.png)


    32899 [Discriminator Loss: 0.180745, Acc.: 50.00%] [Generator Loss: 3.673698]
    


![png](out/output_16_659.png)


    32999 [Discriminator Loss: 0.180939, Acc.: 50.00%] [Generator Loss: 4.049567]
    


![png](out/output_16_661.png)


    33099 [Discriminator Loss: 0.166844, Acc.: 50.00%] [Generator Loss: 3.537128]
    


![png](out/output_16_663.png)


    33199 [Discriminator Loss: 0.166730, Acc.: 50.00%] [Generator Loss: 3.564548]
    


![png](out/output_16_665.png)


    33299 [Discriminator Loss: 0.171683, Acc.: 50.00%] [Generator Loss: 3.572303]
    


![png](out/output_16_667.png)


    33399 [Discriminator Loss: 0.173215, Acc.: 50.00%] [Generator Loss: 3.281308]
    


![png](out/output_16_669.png)


    33499 [Discriminator Loss: 0.178236, Acc.: 50.00%] [Generator Loss: 3.348641]
    


![png](out/output_16_671.png)


    33599 [Discriminator Loss: 0.169328, Acc.: 50.00%] [Generator Loss: 3.436580]
    


![png](out/output_16_673.png)


    33699 [Discriminator Loss: 0.172346, Acc.: 50.00%] [Generator Loss: 4.092560]
    


![png](out/output_16_675.png)


    33799 [Discriminator Loss: 0.176863, Acc.: 50.00%] [Generator Loss: 4.161169]
    


![png](out/output_16_677.png)


    33899 [Discriminator Loss: 0.172595, Acc.: 50.00%] [Generator Loss: 3.394410]
    


![png](out/output_16_679.png)


    33999 [Discriminator Loss: 0.173533, Acc.: 50.00%] [Generator Loss: 4.314133]
    


![png](out/output_16_681.png)


    34099 [Discriminator Loss: 0.180638, Acc.: 50.00%] [Generator Loss: 3.937844]
    


![png](out/output_16_683.png)


    34199 [Discriminator Loss: 0.179659, Acc.: 50.00%] [Generator Loss: 3.265039]
    


![png](out/output_16_685.png)


    34299 [Discriminator Loss: 0.172406, Acc.: 50.00%] [Generator Loss: 2.923755]
    


![png](out/output_16_687.png)


    34399 [Discriminator Loss: 0.170705, Acc.: 50.00%] [Generator Loss: 3.757886]
    


![png](out/output_16_689.png)


    34499 [Discriminator Loss: 0.167745, Acc.: 50.00%] [Generator Loss: 3.173198]
    


![png](out/output_16_691.png)


    34599 [Discriminator Loss: 0.176783, Acc.: 50.00%] [Generator Loss: 3.295165]
    


![png](out/output_16_693.png)


    34699 [Discriminator Loss: 0.175084, Acc.: 50.00%] [Generator Loss: 3.409611]
    


![png](out/output_16_695.png)


    34799 [Discriminator Loss: 0.171556, Acc.: 50.00%] [Generator Loss: 3.375717]
    


![png](out/output_16_697.png)


    34899 [Discriminator Loss: 0.176366, Acc.: 50.00%] [Generator Loss: 3.464029]
    


![png](out/output_16_699.png)


    34999 [Discriminator Loss: 0.167623, Acc.: 50.00%] [Generator Loss: 3.625896]
    


![png](out/output_16_701.png)


    35099 [Discriminator Loss: 0.169257, Acc.: 50.00%] [Generator Loss: 4.188911]
    


![png](out/output_16_703.png)


    35199 [Discriminator Loss: 0.178808, Acc.: 50.00%] [Generator Loss: 3.555305]
    


![png](out/output_16_705.png)


    35299 [Discriminator Loss: 0.170210, Acc.: 50.00%] [Generator Loss: 3.759647]
    


![png](out/output_16_707.png)


    35399 [Discriminator Loss: 0.172937, Acc.: 50.00%] [Generator Loss: 3.324501]
    


![png](out/output_16_709.png)


    35499 [Discriminator Loss: 0.175669, Acc.: 50.00%] [Generator Loss: 4.234406]
    


![png](out/output_16_711.png)


    35599 [Discriminator Loss: 0.181884, Acc.: 50.00%] [Generator Loss: 3.579845]
    


![png](out/output_16_713.png)


    35699 [Discriminator Loss: 0.180013, Acc.: 50.00%] [Generator Loss: 3.532092]
    


![png](out/output_16_715.png)


    35799 [Discriminator Loss: 0.194519, Acc.: 50.00%] [Generator Loss: 3.616288]
    


![png](out/output_16_717.png)


    35899 [Discriminator Loss: 0.170493, Acc.: 50.00%] [Generator Loss: 3.813031]
    


![png](out/output_16_719.png)


    35999 [Discriminator Loss: 0.174089, Acc.: 50.00%] [Generator Loss: 3.395636]
    


![png](out/output_16_721.png)


    36099 [Discriminator Loss: 0.184152, Acc.: 50.00%] [Generator Loss: 3.411500]
    


![png](out/output_16_723.png)


    36199 [Discriminator Loss: 0.170762, Acc.: 50.00%] [Generator Loss: 3.466535]
    


![png](out/output_16_725.png)


    36299 [Discriminator Loss: 0.173091, Acc.: 50.00%] [Generator Loss: 3.414053]
    


![png](out/output_16_727.png)


    36399 [Discriminator Loss: 0.165648, Acc.: 50.00%] [Generator Loss: 4.039439]
    


![png](out/output_16_729.png)


    36499 [Discriminator Loss: 0.194300, Acc.: 50.00%] [Generator Loss: 3.493540]
    


![png](out/output_16_731.png)


    36599 [Discriminator Loss: 0.171392, Acc.: 50.00%] [Generator Loss: 2.994950]
    


![png](out/output_16_733.png)


    36699 [Discriminator Loss: 0.171615, Acc.: 50.00%] [Generator Loss: 4.182058]
    


![png](out/output_16_735.png)


    36799 [Discriminator Loss: 0.173386, Acc.: 50.00%] [Generator Loss: 4.029098]
    


![png](out/output_16_737.png)


    36899 [Discriminator Loss: 0.222469, Acc.: 50.00%] [Generator Loss: 3.800794]
    


![png](out/output_16_739.png)


    36999 [Discriminator Loss: 0.167523, Acc.: 50.00%] [Generator Loss: 3.444054]
    


![png](out/output_16_741.png)


    37099 [Discriminator Loss: 0.172212, Acc.: 50.00%] [Generator Loss: 3.758762]
    


![png](out/output_16_743.png)


    37199 [Discriminator Loss: 0.170700, Acc.: 50.00%] [Generator Loss: 3.327916]
    


![png](out/output_16_745.png)


    37299 [Discriminator Loss: 0.206432, Acc.: 50.00%] [Generator Loss: 3.926234]
    


![png](out/output_16_747.png)


    37399 [Discriminator Loss: 0.174715, Acc.: 50.00%] [Generator Loss: 3.355088]
    


![png](out/output_16_749.png)


    37499 [Discriminator Loss: 0.190374, Acc.: 50.00%] [Generator Loss: 3.293921]
    


![png](out/output_16_751.png)


    37599 [Discriminator Loss: 0.174114, Acc.: 50.00%] [Generator Loss: 3.386294]
    


![png](out/output_16_753.png)


    37699 [Discriminator Loss: 0.172222, Acc.: 50.00%] [Generator Loss: 3.480362]
    


![png](out/output_16_755.png)


    37799 [Discriminator Loss: 0.166801, Acc.: 50.00%] [Generator Loss: 3.493763]
    


![png](out/output_16_757.png)


    37899 [Discriminator Loss: 0.167388, Acc.: 50.00%] [Generator Loss: 3.866719]
    


![png](out/output_16_759.png)


    37999 [Discriminator Loss: 0.186767, Acc.: 50.00%] [Generator Loss: 3.574927]
    


![png](out/output_16_761.png)


    38099 [Discriminator Loss: 0.180422, Acc.: 50.00%] [Generator Loss: 3.105693]
    


![png](out/output_16_763.png)


    38199 [Discriminator Loss: 0.176546, Acc.: 50.00%] [Generator Loss: 4.048934]
    


![png](out/output_16_765.png)


    38299 [Discriminator Loss: 0.168482, Acc.: 50.00%] [Generator Loss: 3.875261]
    


![png](out/output_16_767.png)


    38399 [Discriminator Loss: 0.191372, Acc.: 50.00%] [Generator Loss: 3.624226]
    


![png](out/output_16_769.png)


    38499 [Discriminator Loss: 0.171411, Acc.: 50.00%] [Generator Loss: 4.052884]
    


![png](out/output_16_771.png)


    38599 [Discriminator Loss: 0.172535, Acc.: 50.00%] [Generator Loss: 4.091908]
    


![png](out/output_16_773.png)


    38699 [Discriminator Loss: 0.174461, Acc.: 50.00%] [Generator Loss: 4.490830]
    


![png](out/output_16_775.png)


    38799 [Discriminator Loss: 0.171342, Acc.: 50.00%] [Generator Loss: 4.182658]
    


![png](out/output_16_777.png)


    38899 [Discriminator Loss: 0.174326, Acc.: 50.00%] [Generator Loss: 3.299911]
    


![png](out/output_16_779.png)


    38999 [Discriminator Loss: 0.168123, Acc.: 50.00%] [Generator Loss: 4.318399]
    


![png](out/output_16_781.png)


    39099 [Discriminator Loss: 0.165896, Acc.: 50.00%] [Generator Loss: 4.070023]
    


![png](out/output_16_783.png)


    39199 [Discriminator Loss: 0.175434, Acc.: 50.00%] [Generator Loss: 3.747022]
    


![png](out/output_16_785.png)


    39299 [Discriminator Loss: 0.169694, Acc.: 50.00%] [Generator Loss: 3.668037]
    


![png](out/output_16_787.png)


    39399 [Discriminator Loss: 0.168083, Acc.: 50.00%] [Generator Loss: 3.469756]
    


![png](out/output_16_789.png)


    39499 [Discriminator Loss: 0.174328, Acc.: 50.00%] [Generator Loss: 3.529827]
    


![png](out/output_16_791.png)


    39599 [Discriminator Loss: 0.167990, Acc.: 50.00%] [Generator Loss: 3.704803]
    


![png](out/output_16_793.png)


    39699 [Discriminator Loss: 0.184581, Acc.: 50.00%] [Generator Loss: 4.260494]
    


![png](out/output_16_795.png)


    39799 [Discriminator Loss: 0.176272, Acc.: 50.00%] [Generator Loss: 3.652171]
    


![png](out/output_16_797.png)


    39899 [Discriminator Loss: 0.194486, Acc.: 46.88%] [Generator Loss: 4.164394]
    


![png](out/output_16_799.png)


    39999 [Discriminator Loss: 0.172214, Acc.: 50.00%] [Generator Loss: 3.294837]
    


![png](out/output_16_801.png)


    40099 [Discriminator Loss: 0.168753, Acc.: 50.00%] [Generator Loss: 4.566163]
    


![png](out/output_16_803.png)


    40199 [Discriminator Loss: 0.217923, Acc.: 50.00%] [Generator Loss: 4.049767]
    


![png](out/output_16_805.png)


    40299 [Discriminator Loss: 0.171979, Acc.: 50.00%] [Generator Loss: 3.233850]
    


![png](out/output_16_807.png)


    40399 [Discriminator Loss: 0.169647, Acc.: 50.00%] [Generator Loss: 3.802708]
    


![png](out/output_16_809.png)


    40499 [Discriminator Loss: 0.175402, Acc.: 50.00%] [Generator Loss: 3.517692]
    


![png](out/output_16_811.png)


    40599 [Discriminator Loss: 0.181517, Acc.: 50.00%] [Generator Loss: 3.861989]
    


![png](out/output_16_813.png)


    40699 [Discriminator Loss: 0.171543, Acc.: 50.00%] [Generator Loss: 3.507478]
    


![png](out/output_16_815.png)


    40799 [Discriminator Loss: 0.171749, Acc.: 50.00%] [Generator Loss: 3.952843]
    


![png](out/output_16_817.png)


    40899 [Discriminator Loss: 0.177990, Acc.: 50.00%] [Generator Loss: 4.321552]
    


![png](out/output_16_819.png)


    40999 [Discriminator Loss: 0.180806, Acc.: 50.00%] [Generator Loss: 3.577153]
    


![png](out/output_16_821.png)


    41099 [Discriminator Loss: 0.170340, Acc.: 50.00%] [Generator Loss: 4.045473]
    


![png](out/output_16_823.png)


    41199 [Discriminator Loss: 0.175298, Acc.: 50.00%] [Generator Loss: 4.157060]
    


![png](out/output_16_825.png)


    41299 [Discriminator Loss: 0.168703, Acc.: 50.00%] [Generator Loss: 3.451894]
    


![png](out/output_16_827.png)


    41399 [Discriminator Loss: 0.169202, Acc.: 50.00%] [Generator Loss: 3.868298]
    


![png](out/output_16_829.png)


    41499 [Discriminator Loss: 0.176197, Acc.: 50.00%] [Generator Loss: 4.186415]
    


![png](out/output_16_831.png)


    41599 [Discriminator Loss: 0.168064, Acc.: 50.00%] [Generator Loss: 3.723767]
    


![png](out/output_16_833.png)


    41699 [Discriminator Loss: 0.167493, Acc.: 50.00%] [Generator Loss: 3.894537]
    


![png](out/output_16_835.png)


    41799 [Discriminator Loss: 0.173534, Acc.: 50.00%] [Generator Loss: 3.729501]
    


![png](out/output_16_837.png)


    41899 [Discriminator Loss: 0.171809, Acc.: 50.00%] [Generator Loss: 3.881087]
    


![png](out/output_16_839.png)


    41999 [Discriminator Loss: 0.169937, Acc.: 50.00%] [Generator Loss: 3.935064]
    


![png](out/output_16_841.png)


    42099 [Discriminator Loss: 0.188037, Acc.: 50.00%] [Generator Loss: 3.896228]
    


![png](out/output_16_843.png)


    42199 [Discriminator Loss: 0.167715, Acc.: 50.00%] [Generator Loss: 4.331194]
    


![png](out/output_16_845.png)


    42299 [Discriminator Loss: 0.193048, Acc.: 50.00%] [Generator Loss: 3.746816]
    


![png](out/output_16_847.png)


    42399 [Discriminator Loss: 0.179800, Acc.: 50.00%] [Generator Loss: 4.526221]
    


![png](out/output_16_849.png)


    42499 [Discriminator Loss: 0.167695, Acc.: 50.00%] [Generator Loss: 3.784927]
    


![png](out/output_16_851.png)


    42599 [Discriminator Loss: 0.189289, Acc.: 50.00%] [Generator Loss: 3.886573]
    


![png](out/output_16_853.png)


    42699 [Discriminator Loss: 0.167097, Acc.: 50.00%] [Generator Loss: 4.021296]
    


![png](out/output_16_855.png)


    42799 [Discriminator Loss: 0.165672, Acc.: 50.00%] [Generator Loss: 3.882214]
    


![png](out/output_16_857.png)


    42899 [Discriminator Loss: 0.176815, Acc.: 50.00%] [Generator Loss: 3.652494]
    


![png](out/output_16_859.png)


    42999 [Discriminator Loss: 0.170600, Acc.: 50.00%] [Generator Loss: 4.555285]
    


![png](out/output_16_861.png)


    43099 [Discriminator Loss: 0.169470, Acc.: 50.00%] [Generator Loss: 3.387407]
    


![png](out/output_16_863.png)


    43199 [Discriminator Loss: 0.169031, Acc.: 50.00%] [Generator Loss: 4.268742]
    


![png](out/output_16_865.png)


    43299 [Discriminator Loss: 0.171131, Acc.: 50.00%] [Generator Loss: 4.487414]
    


![png](out/output_16_867.png)


    43399 [Discriminator Loss: 0.168896, Acc.: 50.00%] [Generator Loss: 4.002507]
    


![png](out/output_16_869.png)


    43499 [Discriminator Loss: 0.171232, Acc.: 50.00%] [Generator Loss: 4.035852]
    


![png](out/output_16_871.png)


    43599 [Discriminator Loss: 0.169279, Acc.: 50.00%] [Generator Loss: 4.133211]
    


![png](out/output_16_873.png)


    43699 [Discriminator Loss: 0.170520, Acc.: 50.00%] [Generator Loss: 3.741114]
    


![png](out/output_16_875.png)


    43799 [Discriminator Loss: 0.205959, Acc.: 50.00%] [Generator Loss: 4.090072]
    


![png](out/output_16_877.png)


    43899 [Discriminator Loss: 0.170131, Acc.: 50.00%] [Generator Loss: 4.382812]
    


![png](out/output_16_879.png)


    43999 [Discriminator Loss: 0.198339, Acc.: 50.00%] [Generator Loss: 4.393247]
    


![png](out/output_16_881.png)


    44099 [Discriminator Loss: 0.196766, Acc.: 46.88%] [Generator Loss: 4.458750]
    


![png](out/output_16_883.png)


    44199 [Discriminator Loss: 0.182276, Acc.: 50.00%] [Generator Loss: 4.073106]
    


![png](out/output_16_885.png)


    44299 [Discriminator Loss: 0.174179, Acc.: 50.00%] [Generator Loss: 4.014807]
    


![png](out/output_16_887.png)


    44399 [Discriminator Loss: 0.177526, Acc.: 50.00%] [Generator Loss: 4.150361]
    


![png](out/output_16_889.png)


    44499 [Discriminator Loss: 0.174123, Acc.: 50.00%] [Generator Loss: 4.642211]
    


![png](out/output_16_891.png)


    44599 [Discriminator Loss: 0.185741, Acc.: 50.00%] [Generator Loss: 3.765075]
    


![png](out/output_16_893.png)


    44699 [Discriminator Loss: 0.166751, Acc.: 50.00%] [Generator Loss: 4.323178]
    


![png](out/output_16_895.png)


    44799 [Discriminator Loss: 0.182885, Acc.: 50.00%] [Generator Loss: 3.848352]
    


![png](out/output_16_897.png)


    44899 [Discriminator Loss: 0.171243, Acc.: 50.00%] [Generator Loss: 4.391182]
    


![png](out/output_16_899.png)


    44999 [Discriminator Loss: 0.174106, Acc.: 50.00%] [Generator Loss: 4.405748]
    


![png](out/output_16_901.png)


    45099 [Discriminator Loss: 0.198862, Acc.: 50.00%] [Generator Loss: 5.042504]
    


![png](out/output_16_903.png)


    45199 [Discriminator Loss: 0.166279, Acc.: 50.00%] [Generator Loss: 4.093554]
    


![png](out/output_16_905.png)


    45299 [Discriminator Loss: 0.179308, Acc.: 50.00%] [Generator Loss: 4.059516]
    


![png](out/output_16_907.png)


    45399 [Discriminator Loss: 0.180266, Acc.: 50.00%] [Generator Loss: 4.061395]
    


![png](out/output_16_909.png)


    45499 [Discriminator Loss: 0.212611, Acc.: 50.00%] [Generator Loss: 4.511559]
    


![png](out/output_16_911.png)


    45599 [Discriminator Loss: 0.170549, Acc.: 50.00%] [Generator Loss: 4.298257]
    


![png](out/output_16_913.png)


    45699 [Discriminator Loss: 0.169718, Acc.: 50.00%] [Generator Loss: 3.844030]
    


![png](out/output_16_915.png)


    45799 [Discriminator Loss: 0.169654, Acc.: 50.00%] [Generator Loss: 4.095959]
    


![png](out/output_16_917.png)


    45899 [Discriminator Loss: 0.171248, Acc.: 50.00%] [Generator Loss: 3.915095]
    


![png](out/output_16_919.png)


    45999 [Discriminator Loss: 0.170620, Acc.: 50.00%] [Generator Loss: 4.595392]
    


![png](out/output_16_921.png)


    46099 [Discriminator Loss: 0.173369, Acc.: 50.00%] [Generator Loss: 4.069066]
    


![png](out/output_16_923.png)


    46199 [Discriminator Loss: 0.170859, Acc.: 50.00%] [Generator Loss: 3.442462]
    


![png](out/output_16_925.png)


    46299 [Discriminator Loss: 0.171299, Acc.: 50.00%] [Generator Loss: 3.686182]
    


![png](out/output_16_927.png)


    46399 [Discriminator Loss: 0.166395, Acc.: 50.00%] [Generator Loss: 4.255854]
    


![png](out/output_16_929.png)


    46499 [Discriminator Loss: 0.168418, Acc.: 50.00%] [Generator Loss: 3.658561]
    


![png](out/output_16_931.png)


    46599 [Discriminator Loss: 0.169841, Acc.: 50.00%] [Generator Loss: 4.411891]
    


![png](out/output_16_933.png)


    46699 [Discriminator Loss: 0.168519, Acc.: 50.00%] [Generator Loss: 3.970046]
    


![png](out/output_16_935.png)


    46799 [Discriminator Loss: 0.169328, Acc.: 50.00%] [Generator Loss: 4.160697]
    


![png](out/output_16_937.png)


    46899 [Discriminator Loss: 0.166029, Acc.: 50.00%] [Generator Loss: 4.119895]
    


![png](out/output_16_939.png)


    46999 [Discriminator Loss: 0.170840, Acc.: 50.00%] [Generator Loss: 4.263502]
    


![png](out/output_16_941.png)


    47099 [Discriminator Loss: 0.180829, Acc.: 50.00%] [Generator Loss: 4.228706]
    


![png](out/output_16_943.png)


    47199 [Discriminator Loss: 0.174696, Acc.: 50.00%] [Generator Loss: 3.928238]
    


![png](out/output_16_945.png)


    47299 [Discriminator Loss: 0.170483, Acc.: 50.00%] [Generator Loss: 4.442750]
    


![png](out/output_16_947.png)


    47399 [Discriminator Loss: 0.164483, Acc.: 50.00%] [Generator Loss: 3.967094]
    


![png](out/output_16_949.png)


    47499 [Discriminator Loss: 0.173568, Acc.: 50.00%] [Generator Loss: 3.948384]
    


![png](out/output_16_951.png)


    47599 [Discriminator Loss: 0.181831, Acc.: 50.00%] [Generator Loss: 4.233997]
    


![png](out/output_16_953.png)


    47699 [Discriminator Loss: 0.166268, Acc.: 50.00%] [Generator Loss: 4.447830]
    


![png](out/output_16_955.png)


    47799 [Discriminator Loss: 0.174530, Acc.: 50.00%] [Generator Loss: 4.406268]
    


![png](out/output_16_957.png)


    47899 [Discriminator Loss: 0.167720, Acc.: 50.00%] [Generator Loss: 4.043373]
    


![png](out/output_16_959.png)


    47999 [Discriminator Loss: 0.171427, Acc.: 50.00%] [Generator Loss: 4.445008]
    


![png](out/output_16_961.png)


    48099 [Discriminator Loss: 0.178562, Acc.: 50.00%] [Generator Loss: 4.182507]
    


![png](out/output_16_963.png)


    48199 [Discriminator Loss: 0.175552, Acc.: 50.00%] [Generator Loss: 4.542088]
    


![png](out/output_16_965.png)


    48299 [Discriminator Loss: 0.231095, Acc.: 50.00%] [Generator Loss: 4.203918]
    


![png](out/output_16_967.png)


    48399 [Discriminator Loss: 0.189862, Acc.: 50.00%] [Generator Loss: 4.250311]
    


![png](out/output_16_969.png)


    48499 [Discriminator Loss: 0.166376, Acc.: 50.00%] [Generator Loss: 4.290440]
    


![png](out/output_16_971.png)


    48599 [Discriminator Loss: 0.167829, Acc.: 50.00%] [Generator Loss: 4.219207]
    


![png](out/output_16_973.png)


    48699 [Discriminator Loss: 0.178496, Acc.: 50.00%] [Generator Loss: 4.320604]
    


![png](out/output_16_975.png)


    48799 [Discriminator Loss: 0.173940, Acc.: 50.00%] [Generator Loss: 4.274955]
    


![png](out/output_16_977.png)


    48899 [Discriminator Loss: 0.167065, Acc.: 50.00%] [Generator Loss: 4.332558]
    


![png](out/output_16_979.png)


    48999 [Discriminator Loss: 0.174003, Acc.: 50.00%] [Generator Loss: 4.585050]
    


![png](out/output_16_981.png)


    49099 [Discriminator Loss: 0.177381, Acc.: 50.00%] [Generator Loss: 4.693356]
    


![png](out/output_16_983.png)


    49199 [Discriminator Loss: 0.168161, Acc.: 50.00%] [Generator Loss: 4.360839]
    


![png](out/output_16_985.png)


    49299 [Discriminator Loss: 0.169208, Acc.: 50.00%] [Generator Loss: 3.882297]
    


![png](out/output_16_987.png)


    49399 [Discriminator Loss: 0.168784, Acc.: 50.00%] [Generator Loss: 3.757976]
    


![png](out/output_16_989.png)


    49499 [Discriminator Loss: 0.166709, Acc.: 50.00%] [Generator Loss: 3.694577]
    


![png](out/output_16_991.png)


    49599 [Discriminator Loss: 0.169207, Acc.: 50.00%] [Generator Loss: 4.215886]
    


![png](out/output_16_993.png)


    49699 [Discriminator Loss: 0.168009, Acc.: 50.00%] [Generator Loss: 3.723871]
    


![png](out/output_16_995.png)


    49799 [Discriminator Loss: 0.227255, Acc.: 50.00%] [Generator Loss: 4.130327]
    


![png](out/output_16_997.png)


    49899 [Discriminator Loss: 0.174219, Acc.: 50.00%] [Generator Loss: 4.125039]
    


![png](out/output_16_999.png)


    49999 [Discriminator Loss: 0.175618, Acc.: 50.00%] [Generator Loss: 3.645547]
    


![png](out/output_16_1001.png)


    50099 [Discriminator Loss: 0.170797, Acc.: 50.00%] [Generator Loss: 3.687228]
    


![png](out/output_16_1003.png)


    50199 [Discriminator Loss: 0.164828, Acc.: 50.00%] [Generator Loss: 3.864794]
    


![png](out/output_16_1005.png)


    50299 [Discriminator Loss: 0.168882, Acc.: 50.00%] [Generator Loss: 3.940813]
    


![png](out/output_16_1007.png)


    50399 [Discriminator Loss: 0.173035, Acc.: 50.00%] [Generator Loss: 3.714804]
    


![png](out/output_16_1009.png)


    50499 [Discriminator Loss: 0.167504, Acc.: 50.00%] [Generator Loss: 3.407152]
    


![png](out/output_16_1011.png)


    50599 [Discriminator Loss: 0.168759, Acc.: 50.00%] [Generator Loss: 4.225538]
    


![png](out/output_16_1013.png)


    50699 [Discriminator Loss: 0.165737, Acc.: 50.00%] [Generator Loss: 3.696607]
    


![png](out/output_16_1015.png)


    50799 [Discriminator Loss: 0.172496, Acc.: 50.00%] [Generator Loss: 4.547211]
    


![png](out/output_16_1017.png)


    50899 [Discriminator Loss: 0.167549, Acc.: 50.00%] [Generator Loss: 3.412059]
    


![png](out/output_16_1019.png)


    50999 [Discriminator Loss: 0.209590, Acc.: 50.00%] [Generator Loss: 3.937885]
    


![png](out/output_16_1021.png)


    51099 [Discriminator Loss: 0.167474, Acc.: 50.00%] [Generator Loss: 3.622254]
    


![png](out/output_16_1023.png)


    51199 [Discriminator Loss: 0.166673, Acc.: 50.00%] [Generator Loss: 3.457250]
    


![png](out/output_16_1025.png)


    51299 [Discriminator Loss: 0.173143, Acc.: 50.00%] [Generator Loss: 3.187817]
    


![png](out/output_16_1027.png)


    51399 [Discriminator Loss: 0.217795, Acc.: 50.00%] [Generator Loss: 3.767855]
    


![png](out/output_16_1029.png)


    51499 [Discriminator Loss: 0.176580, Acc.: 50.00%] [Generator Loss: 3.866333]
    


![png](out/output_16_1031.png)


    51599 [Discriminator Loss: 0.173500, Acc.: 50.00%] [Generator Loss: 4.133635]
    


![png](out/output_16_1033.png)


    51699 [Discriminator Loss: 0.171981, Acc.: 50.00%] [Generator Loss: 4.221078]
    


![png](out/output_16_1035.png)


    51799 [Discriminator Loss: 0.170614, Acc.: 50.00%] [Generator Loss: 3.844988]
    


![png](out/output_16_1037.png)


    51899 [Discriminator Loss: 0.166673, Acc.: 50.00%] [Generator Loss: 3.615747]
    


![png](out/output_16_1039.png)


    51999 [Discriminator Loss: 0.171672, Acc.: 50.00%] [Generator Loss: 3.864882]
    


![png](out/output_16_1041.png)


    52099 [Discriminator Loss: 0.167622, Acc.: 50.00%] [Generator Loss: 3.702821]
    


![png](out/output_16_1043.png)


    52199 [Discriminator Loss: 0.167486, Acc.: 50.00%] [Generator Loss: 4.797946]
    


![png](out/output_16_1045.png)


    52299 [Discriminator Loss: 0.175900, Acc.: 50.00%] [Generator Loss: 4.009771]
    


![png](out/output_16_1047.png)


    52399 [Discriminator Loss: 0.170296, Acc.: 50.00%] [Generator Loss: 4.280422]
    


![png](out/output_16_1049.png)


    52499 [Discriminator Loss: 0.176467, Acc.: 50.00%] [Generator Loss: 3.794808]
    


![png](out/output_16_1051.png)


    52599 [Discriminator Loss: 0.167977, Acc.: 50.00%] [Generator Loss: 3.470452]
    


![png](out/output_16_1053.png)


    52699 [Discriminator Loss: 0.173072, Acc.: 50.00%] [Generator Loss: 4.339450]
    


![png](out/output_16_1055.png)


    52799 [Discriminator Loss: 0.181567, Acc.: 50.00%] [Generator Loss: 3.939634]
    


![png](out/output_16_1057.png)


    52899 [Discriminator Loss: 0.175589, Acc.: 50.00%] [Generator Loss: 4.042629]
    


![png](out/output_16_1059.png)


    52999 [Discriminator Loss: 0.167611, Acc.: 50.00%] [Generator Loss: 4.342577]
    


![png](out/output_16_1061.png)


    53099 [Discriminator Loss: 0.166924, Acc.: 50.00%] [Generator Loss: 3.723512]
    


![png](out/output_16_1063.png)


    53199 [Discriminator Loss: 0.170355, Acc.: 50.00%] [Generator Loss: 4.417939]
    


![png](out/output_16_1065.png)


    53299 [Discriminator Loss: 0.196134, Acc.: 50.00%] [Generator Loss: 3.617288]
    


![png](out/output_16_1067.png)


    53399 [Discriminator Loss: 0.170047, Acc.: 50.00%] [Generator Loss: 3.761254]
    


![png](out/output_16_1069.png)


    53499 [Discriminator Loss: 0.165916, Acc.: 50.00%] [Generator Loss: 4.194189]
    


![png](out/output_16_1071.png)


    53599 [Discriminator Loss: 0.167884, Acc.: 50.00%] [Generator Loss: 3.668500]
    


![png](out/output_16_1073.png)


    53699 [Discriminator Loss: 0.169597, Acc.: 50.00%] [Generator Loss: 3.769317]
    


![png](out/output_16_1075.png)


    53799 [Discriminator Loss: 0.168160, Acc.: 50.00%] [Generator Loss: 3.903257]
    


![png](out/output_16_1077.png)


    53899 [Discriminator Loss: 0.173584, Acc.: 50.00%] [Generator Loss: 3.417048]
    


![png](out/output_16_1079.png)


    53999 [Discriminator Loss: 0.171801, Acc.: 50.00%] [Generator Loss: 3.547168]
    


![png](out/output_16_1081.png)


    54099 [Discriminator Loss: 0.164069, Acc.: 50.00%] [Generator Loss: 3.903095]
    


![png](out/output_16_1083.png)


    54199 [Discriminator Loss: 0.173825, Acc.: 50.00%] [Generator Loss: 3.855528]
    


![png](out/output_16_1085.png)


    54299 [Discriminator Loss: 0.180514, Acc.: 50.00%] [Generator Loss: 3.925722]
    


![png](out/output_16_1087.png)


    54399 [Discriminator Loss: 0.168292, Acc.: 50.00%] [Generator Loss: 4.159027]
    


![png](out/output_16_1089.png)


    54499 [Discriminator Loss: 0.172820, Acc.: 50.00%] [Generator Loss: 3.785685]
    


![png](out/output_16_1091.png)


    54599 [Discriminator Loss: 0.180511, Acc.: 50.00%] [Generator Loss: 4.016995]
    


![png](out/output_16_1093.png)


    54699 [Discriminator Loss: 0.165147, Acc.: 50.00%] [Generator Loss: 3.854962]
    


![png](out/output_16_1095.png)


    54799 [Discriminator Loss: 0.169855, Acc.: 50.00%] [Generator Loss: 4.885020]
    


![png](out/output_16_1097.png)


    54899 [Discriminator Loss: 0.166100, Acc.: 50.00%] [Generator Loss: 4.265071]
    


![png](out/output_16_1099.png)


    54999 [Discriminator Loss: 0.169438, Acc.: 50.00%] [Generator Loss: 3.960819]
    


![png](out/output_16_1101.png)


    55099 [Discriminator Loss: 0.176142, Acc.: 50.00%] [Generator Loss: 4.398124]
    


![png](out/output_16_1103.png)


    55199 [Discriminator Loss: 0.169230, Acc.: 50.00%] [Generator Loss: 3.821232]
    


![png](out/output_16_1105.png)


    55299 [Discriminator Loss: 0.164905, Acc.: 50.00%] [Generator Loss: 3.686761]
    


![png](out/output_16_1107.png)


    55399 [Discriminator Loss: 0.178887, Acc.: 50.00%] [Generator Loss: 3.635380]
    


![png](out/output_16_1109.png)


    55499 [Discriminator Loss: 0.166595, Acc.: 50.00%] [Generator Loss: 3.934341]
    


![png](out/output_16_1111.png)


    55599 [Discriminator Loss: 0.169662, Acc.: 50.00%] [Generator Loss: 4.366345]
    


![png](out/output_16_1113.png)


    55699 [Discriminator Loss: 0.169615, Acc.: 50.00%] [Generator Loss: 4.667460]
    


![png](out/output_16_1115.png)


    55799 [Discriminator Loss: 0.216324, Acc.: 50.00%] [Generator Loss: 3.866903]
    


![png](out/output_16_1117.png)


    55899 [Discriminator Loss: 0.188320, Acc.: 50.00%] [Generator Loss: 4.104471]
    


![png](out/output_16_1119.png)


    55999 [Discriminator Loss: 0.169246, Acc.: 50.00%] [Generator Loss: 4.137195]
    


![png](out/output_16_1121.png)


    56099 [Discriminator Loss: 0.174673, Acc.: 50.00%] [Generator Loss: 4.247594]
    


![png](out/output_16_1123.png)


    56199 [Discriminator Loss: 0.168607, Acc.: 50.00%] [Generator Loss: 4.349704]
    


![png](out/output_16_1125.png)


    56299 [Discriminator Loss: 0.176328, Acc.: 50.00%] [Generator Loss: 4.122030]
    


![png](out/output_16_1127.png)


    56399 [Discriminator Loss: 0.168473, Acc.: 50.00%] [Generator Loss: 4.044287]
    


![png](out/output_16_1129.png)


    56499 [Discriminator Loss: 0.173801, Acc.: 50.00%] [Generator Loss: 4.054206]
    


![png](out/output_16_1131.png)


    56599 [Discriminator Loss: 0.166484, Acc.: 50.00%] [Generator Loss: 4.236649]
    


![png](out/output_16_1133.png)


    56699 [Discriminator Loss: 0.169147, Acc.: 50.00%] [Generator Loss: 4.239464]
    


![png](out/output_16_1135.png)


    56799 [Discriminator Loss: 0.167080, Acc.: 50.00%] [Generator Loss: 3.657972]
    


![png](out/output_16_1137.png)


    56899 [Discriminator Loss: 0.176176, Acc.: 50.00%] [Generator Loss: 3.604937]
    


![png](out/output_16_1139.png)


    56999 [Discriminator Loss: 0.168239, Acc.: 50.00%] [Generator Loss: 4.082346]
    


![png](out/output_16_1141.png)


    57099 [Discriminator Loss: 0.178261, Acc.: 50.00%] [Generator Loss: 4.141805]
    


![png](out/output_16_1143.png)


    57199 [Discriminator Loss: 0.169979, Acc.: 50.00%] [Generator Loss: 4.026227]
    


![png](out/output_16_1145.png)


    57299 [Discriminator Loss: 0.166179, Acc.: 50.00%] [Generator Loss: 4.479825]
    


![png](out/output_16_1147.png)


    57399 [Discriminator Loss: 0.174924, Acc.: 50.00%] [Generator Loss: 3.551854]
    


![png](out/output_16_1149.png)


    57499 [Discriminator Loss: 0.166111, Acc.: 50.00%] [Generator Loss: 4.037167]
    


![png](out/output_16_1151.png)


    57599 [Discriminator Loss: 0.165450, Acc.: 50.00%] [Generator Loss: 4.031806]
    


![png](out/output_16_1153.png)


    57699 [Discriminator Loss: 0.170129, Acc.: 50.00%] [Generator Loss: 4.171120]
    


![png](out/output_16_1155.png)


    57799 [Discriminator Loss: 0.174997, Acc.: 50.00%] [Generator Loss: 4.140914]
    


![png](out/output_16_1157.png)


    57899 [Discriminator Loss: 0.166735, Acc.: 50.00%] [Generator Loss: 4.117665]
    


![png](out/output_16_1159.png)


    57999 [Discriminator Loss: 0.167547, Acc.: 50.00%] [Generator Loss: 4.529259]
    


![png](out/output_16_1161.png)


    58099 [Discriminator Loss: 0.168910, Acc.: 50.00%] [Generator Loss: 3.979825]
    


![png](out/output_16_1163.png)


    58199 [Discriminator Loss: 0.165486, Acc.: 50.00%] [Generator Loss: 4.228071]
    


![png](out/output_16_1165.png)


    58299 [Discriminator Loss: 0.166807, Acc.: 50.00%] [Generator Loss: 4.073617]
    


![png](out/output_16_1167.png)


    58399 [Discriminator Loss: 0.176588, Acc.: 50.00%] [Generator Loss: 4.081230]
    


![png](out/output_16_1169.png)


    58499 [Discriminator Loss: 0.177122, Acc.: 50.00%] [Generator Loss: 4.308475]
    


![png](out/output_16_1171.png)


    58599 [Discriminator Loss: 0.197232, Acc.: 50.00%] [Generator Loss: 3.965599]
    


![png](out/output_16_1173.png)


    58699 [Discriminator Loss: 0.171331, Acc.: 50.00%] [Generator Loss: 4.095304]
    


![png](out/output_16_1175.png)


    58799 [Discriminator Loss: 0.167030, Acc.: 50.00%] [Generator Loss: 3.952383]
    


![png](out/output_16_1177.png)


    58899 [Discriminator Loss: 0.175406, Acc.: 50.00%] [Generator Loss: 3.810039]
    


![png](out/output_16_1179.png)


    58999 [Discriminator Loss: 0.169419, Acc.: 50.00%] [Generator Loss: 4.185541]
    


![png](out/output_16_1181.png)


    59099 [Discriminator Loss: 0.167212, Acc.: 50.00%] [Generator Loss: 4.207330]
    


![png](out/output_16_1183.png)


    59199 [Discriminator Loss: 0.172043, Acc.: 50.00%] [Generator Loss: 4.303891]
    


![png](out/output_16_1185.png)


    59299 [Discriminator Loss: 0.168222, Acc.: 50.00%] [Generator Loss: 3.881241]
    


![png](out/output_16_1187.png)


    59399 [Discriminator Loss: 0.174271, Acc.: 50.00%] [Generator Loss: 4.520933]
    


![png](out/output_16_1189.png)


    59499 [Discriminator Loss: 0.167494, Acc.: 50.00%] [Generator Loss: 3.700035]
    


![png](out/output_16_1191.png)


    59599 [Discriminator Loss: 0.166969, Acc.: 50.00%] [Generator Loss: 3.924810]
    


![png](out/output_16_1193.png)


    59699 [Discriminator Loss: 0.172173, Acc.: 50.00%] [Generator Loss: 4.664700]
    


![png](out/output_16_1195.png)


    59799 [Discriminator Loss: 0.165281, Acc.: 50.00%] [Generator Loss: 4.313945]
    


![png](out/output_16_1197.png)


    59899 [Discriminator Loss: 0.166908, Acc.: 50.00%] [Generator Loss: 4.338559]
    


![png](out/output_16_1199.png)


    59999 [Discriminator Loss: 0.173009, Acc.: 50.00%] [Generator Loss: 4.639966]
    


![png](out/output_16_1201.png)


    60099 [Discriminator Loss: 0.167725, Acc.: 50.00%] [Generator Loss: 4.098518]
    


![png](out/output_16_1203.png)


    60199 [Discriminator Loss: 0.213480, Acc.: 50.00%] [Generator Loss: 4.122925]
    


![png](out/output_16_1205.png)


    60299 [Discriminator Loss: 0.166269, Acc.: 50.00%] [Generator Loss: 4.056790]
    


![png](out/output_16_1207.png)


    60399 [Discriminator Loss: 0.170606, Acc.: 50.00%] [Generator Loss: 5.096484]
    


![png](out/output_16_1209.png)


    60499 [Discriminator Loss: 0.175672, Acc.: 50.00%] [Generator Loss: 4.142156]
    


![png](out/output_16_1211.png)


    60599 [Discriminator Loss: 0.167845, Acc.: 50.00%] [Generator Loss: 4.432037]
    


![png](out/output_16_1213.png)


    60699 [Discriminator Loss: 0.178921, Acc.: 50.00%] [Generator Loss: 5.090062]
    


![png](out/output_16_1215.png)


    60799 [Discriminator Loss: 0.171630, Acc.: 50.00%] [Generator Loss: 4.567879]
    


![png](out/output_16_1217.png)


    60899 [Discriminator Loss: 0.165445, Acc.: 50.00%] [Generator Loss: 4.626466]
    


![png](out/output_16_1219.png)


    60999 [Discriminator Loss: 0.175603, Acc.: 50.00%] [Generator Loss: 4.617632]
    


![png](out/output_16_1221.png)


    61099 [Discriminator Loss: 0.174698, Acc.: 50.00%] [Generator Loss: 4.139964]
    


![png](out/output_16_1223.png)


    61199 [Discriminator Loss: 0.177740, Acc.: 50.00%] [Generator Loss: 4.462716]
    


![png](out/output_16_1225.png)


    61299 [Discriminator Loss: 0.169536, Acc.: 50.00%] [Generator Loss: 4.117301]
    


![png](out/output_16_1227.png)


    61399 [Discriminator Loss: 0.164828, Acc.: 50.00%] [Generator Loss: 4.935591]
    


![png](out/output_16_1229.png)


    61499 [Discriminator Loss: 0.166414, Acc.: 50.00%] [Generator Loss: 4.248156]
    


![png](out/output_16_1231.png)


    61599 [Discriminator Loss: 0.166722, Acc.: 50.00%] [Generator Loss: 4.628221]
    


![png](out/output_16_1233.png)


    61699 [Discriminator Loss: 0.175511, Acc.: 50.00%] [Generator Loss: 4.513262]
    


![png](out/output_16_1235.png)


    61799 [Discriminator Loss: 0.172291, Acc.: 50.00%] [Generator Loss: 4.947976]
    


![png](out/output_16_1237.png)


    61899 [Discriminator Loss: 0.167224, Acc.: 50.00%] [Generator Loss: 4.897103]
    


![png](out/output_16_1239.png)


    61999 [Discriminator Loss: 0.165744, Acc.: 50.00%] [Generator Loss: 4.465592]
    


![png](out/output_16_1241.png)


    62099 [Discriminator Loss: 0.174781, Acc.: 50.00%] [Generator Loss: 3.955094]
    


![png](out/output_16_1243.png)


    62199 [Discriminator Loss: 0.175927, Acc.: 50.00%] [Generator Loss: 4.658178]
    


![png](out/output_16_1245.png)


    62299 [Discriminator Loss: 0.173974, Acc.: 50.00%] [Generator Loss: 3.961659]
    


![png](out/output_16_1247.png)


    62399 [Discriminator Loss: 0.169393, Acc.: 50.00%] [Generator Loss: 4.531356]
    


![png](out/output_16_1249.png)


    62499 [Discriminator Loss: 0.170621, Acc.: 50.00%] [Generator Loss: 4.685771]
    


![png](out/output_16_1251.png)


    62599 [Discriminator Loss: 0.172102, Acc.: 50.00%] [Generator Loss: 4.006613]
    


![png](out/output_16_1253.png)


    62699 [Discriminator Loss: 0.194301, Acc.: 50.00%] [Generator Loss: 3.929378]
    


![png](out/output_16_1255.png)


    62799 [Discriminator Loss: 0.166910, Acc.: 50.00%] [Generator Loss: 4.424855]
    


![png](out/output_16_1257.png)


    62899 [Discriminator Loss: 0.172056, Acc.: 50.00%] [Generator Loss: 4.340014]
    


![png](out/output_16_1259.png)


    62999 [Discriminator Loss: 0.170506, Acc.: 50.00%] [Generator Loss: 4.798024]
    


![png](out/output_16_1261.png)


    63099 [Discriminator Loss: 0.169191, Acc.: 50.00%] [Generator Loss: 3.485797]
    


![png](out/output_16_1263.png)


    63199 [Discriminator Loss: 0.167270, Acc.: 50.00%] [Generator Loss: 4.424275]
    


![png](out/output_16_1265.png)


    63299 [Discriminator Loss: 0.169480, Acc.: 50.00%] [Generator Loss: 4.130389]
    


![png](out/output_16_1267.png)


    63399 [Discriminator Loss: 0.172036, Acc.: 50.00%] [Generator Loss: 4.167107]
    


![png](out/output_16_1269.png)


    63499 [Discriminator Loss: 0.171768, Acc.: 50.00%] [Generator Loss: 4.070073]
    


![png](out/output_16_1271.png)


    63599 [Discriminator Loss: 0.168135, Acc.: 50.00%] [Generator Loss: 3.978767]
    


![png](out/output_16_1273.png)


    63699 [Discriminator Loss: 0.165909, Acc.: 50.00%] [Generator Loss: 4.103577]
    


![png](out/output_16_1275.png)


    63799 [Discriminator Loss: 0.171579, Acc.: 50.00%] [Generator Loss: 5.156585]
    


![png](out/output_16_1277.png)


    63899 [Discriminator Loss: 0.166584, Acc.: 50.00%] [Generator Loss: 4.313296]
    


![png](out/output_16_1279.png)


    63999 [Discriminator Loss: 0.166906, Acc.: 50.00%] [Generator Loss: 5.110004]
    


![png](out/output_16_1281.png)


    64099 [Discriminator Loss: 0.177400, Acc.: 50.00%] [Generator Loss: 4.041461]
    


![png](out/output_16_1283.png)


    64199 [Discriminator Loss: 0.168497, Acc.: 50.00%] [Generator Loss: 4.456689]
    


![png](out/output_16_1285.png)


    64299 [Discriminator Loss: 0.170919, Acc.: 50.00%] [Generator Loss: 4.247716]
    


![png](out/output_16_1287.png)


    64399 [Discriminator Loss: 0.169906, Acc.: 50.00%] [Generator Loss: 4.588689]
    


![png](out/output_16_1289.png)


    64499 [Discriminator Loss: 0.181114, Acc.: 50.00%] [Generator Loss: 4.616738]
    


![png](out/output_16_1291.png)


    64599 [Discriminator Loss: 0.165722, Acc.: 50.00%] [Generator Loss: 4.071700]
    


![png](out/output_16_1293.png)


    64699 [Discriminator Loss: 0.169805, Acc.: 50.00%] [Generator Loss: 4.541810]
    


![png](out/output_16_1295.png)


    64799 [Discriminator Loss: 0.194790, Acc.: 50.00%] [Generator Loss: 4.238873]
    


![png](out/output_16_1297.png)


    64899 [Discriminator Loss: 0.170307, Acc.: 50.00%] [Generator Loss: 4.461103]
    


![png](out/output_16_1299.png)


    64999 [Discriminator Loss: 0.167029, Acc.: 50.00%] [Generator Loss: 4.088603]
    


![png](out/output_16_1301.png)


    65099 [Discriminator Loss: 0.165794, Acc.: 50.00%] [Generator Loss: 4.625719]
    


![png](out/output_16_1303.png)


    65199 [Discriminator Loss: 0.169525, Acc.: 50.00%] [Generator Loss: 4.656816]
    


![png](out/output_16_1305.png)


    65299 [Discriminator Loss: 0.168301, Acc.: 50.00%] [Generator Loss: 4.318894]
    


![png](out/output_16_1307.png)


    65399 [Discriminator Loss: 0.179948, Acc.: 50.00%] [Generator Loss: 4.373448]
    


![png](out/output_16_1309.png)


    65499 [Discriminator Loss: 0.169976, Acc.: 50.00%] [Generator Loss: 3.641737]
    


![png](out/output_16_1311.png)


    65599 [Discriminator Loss: 0.175883, Acc.: 50.00%] [Generator Loss: 3.697480]
    


![png](out/output_16_1313.png)


    65699 [Discriminator Loss: 0.166522, Acc.: 50.00%] [Generator Loss: 4.207835]
    


![png](out/output_16_1315.png)


    65799 [Discriminator Loss: 0.165440, Acc.: 50.00%] [Generator Loss: 4.050221]
    


![png](out/output_16_1317.png)


    65899 [Discriminator Loss: 0.189055, Acc.: 50.00%] [Generator Loss: 4.602151]
    


![png](out/output_16_1319.png)


    65999 [Discriminator Loss: 0.179398, Acc.: 50.00%] [Generator Loss: 4.267935]
    


![png](out/output_16_1321.png)


    66099 [Discriminator Loss: 0.167031, Acc.: 50.00%] [Generator Loss: 4.333798]
    


![png](out/output_16_1323.png)


    66199 [Discriminator Loss: 0.165650, Acc.: 50.00%] [Generator Loss: 4.144822]
    


![png](out/output_16_1325.png)


    66299 [Discriminator Loss: 0.196881, Acc.: 50.00%] [Generator Loss: 4.883131]
    


![png](out/output_16_1327.png)


    66399 [Discriminator Loss: 0.171132, Acc.: 50.00%] [Generator Loss: 4.160036]
    


![png](out/output_16_1329.png)


    66499 [Discriminator Loss: 0.174217, Acc.: 50.00%] [Generator Loss: 5.183804]
    


![png](out/output_16_1331.png)


    66599 [Discriminator Loss: 0.168302, Acc.: 50.00%] [Generator Loss: 4.679398]
    


![png](out/output_16_1333.png)


    66699 [Discriminator Loss: 0.173234, Acc.: 50.00%] [Generator Loss: 4.863267]
    


![png](out/output_16_1335.png)


    66799 [Discriminator Loss: 0.166852, Acc.: 50.00%] [Generator Loss: 4.500232]
    


![png](out/output_16_1337.png)


    66899 [Discriminator Loss: 0.173361, Acc.: 50.00%] [Generator Loss: 4.627821]
    


![png](out/output_16_1339.png)


    66999 [Discriminator Loss: 0.169380, Acc.: 50.00%] [Generator Loss: 4.406913]
    


![png](out/output_16_1341.png)


    67099 [Discriminator Loss: 0.169738, Acc.: 50.00%] [Generator Loss: 4.663369]
    


![png](out/output_16_1343.png)


    67199 [Discriminator Loss: 0.168382, Acc.: 50.00%] [Generator Loss: 4.383565]
    


![png](out/output_16_1345.png)


    67299 [Discriminator Loss: 0.190121, Acc.: 50.00%] [Generator Loss: 5.018877]
    


![png](out/output_16_1347.png)


    67399 [Discriminator Loss: 0.181190, Acc.: 50.00%] [Generator Loss: 4.830857]
    


![png](out/output_16_1349.png)


    67499 [Discriminator Loss: 0.165401, Acc.: 50.00%] [Generator Loss: 4.926127]
    


![png](out/output_16_1351.png)


    67599 [Discriminator Loss: 0.172890, Acc.: 50.00%] [Generator Loss: 4.646172]
    


![png](out/output_16_1353.png)


    67699 [Discriminator Loss: 0.193707, Acc.: 46.88%] [Generator Loss: 4.241987]
    


![png](out/output_16_1355.png)


    67799 [Discriminator Loss: 0.180416, Acc.: 50.00%] [Generator Loss: 5.126523]
    


![png](out/output_16_1357.png)


    67899 [Discriminator Loss: 0.167107, Acc.: 50.00%] [Generator Loss: 3.874007]
    


![png](out/output_16_1359.png)


    67999 [Discriminator Loss: 0.169201, Acc.: 50.00%] [Generator Loss: 4.530756]
    


![png](out/output_16_1361.png)


    68099 [Discriminator Loss: 0.167073, Acc.: 50.00%] [Generator Loss: 4.171018]
    


![png](out/output_16_1363.png)


    68199 [Discriminator Loss: 0.169188, Acc.: 50.00%] [Generator Loss: 4.406073]
    


![png](out/output_16_1365.png)


    68299 [Discriminator Loss: 0.166762, Acc.: 50.00%] [Generator Loss: 4.119192]
    


![png](out/output_16_1367.png)


    68399 [Discriminator Loss: 0.166087, Acc.: 50.00%] [Generator Loss: 4.501979]
    


![png](out/output_16_1369.png)


    68499 [Discriminator Loss: 0.167458, Acc.: 50.00%] [Generator Loss: 3.891165]
    


![png](out/output_16_1371.png)


    68599 [Discriminator Loss: 0.220542, Acc.: 50.00%] [Generator Loss: 4.025823]
    


![png](out/output_16_1373.png)


    68699 [Discriminator Loss: 0.173092, Acc.: 50.00%] [Generator Loss: 4.461297]
    


![png](out/output_16_1375.png)


    68799 [Discriminator Loss: 0.182430, Acc.: 50.00%] [Generator Loss: 4.529753]
    


![png](out/output_16_1377.png)


    68899 [Discriminator Loss: 0.167769, Acc.: 50.00%] [Generator Loss: 4.643756]
    


![png](out/output_16_1379.png)


    68999 [Discriminator Loss: 0.165248, Acc.: 50.00%] [Generator Loss: 4.787611]
    


![png](out/output_16_1381.png)


    69099 [Discriminator Loss: 0.165874, Acc.: 50.00%] [Generator Loss: 4.810405]
    


![png](out/output_16_1383.png)


    69199 [Discriminator Loss: 0.170205, Acc.: 50.00%] [Generator Loss: 4.844710]
    


![png](out/output_16_1385.png)


    69299 [Discriminator Loss: 0.168590, Acc.: 50.00%] [Generator Loss: 4.701083]
    


![png](out/output_16_1387.png)


    69399 [Discriminator Loss: 0.170270, Acc.: 50.00%] [Generator Loss: 4.188552]
    


![png](out/output_16_1389.png)


    69499 [Discriminator Loss: 0.168285, Acc.: 50.00%] [Generator Loss: 4.039039]
    


![png](out/output_16_1391.png)


    69599 [Discriminator Loss: 0.167099, Acc.: 50.00%] [Generator Loss: 3.901492]
    


![png](out/output_16_1393.png)


    69699 [Discriminator Loss: 0.169694, Acc.: 50.00%] [Generator Loss: 4.779764]
    


![png](out/output_16_1395.png)


    69799 [Discriminator Loss: 0.169683, Acc.: 50.00%] [Generator Loss: 4.836380]
    


![png](out/output_16_1397.png)


    69899 [Discriminator Loss: 0.175960, Acc.: 50.00%] [Generator Loss: 4.375410]
    


![png](out/output_16_1399.png)


    69999 [Discriminator Loss: 0.181891, Acc.: 50.00%] [Generator Loss: 5.053218]
    


![png](out/output_16_1401.png)


    70099 [Discriminator Loss: 0.181710, Acc.: 50.00%] [Generator Loss: 4.954476]
    


![png](out/output_16_1403.png)


    70199 [Discriminator Loss: 0.170512, Acc.: 50.00%] [Generator Loss: 4.165116]
    


![png](out/output_16_1405.png)


    70299 [Discriminator Loss: 0.169179, Acc.: 50.00%] [Generator Loss: 4.428484]
    


![png](out/output_16_1407.png)


    70399 [Discriminator Loss: 0.165784, Acc.: 50.00%] [Generator Loss: 4.197246]
    


![png](out/output_16_1409.png)


    70499 [Discriminator Loss: 0.165434, Acc.: 50.00%] [Generator Loss: 5.277501]
    


![png](out/output_16_1411.png)


    70599 [Discriminator Loss: 0.173781, Acc.: 50.00%] [Generator Loss: 4.027075]
    


![png](out/output_16_1413.png)


    70699 [Discriminator Loss: 0.180100, Acc.: 50.00%] [Generator Loss: 4.230900]
    


![png](out/output_16_1415.png)


    70799 [Discriminator Loss: 0.174753, Acc.: 50.00%] [Generator Loss: 4.726176]
    


![png](out/output_16_1417.png)


    70899 [Discriminator Loss: 0.167475, Acc.: 50.00%] [Generator Loss: 4.583385]
    


![png](out/output_16_1419.png)


    70999 [Discriminator Loss: 0.169029, Acc.: 50.00%] [Generator Loss: 4.658963]
    


![png](out/output_16_1421.png)


    71099 [Discriminator Loss: 0.167963, Acc.: 50.00%] [Generator Loss: 4.779940]
    


![png](out/output_16_1423.png)


    71199 [Discriminator Loss: 0.167022, Acc.: 50.00%] [Generator Loss: 4.999280]
    


![png](out/output_16_1425.png)


    71299 [Discriminator Loss: 0.172727, Acc.: 50.00%] [Generator Loss: 4.805186]
    


![png](out/output_16_1427.png)


    71399 [Discriminator Loss: 0.182839, Acc.: 50.00%] [Generator Loss: 4.206763]
    


![png](out/output_16_1429.png)


    71499 [Discriminator Loss: 0.166700, Acc.: 50.00%] [Generator Loss: 4.235759]
    


![png](out/output_16_1431.png)


    71599 [Discriminator Loss: 0.171646, Acc.: 50.00%] [Generator Loss: 4.527113]
    


![png](out/output_16_1433.png)


    71699 [Discriminator Loss: 0.169280, Acc.: 50.00%] [Generator Loss: 3.765200]
    


![png](out/output_16_1435.png)


    71799 [Discriminator Loss: 0.168068, Acc.: 50.00%] [Generator Loss: 4.506386]
    


![png](out/output_16_1437.png)


    71899 [Discriminator Loss: 0.168626, Acc.: 50.00%] [Generator Loss: 4.748796]
    


![png](out/output_16_1439.png)


    71999 [Discriminator Loss: 0.169690, Acc.: 50.00%] [Generator Loss: 4.371393]
    


![png](out/output_16_1441.png)


    72099 [Discriminator Loss: 0.167372, Acc.: 50.00%] [Generator Loss: 3.962623]
    


![png](out/output_16_1443.png)


    72199 [Discriminator Loss: 0.171933, Acc.: 50.00%] [Generator Loss: 4.626905]
    


![png](out/output_16_1445.png)


    72299 [Discriminator Loss: 0.174476, Acc.: 50.00%] [Generator Loss: 4.284109]
    


![png](out/output_16_1447.png)


    72399 [Discriminator Loss: 0.169022, Acc.: 50.00%] [Generator Loss: 4.486643]
    


![png](out/output_16_1449.png)


    72499 [Discriminator Loss: 0.169151, Acc.: 50.00%] [Generator Loss: 5.352050]
    


![png](out/output_16_1451.png)


    72599 [Discriminator Loss: 0.169377, Acc.: 50.00%] [Generator Loss: 5.206195]
    


![png](out/output_16_1453.png)


    72699 [Discriminator Loss: 0.171464, Acc.: 50.00%] [Generator Loss: 4.250089]
    


![png](out/output_16_1455.png)


    72799 [Discriminator Loss: 0.170917, Acc.: 50.00%] [Generator Loss: 3.682859]
    


![png](out/output_16_1457.png)


    72899 [Discriminator Loss: 0.169559, Acc.: 50.00%] [Generator Loss: 5.034130]
    


![png](out/output_16_1459.png)


    72999 [Discriminator Loss: 0.175062, Acc.: 50.00%] [Generator Loss: 4.818799]
    


![png](out/output_16_1461.png)


    73099 [Discriminator Loss: 0.164581, Acc.: 50.00%] [Generator Loss: 5.340331]
    


![png](out/output_16_1463.png)


    73199 [Discriminator Loss: 0.169246, Acc.: 50.00%] [Generator Loss: 4.834278]
    


![png](out/output_16_1465.png)


    73299 [Discriminator Loss: 0.175382, Acc.: 50.00%] [Generator Loss: 4.870599]
    


![png](out/output_16_1467.png)


    73399 [Discriminator Loss: 0.172902, Acc.: 50.00%] [Generator Loss: 4.886008]
    


![png](out/output_16_1469.png)


    73499 [Discriminator Loss: 0.168509, Acc.: 50.00%] [Generator Loss: 4.240401]
    


![png](out/output_16_1471.png)


    73599 [Discriminator Loss: 0.167507, Acc.: 50.00%] [Generator Loss: 4.540308]
    


![png](out/output_16_1473.png)


    73699 [Discriminator Loss: 0.168878, Acc.: 50.00%] [Generator Loss: 4.807339]
    


![png](out/output_16_1475.png)


    73799 [Discriminator Loss: 0.168885, Acc.: 50.00%] [Generator Loss: 5.247983]
    


![png](out/output_16_1477.png)


    73899 [Discriminator Loss: 0.182180, Acc.: 50.00%] [Generator Loss: 4.968494]
    


![png](out/output_16_1479.png)


    73999 [Discriminator Loss: 0.174679, Acc.: 50.00%] [Generator Loss: 5.440151]
    


![png](out/output_16_1481.png)


    74099 [Discriminator Loss: 0.168411, Acc.: 50.00%] [Generator Loss: 5.291676]
    


![png](out/output_16_1483.png)


    74199 [Discriminator Loss: 0.169148, Acc.: 50.00%] [Generator Loss: 4.799872]
    


![png](out/output_16_1485.png)


    74299 [Discriminator Loss: 0.166193, Acc.: 50.00%] [Generator Loss: 5.101077]
    


![png](out/output_16_1487.png)


    74399 [Discriminator Loss: 0.172780, Acc.: 50.00%] [Generator Loss: 5.277898]
    


![png](out/output_16_1489.png)


    74499 [Discriminator Loss: 0.167955, Acc.: 50.00%] [Generator Loss: 4.200174]
    


![png](out/output_16_1491.png)


    74599 [Discriminator Loss: 0.168300, Acc.: 50.00%] [Generator Loss: 5.145145]
    


![png](out/output_16_1493.png)


    74699 [Discriminator Loss: 0.171464, Acc.: 50.00%] [Generator Loss: 4.675567]
    


![png](out/output_16_1495.png)


    74799 [Discriminator Loss: 0.168079, Acc.: 50.00%] [Generator Loss: 4.997682]
    


![png](out/output_16_1497.png)


    74899 [Discriminator Loss: 0.176168, Acc.: 50.00%] [Generator Loss: 5.349506]
    


![png](out/output_16_1499.png)


    74999 [Discriminator Loss: 0.171028, Acc.: 50.00%] [Generator Loss: 4.294617]
    


![png](out/output_16_1501.png)


    75099 [Discriminator Loss: 0.168677, Acc.: 50.00%] [Generator Loss: 6.047255]
    


![png](out/output_16_1503.png)


    75199 [Discriminator Loss: 0.170488, Acc.: 50.00%] [Generator Loss: 5.182962]
    


![png](out/output_16_1505.png)


    75299 [Discriminator Loss: 0.171777, Acc.: 50.00%] [Generator Loss: 5.871482]
    


![png](out/output_16_1507.png)


    75399 [Discriminator Loss: 0.175347, Acc.: 50.00%] [Generator Loss: 5.146563]
    


![png](out/output_16_1509.png)


    75499 [Discriminator Loss: 0.184181, Acc.: 50.00%] [Generator Loss: 5.024910]
    


![png](out/output_16_1511.png)


    75599 [Discriminator Loss: 0.186701, Acc.: 50.00%] [Generator Loss: 5.120627]
    


![png](out/output_16_1513.png)


    75699 [Discriminator Loss: 0.167293, Acc.: 50.00%] [Generator Loss: 4.896142]
    


![png](out/output_16_1515.png)


    75799 [Discriminator Loss: 0.169389, Acc.: 50.00%] [Generator Loss: 4.166840]
    


![png](out/output_16_1517.png)


    75899 [Discriminator Loss: 0.192229, Acc.: 50.00%] [Generator Loss: 4.874662]
    


![png](out/output_16_1519.png)


    75999 [Discriminator Loss: 0.175628, Acc.: 50.00%] [Generator Loss: 5.562210]
    


![png](out/output_16_1521.png)


    76099 [Discriminator Loss: 0.176746, Acc.: 50.00%] [Generator Loss: 4.950003]
    


![png](out/output_16_1523.png)


    76199 [Discriminator Loss: 0.177240, Acc.: 50.00%] [Generator Loss: 5.619504]
    


![png](out/output_16_1525.png)


    76299 [Discriminator Loss: 0.180336, Acc.: 50.00%] [Generator Loss: 5.111029]
    


![png](out/output_16_1527.png)


    76399 [Discriminator Loss: 0.182012, Acc.: 50.00%] [Generator Loss: 4.090619]
    


![png](out/output_16_1529.png)


    76499 [Discriminator Loss: 0.172024, Acc.: 50.00%] [Generator Loss: 5.208459]
    


![png](out/output_16_1531.png)


    76599 [Discriminator Loss: 0.170915, Acc.: 50.00%] [Generator Loss: 5.769229]
    


![png](out/output_16_1533.png)


    76699 [Discriminator Loss: 0.210284, Acc.: 50.00%] [Generator Loss: 4.445591]
    


![png](out/output_16_1535.png)


    76799 [Discriminator Loss: 0.174964, Acc.: 50.00%] [Generator Loss: 5.791089]
    


![png](out/output_16_1537.png)


    76899 [Discriminator Loss: 0.172569, Acc.: 50.00%] [Generator Loss: 5.358300]
    


![png](out/output_16_1539.png)


    76999 [Discriminator Loss: 0.194332, Acc.: 50.00%] [Generator Loss: 4.333587]
    


![png](out/output_16_1541.png)


    77099 [Discriminator Loss: 0.185006, Acc.: 50.00%] [Generator Loss: 4.450814]
    


![png](out/output_16_1543.png)


    77199 [Discriminator Loss: 0.171273, Acc.: 50.00%] [Generator Loss: 5.950989]
    


![png](out/output_16_1545.png)


    77299 [Discriminator Loss: 0.172406, Acc.: 50.00%] [Generator Loss: 4.115047]
    


![png](out/output_16_1547.png)


    77399 [Discriminator Loss: 0.185894, Acc.: 50.00%] [Generator Loss: 4.860103]
    


![png](out/output_16_1549.png)


    77499 [Discriminator Loss: 0.166369, Acc.: 50.00%] [Generator Loss: 4.199191]
    


![png](out/output_16_1551.png)


    77599 [Discriminator Loss: 0.220327, Acc.: 50.00%] [Generator Loss: 4.987281]
    


![png](out/output_16_1553.png)


    77699 [Discriminator Loss: 0.194220, Acc.: 46.88%] [Generator Loss: 5.241277]
    


![png](out/output_16_1555.png)


    77799 [Discriminator Loss: 0.170834, Acc.: 50.00%] [Generator Loss: 6.200205]
    


![png](out/output_16_1557.png)


    77899 [Discriminator Loss: 0.176815, Acc.: 50.00%] [Generator Loss: 5.020109]
    


![png](out/output_16_1559.png)


    77999 [Discriminator Loss: 0.171738, Acc.: 50.00%] [Generator Loss: 5.204919]
    


![png](out/output_16_1561.png)


    78099 [Discriminator Loss: 0.168846, Acc.: 50.00%] [Generator Loss: 4.562171]
    


![png](out/output_16_1563.png)


    78199 [Discriminator Loss: 0.169669, Acc.: 50.00%] [Generator Loss: 5.398727]
    


![png](out/output_16_1565.png)


    78299 [Discriminator Loss: 0.167730, Acc.: 50.00%] [Generator Loss: 5.280585]
    


![png](out/output_16_1567.png)


    78399 [Discriminator Loss: 0.179495, Acc.: 50.00%] [Generator Loss: 4.631873]
    


![png](out/output_16_1569.png)


    78499 [Discriminator Loss: 0.174847, Acc.: 50.00%] [Generator Loss: 5.670874]
    


![png](out/output_16_1571.png)


    78599 [Discriminator Loss: 0.171172, Acc.: 50.00%] [Generator Loss: 4.855613]
    


![png](out/output_16_1573.png)


    78699 [Discriminator Loss: 0.171262, Acc.: 50.00%] [Generator Loss: 5.093653]
    


![png](out/output_16_1575.png)


    78799 [Discriminator Loss: 0.178683, Acc.: 50.00%] [Generator Loss: 5.571439]
    


![png](out/output_16_1577.png)


    78899 [Discriminator Loss: 0.176028, Acc.: 50.00%] [Generator Loss: 4.439073]
    


![png](out/output_16_1579.png)


    78999 [Discriminator Loss: 0.166630, Acc.: 50.00%] [Generator Loss: 4.695292]
    


![png](out/output_16_1581.png)


    79099 [Discriminator Loss: 0.192098, Acc.: 50.00%] [Generator Loss: 4.548067]
    


![png](out/output_16_1583.png)


    79199 [Discriminator Loss: 0.171895, Acc.: 50.00%] [Generator Loss: 4.553526]
    


![png](out/output_16_1585.png)


    79299 [Discriminator Loss: 0.169224, Acc.: 50.00%] [Generator Loss: 4.779194]
    


![png](out/output_16_1587.png)


    79399 [Discriminator Loss: 0.170708, Acc.: 50.00%] [Generator Loss: 4.525454]
    


![png](out/output_16_1589.png)


    79499 [Discriminator Loss: 0.175743, Acc.: 50.00%] [Generator Loss: 5.326336]
    


![png](out/output_16_1591.png)


    79599 [Discriminator Loss: 0.167240, Acc.: 50.00%] [Generator Loss: 4.048450]
    


![png](out/output_16_1593.png)


    79699 [Discriminator Loss: 0.167120, Acc.: 50.00%] [Generator Loss: 4.625924]
    
