

```python
import os
os.environ['KERAS_BACKEND']='tensorflow' 
```


```python
import keras.backend as K
if os.environ['KERAS_BACKEND'] =='tensorflow':
    channel_axis=1
    K.set_image_data_format('channels_first')
    channel_first = True
else:
    K.set_image_data_format('channels_last')
    channel_axis=-1
    channel_first = False
```

    Using TensorFlow backend.
    C:\machine_study\Python\Anaconda3\lib\site-packages\h5py\__init__.py:34: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    


```python
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dropout
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers import Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import relu
from keras.initializers import RandomNormal
```


```python
# Weights initializations
# bias are initailized as 0
def __conv_init(a):
    print("conv_init", a)
    k = RandomNormal(0, 0.02)(a) # for convolution kernel
    k.conv_weight = True    
    return k
conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02) # for batch normalization

```


```python
# HACK speed up theano
if K._BACKEND == 'theano':
    import keras.backend.theano_backend as theano_backend
    def _preprocess_conv2d_kernel(kernel, data_format):
        #return kernel
        if hasattr(kernel, "original"):
            print("use original")
            return kernel.original
        elif hasattr(kernel, '_keras_shape'):
            s = kernel._keras_shape
            print("use reshape",s)
            kernel = kernel.reshape((s[3], s[2],s[0], s[1]))
        else:
            kernel = kernel.dimshuffle((3, 2, 0, 1))
        return kernel
    theano_backend._preprocess_conv2d_kernel = _preprocess_conv2d_kernel
```


```python
# Basic discriminator
def conv2d(f, *a, **k):
    return Conv2D(f, kernel_initializer = conv_init, *a, **k)
def batchnorm():
    return BatchNormalization(momentum=0.9, axis=channel_axis, epsilon=1.01e-5,
                                   gamma_initializer = gamma_init)
def BASIC_D(nc_in, ndf, max_layers=3, use_sigmoid=True):
    """DCGAN_D(nc, ndf, max_layers=3)
       nc: channels
       ndf: filters of the first layer
       max_layers: max hidden layers
    """    
    if channel_first:
        input_a =  Input(shape=(nc_in, None, None))
    else:
        input_a = Input(shape=(None, None, nc_in))
    _ = input_a
    _ = conv2d(ndf, kernel_size=4, strides=2, padding="same", name = 'First') (_)
    _ = LeakyReLU(alpha=0.2)(_)
    
    for layer in range(1, max_layers):        
        out_feat = ndf * min(2**layer, 8)
        _ = conv2d(out_feat, kernel_size=4, strides=2, padding="same", 
                   use_bias=False, name = 'pyramid.{0}'.format(layer)             
                        ) (_)
        _ = batchnorm()(_, training=1)        
        _ = LeakyReLU(alpha=0.2)(_)
    
    out_feat = ndf*min(2**max_layers, 8)
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(out_feat, kernel_size=4,  use_bias=False, name = 'pyramid_last') (_)
    _ = batchnorm()(_, training=1)
    _ = LeakyReLU(alpha=0.2)(_)
    
    # final layer
    _ = ZeroPadding2D(1)(_)
    _ = conv2d(1, kernel_size=4, name = 'final'.format(out_feat, 1), 
               activation = "sigmoid" if use_sigmoid else None) (_)    
    return Model(inputs=[input_a], outputs=_)
```


```python
def UNET_G(isize, nc_in=3, nc_out=3, ngf=64, fixed_input_size=True):    
    max_nf = 8*ngf    
    def block(x, s, nf_in, use_batchnorm=True, nf_out=None, nf_next=None):
        # print("block",x,s,nf_in, use_batchnorm, nf_out, nf_next)
        assert s>=2 and s%2==0
        if nf_next is None:
            nf_next = min(nf_in*2, max_nf)
        if nf_out is None:
            nf_out = nf_in
        x = conv2d(nf_next, kernel_size=4, strides=2, use_bias=(not (use_batchnorm and s>2)),
                   padding="same", name = 'conv_{0}'.format(s)) (x)
        if s>2:
            if use_batchnorm:
                x = batchnorm()(x, training=1)
            x2 = LeakyReLU(alpha=0.2)(x)
            x2 = block(x2, s//2, nf_next)
            x = Concatenate(axis=channel_axis)([x, x2])            
        x = Activation("relu")(x)
        x = Conv2DTranspose(nf_out, kernel_size=4, strides=2, use_bias=not use_batchnorm,
                            kernel_initializer = conv_init,          
                            name = 'convt.{0}'.format(s))(x)        
        x = Cropping2D(1)(x)
        if use_batchnorm:
            x = batchnorm()(x, training=1)
        if s <=8:
            x = Dropout(0.5)(x, training=1)
        return x
    
    s = isize if fixed_input_size else None
    if channel_first:
        _ = inputs = Input(shape=(nc_in, s, s))
    else:
        _ = inputs = Input(shape=(s, s, nc_in))        
    _ = block(_, isize, nc_in, False, nf_out=nc_out, nf_next=ngf)
    _ = Activation('tanh')(_)
    return Model(inputs=inputs, outputs=[_])
```


```python
nc_in = 3
nc_out = 3
ngf = 16
ndf = 16
use_lsgan = True
λ = 10 if use_lsgan else 100

loadSize = 56
imageSize = 32
batchSize = 1
lrD = 2e-4
lrG = 2e-4
```


```python
netDA = BASIC_D(nc_in, ndf, use_sigmoid = not use_lsgan)
netDB = BASIC_D(nc_out, ndf, use_sigmoid = not use_lsgan)
netDA.summary()
```

    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_5 (InputLayer)         (None, 3, None, None)     0         
    _________________________________________________________________
    First (Conv2D)               (None, 16, None, None)    784       
    _________________________________________________________________
    leaky_re_lu_19 (LeakyReLU)   (None, 16, None, None)    0         
    _________________________________________________________________
    pyramid.1 (Conv2D)           (None, 32, None, None)    8192      
    _________________________________________________________________
    batch_normalization_25 (Batc (None, 32, None, None)    128       
    _________________________________________________________________
    leaky_re_lu_20 (LeakyReLU)   (None, 32, None, None)    0         
    _________________________________________________________________
    pyramid.2 (Conv2D)           (None, 64, None, None)    32768     
    _________________________________________________________________
    batch_normalization_26 (Batc (None, 64, None, None)    256       
    _________________________________________________________________
    leaky_re_lu_21 (LeakyReLU)   (None, 64, None, None)    0         
    _________________________________________________________________
    zero_padding2d_5 (ZeroPaddin (None, 64, None, None)    0         
    _________________________________________________________________
    pyramid_last (Conv2D)        (None, 128, None, None)   131072    
    _________________________________________________________________
    batch_normalization_27 (Batc (None, 128, None, None)   512       
    _________________________________________________________________
    leaky_re_lu_22 (LeakyReLU)   (None, 128, None, None)   0         
    _________________________________________________________________
    zero_padding2d_6 (ZeroPaddin (None, 128, None, None)   0         
    _________________________________________________________________
    final (Conv2D)               (None, 1, None, None)     2049      
    =================================================================
    Total params: 175,761
    Trainable params: 175,313
    Non-trainable params: 448
    _________________________________________________________________
    


```python
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot


netGB = UNET_G(imageSize, nc_in, nc_out, ngf)
netGA = UNET_G(imageSize, nc_out, nc_in, ngf)
#SVG(model_to_dot(netG, show_shapes=True).create(prog='dot', format='svg'))
netGA.summary()

```

    ____________________________________________________________________________________________________
    Layer (type)                     Output Shape          Param #     Connected to                     
    ====================================================================================================
    input_8 (InputLayer)             (None, 3, 32, 32)     0                                            
    ____________________________________________________________________________________________________
    conv_32 (Conv2D)                 (None, 16, 16, 16)    784         input_8[0][0]                    
    ____________________________________________________________________________________________________
    leaky_re_lu_31 (LeakyReLU)       (None, 16, 16, 16)    0           conv_32[0][0]                    
    ____________________________________________________________________________________________________
    conv_16 (Conv2D)                 (None, 32, 8, 8)      8192        leaky_re_lu_31[0][0]             
    ____________________________________________________________________________________________________
    batch_normalization_38 (BatchNor (None, 32, 8, 8)      128         conv_16[0][0]                    
    ____________________________________________________________________________________________________
    leaky_re_lu_32 (LeakyReLU)       (None, 32, 8, 8)      0           batch_normalization_38[0][0]     
    ____________________________________________________________________________________________________
    conv_8 (Conv2D)                  (None, 64, 4, 4)      32768       leaky_re_lu_32[0][0]             
    ____________________________________________________________________________________________________
    batch_normalization_39 (BatchNor (None, 64, 4, 4)      256         conv_8[0][0]                     
    ____________________________________________________________________________________________________
    leaky_re_lu_33 (LeakyReLU)       (None, 64, 4, 4)      0           batch_normalization_39[0][0]     
    ____________________________________________________________________________________________________
    conv_4 (Conv2D)                  (None, 128, 2, 2)     131072      leaky_re_lu_33[0][0]             
    ____________________________________________________________________________________________________
    batch_normalization_40 (BatchNor (None, 128, 2, 2)     512         conv_4[0][0]                     
    ____________________________________________________________________________________________________
    leaky_re_lu_34 (LeakyReLU)       (None, 128, 2, 2)     0           batch_normalization_40[0][0]     
    ____________________________________________________________________________________________________
    conv_2 (Conv2D)                  (None, 128, 1, 1)     262272      leaky_re_lu_34[0][0]             
    ____________________________________________________________________________________________________
    activation_21 (Activation)       (None, 128, 1, 1)     0           conv_2[0][0]                     
    ____________________________________________________________________________________________________
    convt.2 (Conv2DTranspose)        (None, 128, 4, 4)     262144      activation_21[0][0]              
    ____________________________________________________________________________________________________
    cropping2d_18 (Cropping2D)       (None, 128, 2, 2)     0           convt.2[0][0]                    
    ____________________________________________________________________________________________________
    batch_normalization_41 (BatchNor (None, 128, 2, 2)     512         cropping2d_18[0][0]              
    ____________________________________________________________________________________________________
    dropout_10 (Dropout)             (None, 128, 2, 2)     0           batch_normalization_41[0][0]     
    ____________________________________________________________________________________________________
    concatenate_15 (Concatenate)     (None, 256, 2, 2)     0           batch_normalization_40[0][0]     
                                                                       dropout_10[0][0]                 
    ____________________________________________________________________________________________________
    activation_22 (Activation)       (None, 256, 2, 2)     0           concatenate_15[0][0]             
    ____________________________________________________________________________________________________
    convt.4 (Conv2DTranspose)        (None, 64, 6, 6)      262144      activation_22[0][0]              
    ____________________________________________________________________________________________________
    cropping2d_19 (Cropping2D)       (None, 64, 4, 4)      0           convt.4[0][0]                    
    ____________________________________________________________________________________________________
    batch_normalization_42 (BatchNor (None, 64, 4, 4)      256         cropping2d_19[0][0]              
    ____________________________________________________________________________________________________
    dropout_11 (Dropout)             (None, 64, 4, 4)      0           batch_normalization_42[0][0]     
    ____________________________________________________________________________________________________
    concatenate_16 (Concatenate)     (None, 128, 4, 4)     0           batch_normalization_39[0][0]     
                                                                       dropout_11[0][0]                 
    ____________________________________________________________________________________________________
    activation_23 (Activation)       (None, 128, 4, 4)     0           concatenate_16[0][0]             
    ____________________________________________________________________________________________________
    convt.8 (Conv2DTranspose)        (None, 32, 10, 10)    65536       activation_23[0][0]              
    ____________________________________________________________________________________________________
    cropping2d_20 (Cropping2D)       (None, 32, 8, 8)      0           convt.8[0][0]                    
    ____________________________________________________________________________________________________
    batch_normalization_43 (BatchNor (None, 32, 8, 8)      128         cropping2d_20[0][0]              
    ____________________________________________________________________________________________________
    dropout_12 (Dropout)             (None, 32, 8, 8)      0           batch_normalization_43[0][0]     
    ____________________________________________________________________________________________________
    concatenate_17 (Concatenate)     (None, 64, 8, 8)      0           batch_normalization_38[0][0]     
                                                                       dropout_12[0][0]                 
    ____________________________________________________________________________________________________
    activation_24 (Activation)       (None, 64, 8, 8)      0           concatenate_17[0][0]             
    ____________________________________________________________________________________________________
    convt.16 (Conv2DTranspose)       (None, 16, 18, 18)    16384       activation_24[0][0]              
    ____________________________________________________________________________________________________
    cropping2d_21 (Cropping2D)       (None, 16, 16, 16)    0           convt.16[0][0]                   
    ____________________________________________________________________________________________________
    batch_normalization_44 (BatchNor (None, 16, 16, 16)    64          cropping2d_21[0][0]              
    ____________________________________________________________________________________________________
    concatenate_18 (Concatenate)     (None, 32, 16, 16)    0           conv_32[0][0]                    
                                                                       batch_normalization_44[0][0]     
    ____________________________________________________________________________________________________
    activation_25 (Activation)       (None, 32, 16, 16)    0           concatenate_18[0][0]             
    ____________________________________________________________________________________________________
    convt.32 (Conv2DTranspose)       (None, 3, 34, 34)     1539        activation_25[0][0]              
    ____________________________________________________________________________________________________
    cropping2d_22 (Cropping2D)       (None, 3, 32, 32)     0           convt.32[0][0]                   
    ____________________________________________________________________________________________________
    activation_26 (Activation)       (None, 3, 32, 32)     0           cropping2d_22[0][0]              
    ====================================================================================================
    Total params: 1,044,691
    Trainable params: 1,043,763
    Non-trainable params: 928
    ____________________________________________________________________________________________________
    


```python
from keras.optimizers import RMSprop, SGD, Adam
```


```python
if use_lsgan:
    loss_fn = lambda output, target : K.mean(K.abs(K.square(output-target)))
else:
    loss_fn = lambda output, target : -K.mean(K.log(output+1e-12)*target+K.log(1-output+1e-12)*(1-target))

def cycle_variables(netG1, netG2):
    real_input = netG1.inputs[0]
    fake_output = netG1.outputs[0]
    rec_input = netG2([fake_output])
    fn_generate = K.function([real_input], [fake_output, rec_input])
    return real_input, fake_output, rec_input, fn_generate

real_A, fake_B, rec_A, cycleA_generate = cycle_variables(netGB, netGA)
real_B, fake_A, rec_B, cycleB_generate = cycle_variables(netGA, netGB)
```


```python
def D_loss(netD, real, fake, rec):
    output_real = netD([real])
    output_fake = netD([fake])
    loss_D_real = loss_fn(output_real, K.ones_like(output_real))
    loss_D_fake = loss_fn(output_fake, K.zeros_like(output_fake))
    loss_G = loss_fn(output_fake, K.ones_like(output_fake))
    loss_D = loss_D_real+loss_D_fake
    loss_cyc = K.mean(K.abs(rec-real))
    return loss_D, loss_G, loss_cyc

loss_DA, loss_GA, loss_cycA = D_loss(netDA, real_A, fake_A, rec_A)
loss_DB, loss_GB, loss_cycB = D_loss(netDB, real_B, fake_B, rec_B)
loss_cyc = loss_cycA+loss_cycB
```


```python
loss_G = loss_GA+loss_GB+λ*loss_cyc
loss_D = loss_DA+loss_DB

weightsD = netDA.trainable_weights + netDB.trainable_weights
weightsG = netGA.trainable_weights + netGB.trainable_weights

training_updates = Adam(lr=lrD, beta_1=0.5).get_updates(weightsD,[],loss_D)
netD_train = K.function([real_A, real_B],[loss_DA/2, loss_DB/2], training_updates)
training_updates = Adam(lr=lrG, beta_1=0.5).get_updates(weightsG,[], loss_G)
netG_train = K.function([real_A, real_B], [loss_GA, loss_GB, loss_cyc], training_updates)
```


```python
from PIL import Image
import numpy as np
import glob
from random import randint, shuffle

def load_data(file_pattern):
    return glob.glob(file_pattern)

def read_image(fn):
    im = Image.open(fn).convert('RGB')
    im = im.resize( (loadSize, loadSize), Image.BILINEAR )
    arr = np.array(im)/255*2-1
    w1,w2 = (loadSize-imageSize)//2,(loadSize+imageSize)//2
    h1,h2 = w1,w2
    img = arr[h1:h2, w1:w2, :]
    if randint(0,1):
        img=img[:,::-1]
    if channel_first:        
        img = np.moveaxis(img, 2, 0)
    return img

train_A = load_data('../data/players/train/*.jpg')
train_B = load_data('../data/players/val/*.jpg')

assert len(train_A) and len(train_B)
```


```python
def minibatch(data, batchsize):
    length = len(data)
    epoch = i = 0
    tmpsize = None    
    while True:
        size = tmpsize if tmpsize else batchsize
        if i+size > length:
            shuffle(data)
            i = 0
            epoch+=1        
        rtn = [read_image(data[j]) for j in range(i,i+size)]
        i+=size
        tmpsize = yield epoch, np.float32(rtn)       

def minibatchAB(dataA, dataB, batchsize):
    batchA=minibatch(dataA, batchsize)
    batchB=minibatch(dataB, batchsize)
    tmpsize = None    
    while True:        
        ep1, A = batchA.send(tmpsize)
        ep2, B = batchB.send(tmpsize)
        tmpsize = yield max(ep1, ep2), A, B
```


```python
from IPython.display import display
def showX(X, rows=1):
    assert X.shape[0]%rows == 0
    int_X = ( (X+1)/2*255).clip(0,255).astype('uint8')
    if channel_first:
        int_X = np.moveaxis(int_X.reshape(-1,3,imageSize,imageSize), 1, 3)
    else:
        int_X = int_X.reshape(-1,imageSize,imageSize, 3)
    int_X = int_X.reshape(rows, -1, imageSize, imageSize,3).swapaxes(1,2).reshape(rows*imageSize,-1, 3)
    display(Image.fromarray(int_X))
```


```python
train_batch = minibatchAB(train_A, train_B, 12)

_, A, B = next(train_batch)
showX(A)
showX(B)
del train_batch, A, B
```


![png](output_17_0.png)



![png](output_17_1.png)



```python
def showG(A,B):
    assert A.shape==B.shape
    def G(fn_generate, X):
        r = np.array([fn_generate([X[i:i+1]]) for i in range(X.shape[0])])
        return r.swapaxes(0,1)[:,:,0]        
    rA = G(cycleA_generate, A)
    rB = G(cycleB_generate, B)
    arr = np.concatenate([A,B,rA[0],rB[0],rA[1],rB[1]])
    showX(arr, 3)
```


```python
import time
from IPython.display import clear_output
t0 = time.time()
niter = 50
gen_iterations = 0
epoch = 0
errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0

display_iters = 150
#val_batch = minibatch(valAB, 6, direction)
train_batch = minibatchAB(train_A, train_B, batchSize)
while epoch < niter: 
    epoch, A, B = next(train_batch)        
    errDA, errDB  = netD_train([A, B])
    errDA_sum +=errDA
    errDB_sum +=errDB

    # epoch, trainA, trainB = next(train_batch)
    errGA, errGB, errCyc = netG_train([A, B])
    errGA_sum += errGA
    errGB_sum += errGB
    errCyc_sum += errCyc
    gen_iterations+=1
    if gen_iterations%display_iters==0:
        #if gen_iterations%(5*display_iters)==0:
        clear_output()
        print('[%d/%d][%d] Loss_D: %f %f Loss_G: %f %f loss_cyc %f'
        % (epoch, niter, gen_iterations, errDA_sum/display_iters, errDB_sum/display_iters,
           errGA_sum/display_iters, errGB_sum/display_iters, 
           errCyc_sum/display_iters), time.time()-t0)
        _, A, B = train_batch.send(4)
        showG(A,B)        
        errCyc_sum = errGA_sum = errGB_sum = errDA_sum = errDB_sum = 0
```

    [44/50][1050] Loss_D: 0.033682 0.044071 Loss_G: 1.005236 0.969284 loss_cyc 0.634544 278.52128052711487
    


![png](output_19_1.png)










