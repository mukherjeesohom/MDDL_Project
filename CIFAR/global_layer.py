import tensorflow as tf
from tensorflow.keras import layers
import sys

from building_blocks import *


def get_init_block( planes, block_type = 'default'):
    if block_type == 'BasicBlock':
        init_h = BasicBlock(planes, planes)
    elif block_type == 'Bottleneck':
        init_h = BottleNeck(planes, planes)
    elif block_type == 'DwConv':
        init_h = layers.DepthwiseConv2D(kernel_size=3, strides=1, padding='same', use_bias=False)
    elif block_type == 'FullConv':
        init_h = layers.Conv2D(planes, kernel_size=3, strides=1, padding='same', use_bias=False)
    elif block_type == 'identity' or block_type == 'default':
        init_h = layers.Activation('linear')
    else:
        print('Undefined cell type.', block_type)
    return init_h 


class GlobalFeatureBlock_Diffusion(tf.keras.Model):
    expansion = 1
    
    def __init__(self, planes):
        super(GlobalFeatureBlock_Diffusion, self).__init__()

        K = 2
        nonlinear_pde = True
        pde_state = 0

        cDx = 1.
        cDy = 1.
        dx = 1
        dy = 1
        dt = 0.2

        init_h0_h = False
        self.use_f_for_g = False
        use_diff_eps = True
        use_silu = False
        use_res = False
        use_cDs = False
        use_dw =False
        use_dot = False
        drop_path_rate = 0.
        
        no_f = False
        block_type = 'default'

        # PDE parameters
        constant_Dxy = False

        stride = 1
        in_chs = planes
        out_chs = planes

        print('Global Feature Block Diffusion : (K, planes, nonlinear_pde, pde_state, block_type)', K, planes, nonlinear_pde, pde_state, block_type)
        print('c_Dxy, dt, no_f, use_silu, use_res, cDx, cDy, init_h0_h, dx, dy, use_dot, use_cDs, drop_path_rate ', constant_Dxy, dt, no_f, use_silu, use_res, cDx, cDy, init_h0_h, dx, dy, use_dot, use_cDs, drop_path_rate)

        self.pde_state = pde_state
        self.nonlinear_pde = nonlinear_pde
        self.K = K

        self.init_h = get_init_block(planes, block_type)
        self.bn_out = layers.BatchNormalization()

        self.block_type = block_type
        self.init_h0_h = init_h0_h
        self.dx = dx
        self.dy = dy
        self.cDx = cDx
        self.cDy = cDy
        self.use_res = use_res
        self.use_dot = use_dot
        self.no_f = no_f
        self.dt = dt

        # PDE parameters
        self.constant_Dxy = constant_Dxy

        self.drop_path_rate = drop_path_rate
        self.stride = stride
        self.in_chs = in_chs
        self.out_chs = out_chs
        self.planes = planes

        # Choosing uv
        # self.convg  = get_init_block(planes, block_type = 'identity')
        # self.convg1 = get_init_block(planes, block_type = 'identity')
        # self.convg  = get_init_block(planes, block_type = 'FullConv')
        # self.convg1 = get_init_block(planes, block_type = 'FullConv')
        self.convg  = get_init_block(planes, block_type = 'DwConv')
        self.convg1 = get_init_block(planes, block_type = 'DwConv')
        # self.convg  = get_init_block(planes, block_type = 'BasicBlock')
        # self.convg1 = get_init_block(planes, block_type = 'BasicBlock')
        # self.convg  = get_init_block(planes, block_type = 'Bottleneck')
        # self.convg1 = get_init_block(planes, block_type = 'Bottleneck')

        self.bng = layers.BatchNormalization()
        self.bng1 = layers.BatchNormalization()

        # Choosing DxDy
        # self.convDx = get_init_block(planes, block_type = 'identity')
        # self.convDy = get_init_block(planes, block_type = 'identity')
        # self.convDx = get_init_block(planes, block_type = 'FullConv')
        # self.convDy = get_init_block(planes, block_type = 'FullConv')
        self.convDx = get_init_block(planes, block_type = 'DwConv')
        self.convDy = get_init_block(planes, block_type = 'DwConv')
        # self.convDx = get_init_block(planes, block_type = 'BasicBlock')
        # self.convDy = get_init_block(planes, block_type = 'BasicBlock')
        # self.convDx = get_init_block(planes, block_type = 'Bottleneck')
        # self.convDy = get_init_block(planes, block_type = 'Bottleneck')

        self.bnDx = layers.BatchNormalization()
        self.bnDy = layers.BatchNormalization()


    def call(self, s0):
        f = s0
        h = self.init_h(f)

        if (self.stride != 1) or (self.in_chs != self.out_chs):  
            f = h

        h0 = f  
        g0 = h 

        
        g = tf.keras.activations.relu(self.bng(self.convg(g0)))     # PDE parameter u
        g1 = tf.keras.activations.relu(self.bng1(self.convg1(g0)))  # PDE parameter v
            
        dt = self.dt 
        dx = self.dx 
        dy = self.dy 

        if self.constant_Dxy:
            Dx = self.cDx 
            Dy = self.cDy 
        else:
            Dx  = tf.keras.activations.relu(self.bnDx(self.convDx(h)))  # PDE parameter Dx
            Dy  = tf.keras.activations.relu(self.bnDy(self.convDy(h)))  # PDE parameter Dy

        ux = (1. / (2*dx)) * ( tf.roll(g, dx, axis=2)  - tf.roll(g, -dx, axis=2) )
        vy = (1. / (2*dy)) * ( tf.roll(g1, dy, axis=3) - tf.roll(g1, -dy, axis=3) )

        Ax = g  * (dt / dx)
        Ay = g1 * (dt / dy)
        Bx = Dx * (dt / (dx*dx))
        By = Dy * (dt / (dy*dy))
        E  = (ux + vy) * dt

        D = (1. / (1 + 2*Bx + 2*By))

        for k in range(self.K):
            prev_h = h
            h = D  *   (   (1 - 2*Bx - 2*By) * h0 - 2 * E * h 
                         + (-Ax  + 2*Bx) * tf.roll(h, dx, axis=2) 
                         + ( Ax  + 2*Bx) * tf.roll(h, -dx, axis=2) 
                         + (-Ay  + 2*By) * tf.roll(h, dy, axis=3)  
                         + ( Ay  + 2*By) * tf.roll(h, -dy, axis=3)  
                        )
            h = h + D * 2 * dt * f 
            h0 = prev_h
                
        h = self.bn_out(h)
        h = tf.keras.activations.relu(h)
        return h