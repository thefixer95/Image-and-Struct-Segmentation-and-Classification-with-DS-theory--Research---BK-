from unicodedata import name
import tensorflow as tf
import numpy as np
import sys

class L1_wadd(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(L1_wadd, self).__init__()
        self.w=self.add_weight(
            name='weights_per_class',
            shape=(input_dim, num_class),
            initializer='random_normal',
            trainable=True
        )
        # self.c=self.add_weight(
        #     name='weights_per_class_second',
        #     shape=(input_dim, num_class),
        #     initializer='random_normal',
        #     trainable=True
        # )
        self.num_class=num_class

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_class': self.num_class,
            'weights_per_class': self.w.numpy(),
            # 'weights_per_class_second': self.c.numpy(),
            })
        return config

    def call(self, inputs):
        ### DISTANZA VECCHIA
        # inputs = tf.clip_by_value(inputs,tf.float32.min,tf.float32.max) #clip all values between epsilon and 1

        # print(self.w.shape)
        # print(inputs.shape)

        if len(inputs.shape) < 3:
            u_dist_i = tf.expand_dims(inputs,-1)
        else:
            u_dist_i = inputs
        # print(inputs.shape)
        u_dist_i = tf.subtract(u_dist_i,self.w)
        u_dist_i = tf.abs(u_dist_i)
        u_dist_i = tf.square(u_dist_i)
        # print(tf.print(u_dist_i[0]))

        # ## DISTANZA NUOVA
        # if len(inputs.shape) < 3:
        #     u_dist_i = tf.expand_dims(inputs,-1)
        # else:
        #     u_dist_i = inputs

        # dist2 = tf.subtract(u_dist_i, self.w)
        # # dist1 = tf.transpose(dist2)

        # # c_inv = tf.linalg.inv(self.c)

        # # print("dist2 shape: " + str(dist2.shape))
        # # print("dist1 shape: " + str(dist1.shape))
        # # print("c shape: " + str(self.c.shape))

        # dist = tf.multiply(self.c,dist2) + 1e-10
        # # dist = tf.multiply(dist,dist2)
        # dist = tf.sqrt(dist)

        

        # return dist
        return u_dist_i


class L1_wadd_activate(tf.keras.layers.Layer):
    def __init__(self,input_dim,out_dim):
        super(L1_wadd_activate, self).__init__()
        self.input_dim = input_dim
        self.out_dim = out_dim
        # can also be only two numbers?       
        # self.alpha=4.0
        w_init = tf.random_normal_initializer()
        value = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        # w_init = tf.constant_initializer(np.random.choice(value))
        self.alpha = tf.Variable(
            name='alpha',
            initial_value=w_init(shape=(input_dim, out_dim),
            # initial_value=w_init(shape=[1],
                             dtype='float32'),
            trainable=True)
        
        # self.alpha=self.add_weight(
        #     name='alpha',
        #     # shape=[1],
        #     shape=(input_dim,out_dim),
        #     # initializer='he_normal',
        #     initializer= tf.keras.initializers.RandomUniform(
        #                     minval=1.0, maxval=10.0, seed=None
        #                 ),
        #     trainable=True
        # )

        self.gamma=2.0

        # value = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        # w_init = tf.constant_initializer(np.random.choice(value))
        # self.gamma = tf.Variable(
        #     name='gamma',
        #     # initial_value=w_init(shape=(input_dim, out_dim),
        #     initial_value=w_init(shape=[1],
        #                      dtype='float32'),
        #     trainable=True)
        # self.gamma=self.add_weight(
        #     name='gamma',
        #     # shape=[1],
        #     shape=(input_dim,out_dim),
        # #     initializer='he_normal',
        #     initializer= tf.keras.initializers.RandomUniform(
        #                     minval=1.0, maxval=10.0, seed=None
        #                 ),
        #     trainable=True
        # )

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_dim': self.input_dim,
            'out_dim': self.out_dim,
            # 'alpha': self.alpha,
            'gamma': self.gamma,
            'alpha': self.alpha.numpy(),
            # 'gamma': self.gamma.numpy(),
        })
        return config    

    def call(self, inputs):
        
        # # #OLD DISTANCE APPLICATION
        # # inputs = tf.clip_by_value(inputs,1e-10,tf.float32.max) #clip all values between epsilon and 1

        # # print(tf.print(inputs[0]))        
        # # phi_dist = tf.negative(self.gamma) + 0.001
        # phi_dist = -1.0
        # # phi_dist = tf.multiply(-20,inputs)
        # # print(tf.print(phi_dist))  
        # phi_dist = tf.multiply(phi_dist,inputs)
        # # print(tf.print(phi_dist[0]))         
        # phi_dist = tf.exp(phi_dist)
        # # # print(tf.print(phi_dist[0]))
        # phi_dist = tf.multiply(self.alpha,phi_dist) + 0.000001    #works with full images not segmented
        # # phi_dist = tf.multiply(self.alpha,phi_dist) + 0.001
        # # phi_dist = tf.multiply(self.alpha,phi_dist)

        # # phi_dist = phi_dist / (tf.reduce_sum(phi_dist, axis = -1, keepdims=True) + 1e-10)
        # # print(tf.print(phi_dist[0]))
        
        # ### NEW DISTANCE APPLICATION
        phi_dist = tf.subtract(inputs,self.alpha)
        phi_dist = tf.multiply(phi_dist,self.gamma)
        phi_dist = tf.negative(phi_dist)
        phi_dist = tf.exp(phi_dist)
        phi_dist = tf.add(1.0,phi_dist)
        phi_dist = tf.pow(phi_dist,-1)
        phi_dist = tf.subtract(1.0,phi_dist)




        return phi_dist


class L2_masses(tf.keras.layers.Layer):
    def __init__(self):
        super(L2_masses, self).__init__()
        
    def get_config(self):
        config = super().get_config().copy()
        return config
    
    def call(self, inputs):
        # inputs = tf.clip_by_value(inputs,1e-10,tf.float32.max) #clip all values between epsilon and 1
        
        # print(tf.print(inputs[0]))
        mass_omega_sum=tf.reduce_sum(inputs, -1, keepdims=True)
        # print(tf.print(mass_omega_sum[0]))
        mass_omega_sum=tf.subtract(1., mass_omega_sum[:,:,0], name=None)
        # print(tf.print(mass_omega_sum[0]))
        mass_omega_sum=tf.expand_dims(mass_omega_sum, -1)
        # print(tf.print(mass_omega_sum[0]))
        mass_with_omega=tf.concat([inputs, mass_omega_sum], -1)
        # print(tf.print(mass_omega_sum[0]))

        mass_with_omega = mass_with_omega / (tf.reduce_sum(mass_with_omega, axis = -1, keepdims=True) + 0.0001)

        return mass_with_omega


class L3_combine_masses(tf.keras.layers.Layer):
    def __init__(self):
        super(L3_combine_masses, self).__init__()
    
    def get_config(self):
        config = super().get_config().copy()
        return config
    
    def call(self, inputs):
        
        masses_combined = tf.reduce_sum(inputs,1)

        # masses_combined = tf.abs(masses_combined)
        # masses_combined = tf.reduce_prod(inputs,1)
        # print(tf.print(masses_combined[0]))
        # print(tf.print(tf.reduce_sum(masses_combined, axis = -1, keepdims=True)[0]))
        mass_combine_normalize = masses_combined / (tf.reduce_sum(masses_combined, axis = -1, keepdims=True) + 0.0001)
        # mass_combine_normalize = tf.clip_by_value(mass_combine_normalize,1e-10,1.0) #clip all values between epsilon and 1
        # mass_combine_normalize = tf.math.l2_normalize(masses_combined)
        # print("masse combinate normalizzate: \n")
        # print(tf.print(mass_combine_normalize[0]))
        # print("pippo")
        return mass_combine_normalize






##############
#####           Uncertanty from Dezer
##############


class Uncertanty_Layer(tf.keras.layers.Layer):
    def __init__(self):
        super(Uncertanty_Layer, self).__init__()
        

    def get_config(self):
        config = super().get_config().copy()
        return config

    
    def call(self, inputs):
        uncertanty = tf.reduce_sum(inputs,2)
        uncertanty = 1- uncertanty
        uncertanty = tf.expand_dims(uncertanty, -1)
        # print(tf.print(uncertanty))
        
        self.add_metric(uncertanty, name='uncertanty')

        return uncertanty




class Uncertanty_Layer_1(tf.keras.layers.Layer):
    def __init__(self):
        super(Uncertanty_Layer_1, self).__init__()
        

    def get_config(self):
        config = super().get_config().copy()
        return config

    
    def call(self, inputs):
        uncertanty = tf.reduce_sum(inputs,2)
        uncertanty = 1- uncertanty
        # uncertanty = tf.expand_dims(uncertanty, -1)
        # print(tf.print(uncertanty))
        
        
        # self.add_metric(uncertanty[0], name='uncertanty_2')
        
        return uncertanty



class Uncertanty_Layer_FIXED(tf.keras.layers.Layer):
    def __init__(self):
        super(Uncertanty_Layer_FIXED, self).__init__()
        

    def get_config(self):
        config = super().get_config().copy()
        return config

    
    def call(self, inputs):
        inputs = tf.reduce_sum(inputs,2)
        ###inputs = m(theta_1) m(theta_2)



        uncertanty = 1- uncertanty
        # uncertanty = tf.expand_dims(uncertanty, -1)
        # print(tf.print(uncertanty))
        
        
        # self.add_metric(uncertanty[0], name='uncertanty_2')
        
        return uncertanty



# class Uncertanty_Layer(tf.keras.layers.Layer):
#     def __init__(self):
#         super(Uncertanty_Layer, self).__init__()
#         self.B_w = tf.random_normal_initializer()
        

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#            'B_w': self.B_w,
#         })
#         return config

    
#     def call(self, inputs):
#         ### inputs = Pl





#         return uncertanty



# class L1_wadd(tf.keras.layers.Layer):
#     def __init__(self, input_dim, num_class):
#         super(L1_wadd, self).__init__()
#         self.w=self.add_weight(
#             name='weights_per_class',
#             shape=(input_dim, num_class),
#             initializer='random_normal',
#             trainable=True
#         )
#         # self.c=self.add_weight(
#         #     name='weights_per_class_second',
#         #     shape=(input_dim, num_class),
#         #     initializer='random_normal',
#         #     trainable=True
#         # )
#         self.num_class=num_class

#     def get_config(self):
#         config = super().get_config().copy()
#         config.update({
#             'num_class': self.num_class,
#             'weights_per_class': self.w.numpy(),
#             # 'weights_per_class_second': self.c.numpy(),
#             })
#         return config

#     def call(self, inputs):
#         ### DISTANZA VECCHIA
#         # inputs = tf.clip_by_value(inputs,tf.float32.min,tf.float32.max) #clip all values between epsilon and 1

#         # print(self.w.shape)
#         # print(inputs.shape)

#         if len(inputs.shape) < 3:
#             u_dist_i = tf.expand_dims(inputs,-1)
#         else:
#             u_dist_i = inputs
#         # print(inputs.shape)
#         u_dist_i = tf.subtract(u_dist_i,self.w)
#         u_dist_i = tf.abs(u_dist_i)
#         u_dist_i = tf.square(u_dist_i)
#         # print(tf.print(u_dist_i[0]))

#         # ## DISTANZA NUOVA
#         # if len(inputs.shape) < 3:
#         #     u_dist_i = tf.expand_dims(inputs,-1)
#         # else:
#         #     u_dist_i = inputs

#         # dist2 = tf.subtract(u_dist_i, self.w)
#         # # dist1 = tf.transpose(dist2)

#         # # c_inv = tf.linalg.inv(self.c)

#         # # print("dist2 shape: " + str(dist2.shape))
#         # # print("dist1 shape: " + str(dist1.shape))
#         # # print("c shape: " + str(self.c.shape))

#         # dist = tf.multiply(self.c,dist2) + 1e-10
#         # # dist = tf.multiply(dist,dist2)
#         # dist = tf.sqrt(dist)

        

#         # return dist
#         return u_dist_i


# class L1_wadd_activate(tf.keras.layers.Layer):
#     def __init__(self,input_dim,out_dim):
#         super(L1_wadd_activate, self).__init__()
#         self.input_dim = input_dim
#         self.out_dim = out_dim
#         # can also be only two numbers?       
#         # self.alpha=4.0
#         w_init = tf.random_normal_initializer()
#         value = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
#         # w_init = tf.constant_initializer(np.random.choice(value))
#         self.alpha = tf.Variable(
#             name='alpha',
#             initial_value=w_init(shape=(input_dim, out_dim),
#             # initial_value=w_init(shape=[1],
#                              dtype='float32'),
#             trainable=True)
        
#         # self.alpha=self.add_weight(
#         #     name='alpha',
#         #     # shape=[1],
#         #     shape=(input_dim,out_dim),
#         #     # initializer='he_normal',
#         #     initializer= tf.keras.initializers.RandomUniform(
#         #                     minval=1.0, maxval=10.0, seed=None
#         #                 ),
#         #     trainable=True
#         # )

#         self.gamma=2.0

#         # value = [2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
#         # w_init = tf.constant_initializer(np.random.choice(value))
#         # self.gamma = tf.Variable(
#         #     name='gamma',
#         #     # initial_value=w_init(shape=(input_dim, out_dim),
#         #     initial_value=w_init(shape=[1],
#         #                      dtype='float32'),
#         #     trainable=True)
#         # self.gamma=self.add_weight(
#         #     name='gamma',
#         #     # shape=[1],
#         #     shape=(input_dim,out_dim),
#         # #     initializer='he_normal',
#         #     initializer= tf.keras.initializers.RandomUniform(
#         #                     minval=1.0, maxval=10.0, seed=None
#         #                 ),
#         #     trainable=True
#         # )

#     #AGGIUNTA PER CALLBACK
#     def get_config(self):

#         config = super().get_config().copy()
#         config.update({
#             'input_dim': self.input_dim,
#             'out_dim': self.out_dim,
#             # 'alpha': self.alpha,
#             'gamma': self.gamma,
#             'alpha': self.alpha.numpy(),
#             # 'gamma': self.gamma.numpy(),
#         })
#         return config    

#     def call(self, inputs):
        
#         # # #OLD DISTANCE APPLICATION
#         # # inputs = tf.clip_by_value(inputs,1e-10,tf.float32.max) #clip all values between epsilon and 1

#         # # print(tf.print(inputs[0]))        
#         # # phi_dist = tf.negative(self.gamma) + 0.001
#         # phi_dist = -1.0
#         # # phi_dist = tf.multiply(-20,inputs)
#         # # print(tf.print(phi_dist))  
#         # phi_dist = tf.multiply(phi_dist,inputs)
#         # # print(tf.print(phi_dist[0]))         
#         # phi_dist = tf.exp(phi_dist)
#         # # # print(tf.print(phi_dist[0]))
#         # phi_dist = tf.multiply(self.alpha,phi_dist) + 0.000001    #works with full images not segmented
#         # # phi_dist = tf.multiply(self.alpha,phi_dist) + 0.001
#         # # phi_dist = tf.multiply(self.alpha,phi_dist)

#         # # phi_dist = phi_dist / (tf.reduce_sum(phi_dist, axis = -1, keepdims=True) + 1e-10)
#         # # print(tf.print(phi_dist[0]))
        
#         # ### NEW DISTANCE APPLICATION
#         phi_dist = tf.subtract(inputs,self.alpha)
#         phi_dist = tf.multiply(phi_dist,self.gamma)
#         phi_dist = tf.negative(phi_dist)
#         phi_dist = tf.exp(phi_dist)
#         phi_dist = tf.add(1.0,phi_dist)
#         phi_dist = tf.pow(phi_dist,-1)
#         phi_dist = tf.subtract(1.0,phi_dist)




#         return phi_dist


# class L2_masses(tf.keras.layers.Layer):
#     def __init__(self):
#         super(L2_masses, self).__init__()
        
#     def get_config(self):
#         config = super().get_config().copy()
#         return config
    
#     def call(self, inputs):
#         # inputs = tf.clip_by_value(inputs,1e-10,tf.float32.max) #clip all values between epsilon and 1
        
#         # print(tf.print(inputs[0]))
#         mass_omega_sum=tf.reduce_sum(inputs, -1, keepdims=True)
#         # print(tf.print(mass_omega_sum[0]))
#         mass_omega_sum=tf.subtract(1., mass_omega_sum[:,:,0], name=None)
#         # print(tf.print(mass_omega_sum[0]))
#         mass_omega_sum=tf.expand_dims(mass_omega_sum, -1)
#         # print(tf.print(mass_omega_sum[0]))
#         mass_with_omega=tf.concat([inputs, mass_omega_sum], -1)
#         # print(tf.print(mass_omega_sum[0]))

#         mass_with_omega = mass_with_omega / (tf.reduce_sum(mass_with_omega, axis = -1, keepdims=True) + 0.0001)

#         return mass_with_omega


# class L3_combine_masses(tf.keras.layers.Layer):
#     def __init__(self):
#         super(L3_combine_masses, self).__init__()
    
#     def get_config(self):
#         config = super().get_config().copy()
#         return config
    
#     def call(self, inputs):
        
#         masses_combined = tf.reduce_sum(inputs,1)

#         # masses_combined = tf.abs(masses_combined)
#         # masses_combined = tf.reduce_prod(inputs,1)
#         # print(tf.print(masses_combined[0]))
#         # print(tf.print(tf.reduce_sum(masses_combined, axis = -1, keepdims=True)[0]))
#         mass_combine_normalize = masses_combined / (tf.reduce_sum(masses_combined, axis = -1, keepdims=True) + 0.0001)
#         # mass_combine_normalize = tf.clip_by_value(mass_combine_normalize,1e-10,1.0) #clip all values between epsilon and 1
#         # mass_combine_normalize = tf.math.l2_normalize(masses_combined)
#         # print("masse combinate normalizzate: \n")
#         # print(tf.print(mass_combine_normalize[0]))
#         # print("pippo")
#         return mass_combine_normalize








class DS1(tf.keras.layers.Layer):
    def __init__(self, units, input_dim):
        super(DS1, self).__init__()
        self.w=self.add_weight(
            name='Prototypes',
            shape=(units, input_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.units=units
    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'units': self.units,
            'Prototypes': self.w.numpy(),
        })
        return config           
    
    def call(self, inputs):
        for i in range(self.units):
          if i==0:
            un_mass_i=tf.subtract(self.w[i,:], inputs, name=None)
            un_mass_i=tf.square(un_mass_i, name=None)
            un_mass_i=tf.reduce_sum(un_mass_i, -1, keepdims=True)
            un_mass = un_mass_i

          if i>=1:
            un_mass_i=tf.subtract(self.w[i,:], inputs, name=None)
            un_mass_i=tf.square(un_mass_i, name=None)
            un_mass_i=tf.reduce_sum(un_mass_i, -1, keepdims=True)
            un_mass=tf.concat([un_mass, un_mass_i], -1)
        return un_mass

class DS1_activate(tf.keras.layers.Layer):
    def __init__(self, input_dim):
        super(DS1_activate, self).__init__()
        self.xi=self.add_weight(
            name='xi',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.eta=self.add_weight(
            name='eta',
            shape=(1, input_dim),
            initializer='random_normal',
            trainable=True
        )
        
        self.input_dim=input_dim

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'xi': self.xi.numpy(),
            'eta': self.eta.numpy(),
            'input_dim_DS1_activate': self.input_dim,
        })
        return config    

    def call(self, inputs):
        gamma=tf.square(self.eta, name=None)
        alpha=tf.negative(self.xi, name=None)
        alpha=tf.exp(alpha, name=None)+1
        alpha=tf.divide(1, alpha, name=None)
        si=tf.multiply(gamma, inputs, name=None)
        si=tf.negative(si, name=None)
        si=tf.exp(si, name=None)
        si=tf.multiply(si, alpha, name=None)
        si = si / (tf.reduce_max(si, axis = -1, keepdims=True) + 0.0001)
        return si

class DS2(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(DS2, self).__init__()
        self.beta=self.add_weight(
            name='beta',
            shape=(input_dim, num_class),
            initializer='random_normal',
            trainable=True
        )
        
        self.input_dim=input_dim
        self.num_class=num_class

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'beta': self.beta.numpy(),
            'input_dim_DS2': self.input_dim,
            'num_class_DS2': self.num_class,
        })
        return config    
                    
    def call(self, inputs):
        beta=tf.square(self.beta, name=None)
        beta_sum=tf.reduce_sum(beta, -1, keepdims=True)
        u=tf.divide(beta, beta_sum, name=None)
        inputs_new=tf.expand_dims(inputs, -1)
        for i in range(self.input_dim):
          if i==0:
            mass_prototype_i=tf.multiply(u[i,:], inputs_new[:,i], name=None)
            mass_prototype=tf.expand_dims(mass_prototype_i, -2)
          if i>0:
            mass_prototype_i=tf.expand_dims(tf.multiply(u[i,:], inputs_new[:,i], name=None), -2)
            mass_prototype=tf.concat([mass_prototype, mass_prototype_i], -2)
        mass_prototype=tf.convert_to_tensor(mass_prototype)
        return mass_prototype

class DS2_omega(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(DS2_omega, self).__init__()
        self.input_dim=input_dim
        self.num_class=num_class

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_dim_DS2_omega': self.input_dim,
            'num_class_DS2_omega': self.num_class,
        })
        return config    
                    
    def call(self, inputs):
        mass_omega_sum=tf.reduce_sum(inputs, -1, keepdims=True)
        mass_omega_sum=tf.subtract(1., mass_omega_sum[:,:,0], name=None)
        mass_omega_sum=tf.expand_dims(mass_omega_sum, -1)
        mass_with_omega=tf.concat([inputs, mass_omega_sum], -1)
        return mass_with_omega
    
class DS3_Dempster(tf.keras.layers.Layer):
    def __init__(self, input_dim, num_class):
        super(DS3_Dempster, self).__init__()
        self.input_dim=input_dim
        self.num_class=num_class

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'input_dim_DS3_Dempster': self.input_dim,
            'num_class_DS3_Dempster': self.num_class,
        })
        return config    
        
    def call(self, inputs):
        m1=inputs[:,0,:]
        omega1=tf.expand_dims(inputs[:,0,-1],-1)
        for i in range (self.input_dim-1):
            m2=inputs[:,(i+1),:]
            omega2=tf.expand_dims(inputs[:,(i+1),-1], -1)
            combine1=tf.multiply(m1, m2, name=None)
            combine2=tf.multiply(m1, omega2, name=None)
            combine3=tf.multiply(omega1, m2, name=None)
            combine1_2=tf.add(combine1, combine2, name=None)
            combine2_3=tf.add(combine1_2, combine3, name=None)
            combine2_3 = combine2_3 / tf.reduce_sum(combine2_3, axis = -1, keepdims=True)#后加的
            m1=combine2_3
            omega1=tf.expand_dims(combine2_3[:,-1], -1)
        return m1

class DS3_normalize(tf.keras.layers.Layer):
    def __init__(self):
        super(DS3_normalize, self).__init__()

    #AGGIUNTA PER CALLBACK
    def get_config(self):
        config = super().get_config().copy()
        return config    
        
    def call(self, inputs):
        mass_combine_normalize = inputs / tf.reduce_sum(inputs, axis = -1, keepdims=True)
        return mass_combine_normalize