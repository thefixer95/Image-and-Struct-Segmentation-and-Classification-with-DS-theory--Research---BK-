import tensorflow as tf
class DM_pignistic(tf.keras.layers.Layer):
    def __init__(self, num_class):
        super(DM_pignistic, self).__init__()
        self.num_class=num_class

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'num_class_DM_pignistic': self.num_class,
        })
        return config    
        
    def call(self, inputs):
        aveage_Pignistic=inputs[:,-1]/self.num_class
        aveage_Pignistic=tf.expand_dims(aveage_Pignistic, -1)
        Pignistic_prob=inputs[:,:] + aveage_Pignistic
        Pignistic_prob=Pignistic_prob[:,0:-1]


        return Pignistic_prob

class DM(tf.keras.layers.Layer):
    def __init__(self, nu, num_class):
        super(DM, self).__init__()
        self.nu = nu
        self.num_class=num_class

    #AGGIUNTA PER CALLBACK
    def get_config(self):

        config = super().get_config().copy()
        config.update({
            'nu_DM': self.nu,
            'num_class_DM': self.num_class,
        })
        return config    
		
    def call(self, inputs):
        upper = tf.expand_dims((1-self.nu) * inputs[:,-1], -1)#here 0.1 = 1 - \nu
        upper = tf.tile(upper, [1,self.num_class+1])
        outputs = tf.add(inputs, upper, name=None)[:,0:-1]
        return outputs