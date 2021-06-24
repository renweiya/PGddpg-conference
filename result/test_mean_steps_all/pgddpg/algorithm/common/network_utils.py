import tensorflow as tf
import tensorflow.contrib.layers as layers


def mlp_model(input, num_outputs, scope, reuse=False, num_units=256, layer_norm=False, alpha=0.01):
    # This model takes as input an observation and returns values of all actions
    with tf.variable_scope(scope, reuse=reuse):
        out = input
        #layer 1
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            out = layers.layer_norm(out, center=True, scale=True)
        # nonlinear activation
        out = tf.maximum(alpha * out, out)
        # out = tf.nn.relu(out)
        
        #layer 2
        # #add by laker
        # out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        # if layer_norm:
        #     out = layers.layer_norm(out, center=True, scale=True)
        # out = tf.maximum(alpha * out, out)
        
        #layer 3
        out = layers.fully_connected(out, num_outputs=num_units, activation_fn=None)
        if layer_norm:
            out = layers.layer_norm(out, center=True, scale=True)
        out = tf.maximum(alpha * out, out)
        
        #last layer
        out = layers.fully_connected(out, num_outputs=num_outputs, activation_fn=None)

        return out

        # out = tf.tanh(out)        
        # out = tf.sigmoid(logits)
        # outputs = fully_connected(net, self.action_dim, activation_fn=tf.tanh)#tanh的值域为[-1.0,1.0]
        # scaled_outputs = tf.multiply(outputs, self.action_bound) # Scale output to [-action_bound, action_bound]
                

def plan_model(input, scope, reuse=False):#add by laker
    #plan
    with tf.variable_scope(scope, reuse=reuse):
        # APF_potential=-tf.square(tf.norm(input[:,-4:-2],axis=-1))#last last 4th,3th vector
        APF_potential=-tf.norm(input[:,-4:-2],axis=-1)#last last 4th,3th vector


        xxx_norm=tf.nn.l2_normalize(input[:,-2:], dim=1, epsilon=1e-12,name=None)
        yyy_norm=tf.nn.l2_normalize(input[:,-4:-2], dim=1, epsilon=1e-12,name=None)#last 4th,3th vector
        x_multiply_y=tf.multiply(xxx_norm,yyy_norm)#matrix dot product
        vector_value=tf.reduce_sum(x_multiply_y, axis=1)#sum by line
        
        # out=tf.multiply(APF_potential,1-vector_value)#matrix dot product
        
        q_APF_steer=tf.multiply(APF_potential,1-vector_value)#matrix dot product
        diff_value_velocity=tf.multiply(input[:,-2:],input[:,-2:])
        diff_value_velocity=tf.reduce_sum(diff_value_velocity, axis=1)#sum by line
        diff_value_velocity_ = tf.minimum(diff_value_velocity-1.0, 0.0)
        diff_value_velocity_= tf.square(diff_value_velocity_)
        q_APF_throttle=tf.multiply(APF_potential,diff_value_velocity_)
        out = q_APF_steer + q_APF_throttle #laker 

        return out

def plan_model_force(input, scope, reuse=False):#add by laker
    #plan
    with tf.variable_scope(scope, reuse=reuse):
        # APF_potential=-tf.square(tf.norm(input[:,-4:-2],axis=-1))#last last 4th,3th vector
        APF_potential=-tf.norm(input[:,-4:-2],axis=-1)#last last 4th,3th vector
        
        #input[:,-2:]  #force by actor, belong to [-1,1]
        #input[:,-6:-4]#force by apf, belong to [-1,1]

        force_diff=tf.subtract(input[:,-2:],input[:,-6:-4])
        q_APF_force_diff=tf.norm(force_diff,axis=-1)
	    #q_APF_force_diff=tf.multiply(q_APF_force_diff,0.70710678)
        # q_APF_force_diff=tf.sqrt(tf.norm(force_diff,axis=-1))
        out=tf.multiply(APF_potential,q_APF_force_diff)#matrix dot product

        return out

def plan_model_force2(input, scope, reuse=False):#add by laker
    #plan
    with tf.variable_scope(scope, reuse=reuse):
        # APF_potential=-tf.square(tf.norm(input[:,-4:-2],axis=-1))#last last 4th,3th vector
        APF_potential=-tf.norm(input[:,-4:-2],axis=-1)#last last 4th,3th vector
        
        #input[:,-2:]  #force by actor, belong to [-1,1]
        #input[:,-6:-4]#force by apf, belong to [-1,1]

        xxx_norm=tf.nn.l2_normalize(input[:,-2:], dim=1, epsilon=1e-12,name=None)
        yyy_norm=tf.nn.l2_normalize(input[:,-6:-4], dim=1, epsilon=1e-12,name=None)#last 4th,3th vector
        x_multiply_y=tf.multiply(xxx_norm,yyy_norm)#matrix dot product
        vector_value=tf.reduce_sum(x_multiply_y, axis=1)#sum by line        
        q_APF_steer=tf.multiply(APF_potential,1-vector_value)#matrix dot product

        diff_value_velocity=tf.multiply(input[:,-2:],input[:,-2:])
        diff_value_velocity=tf.reduce_sum(diff_value_velocity, axis=1)#sum by line
        diff_value_velocity_ = tf.minimum(diff_value_velocity-1.0, 0.0)
        diff_value_velocity_= tf.square(diff_value_velocity_)
        q_APF_throttle=tf.multiply(APF_potential,diff_value_velocity_)
        out = q_APF_steer + q_APF_throttle #laker 


        return out