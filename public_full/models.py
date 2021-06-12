import numpy as np
import os
import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl
import tensorflow.keras.regularizers as tfkr
from states import rotate_state

class Model:
    
    def __init__(self,**kwargs):
        self.exp_pool = {0:[]}
        self.exp_pool_size = {0:0}
        self.max_exp_pool_size = int(kwargs.get('max_exp_pool_size',1e5))
        self.exp_sample_size = kwargs.get('exp_sample_size',1024)
        self.save_dir = kwargs.get('save_dir','')
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        self.learn_iter = kwargs.get('learn_iter',1)
        
        self.graph = None
        self.sess = None
        self.saver = None
    
    def save(self,fname=''):
        if self.saver is not None:
            self.saver.save(self.sess,self.save_dir + fname + '/model')
        
    def restore(self,fname=''):
        if self.saver is not None:
            self.saver.restore(self.sess,self.save_dir + fname + '/model')
        
    def add_experience(self,new_exp,exp_pool_id=0):
        if exp_pool_id not in self.exp_pool:
            self.exp_pool[exp_pool_id] = []
            self.exp_pool_size[exp_pool_id] = 0
            
        self.exp_pool[exp_pool_id].append(new_exp)
        self.exp_pool_size[exp_pool_id] += 1
        if self.exp_pool_size[exp_pool_id] > self.max_exp_pool_size:
            self.exp_pool[exp_pool_id].pop(np.random.choice(self.exp_pool_size[exp_pool_id]))
            self.exp_pool_size[exp_pool_id] = self.max_exp_pool_size
        
    def get_experience(self,exp_pool_id=0):
        exp = self.exp_pool[exp_pool_id]
        return exp
        
    def get_experience_sample(self,exp_pool_id=0):
        exp = self.exp_pool[exp_pool_id]
        if len(exp) >= max(self.exp_sample_size*2,1024):
            sample_idx = np.random.choice(len(exp),size=(self.exp_sample_size,),replace=False)
            sample_exp = [exp[i] for i in sample_idx]
            return sample_exp
        else:
            return None
        
    def clear_experience(self):
        self.exp_pool = {0:[]}
        self.exp_pool_size = {0:0}
        
    def faster_run(self,target,feed_dict):
        feed_dict_ = dict([(key,np.asarray(value,dtype=key.dtype.as_numpy_dtype)) \
                           for key,value in feed_dict.items()])
        return self.sess._do_run(None,[],target,feed_dict_,None,None)

    def even_faster_run(self,fetch_list,feed_dict):
        # return self.sess._call_tf_sessionrun(options=None,
        #                                     feed_dict=dict([(t._as_tf_output(),v) for t,v in feed_dict.items()]),
        #                                     fetch_list=[t._as_tf_output() for t in fetch_list],
        #                                     target_list=[],
        #                                     run_metadata=None)
        return self.sess._call_tf_sessionrun(options=None,
                                            feed_dict=feed_dict,
                                            fetch_list=fetch_list,
                                            target_list=[],
                                            run_metadata=None)
            
    def predict(self,state_vec,actions_vec,direc=1):
        n_actions = actions_vec.shape[1]
        qs,probs,action_idx = np.zeros((n_actions,)),np.ones((n_actions,)) / n_actions,0
        return qs,probs,action_idx
        
    def learn(self):
        pass

def transform_vec_tf(x_vec):
    x_vec = tf.stack([tf.cast(tf.equal(x_vec,0),dtype=tf.float32),
                        tf.cast(tf.equal(x_vec,1),dtype=tf.float32),
                        tf.cast(tf.equal(x_vec,2),dtype=tf.float32)],axis=-2)
    # x_vec = tf.expand_dims(x_vec,axis=-1)
    x_vec = tf.reshape(x_vec,(-1,3,54))
    ## reshape the deck from 4*13+2 to 4*(12+6), for the extraction of continuous pair info
    x_vec1 = tf.reshape(x_vec[:,:,:48],(-1,3,4,12))
    x_vec2 = tf.concat([tf.zeros((tf.shape(x_vec)[0],3,3,6)),tf.reshape(x_vec[:,:,48:],(-1,3,1,6))],axis=-2)
    x_vec = tf.reshape(tf.concat([x_vec1,x_vec2],axis=-1),(-1,12,18))
    x_vec = tf.transpose(x_vec,[0,2,1])
    return x_vec

def get_models(vec_hidden_units,state_hidden_units,value_hidden_units,scale_hidden_units,qvalue_hidden_units,probs_hidden_units,activation,reg_scale=1e-5):
    reg = tfkr.L1L2(l1=0.0,l2=reg_scale)
    ## a collection of models
    inputs = tfk.Input(shape=(18,12,))
    dense = inputs
    for unit in vec_hidden_units[:-1]:
        dense = tfkl.Conv1D(unit,3,activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    dense = tfkl.Flatten()(dense)
    outputs = tfkl.Dense(vec_hidden_units[-1],activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    vec_model = tfk.Model(inputs,outputs)
    
    inputs = tfk.Input(shape=(vec_hidden_units[-1]*14+12+13+3+1,))
    dense = inputs
    for unit in state_hidden_units:
        dense = tfkl.Dense(unit,activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    outputs = dense
    state_model = tfk.Model(inputs,outputs)

    inputs = tfk.Input(shape=(state_hidden_units[-1],))
    dense = inputs
    for unit in value_hidden_units:
        dense = tfkl.Dense(unit,activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    outputs = tfkl.Dense(1,activation=tf.tanh,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    value_model = tfk.Model(inputs,outputs)

    inputs = tfk.Input(shape=(state_hidden_units[-1],))
    dense = inputs
    for unit in scale_hidden_units:
        dense = tfkl.Dense(unit,activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    outputs = tfkl.Dense(1,activation=tf.tanh,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    scale_model = tfk.Model(inputs,outputs)
    
    inputs = tfk.Input(shape=(vec_hidden_units[-1]+state_hidden_units[-1]+1,))
    dense = inputs
    for unit in qvalue_hidden_units:
        dense = tfkl.Dense(unit,activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    outputs = tfkl.Dense(1,activation=tf.tanh,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    qvalue_model = tfk.Model(inputs,outputs)

    inputs = tfk.Input(shape=(vec_hidden_units[-1]+state_hidden_units[-1]+1,))
    dense = inputs
    for unit in probs_hidden_units:
        dense = tfkl.Dense(unit,activation=activation,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    outputs = tfkl.Dense(1,activation=None,kernel_regularizer=reg,bias_regularizer=reg)(dense)
    logits_model = tfk.Model(inputs,outputs)
    
    temp = tf.get_variable('temperature',shape=[],dtype=tf.float32,trainable=True)
    
    return vec_model,state_model,value_model,scale_model,qvalue_model,logits_model,temp

def q_net(s,a,As,vec_hidden_units,state_hidden_units,value_hidden_units,scale_hidden_units,qvalue_hidden_units,probs_hidden_units,activation,reg_scale=1e-5):
    s_vec,s_scalar = tf.reshape(s[:,:54 * 14],(-1,14,54)),s[:,54 * 14:]
    a_vec,a_scalar = a[:,:54],a[:,54:]/100
    As_vec,As_scalar = As[:,:,:54],As[:,:,54:]/100
    
    vec_model,state_model,value_model,scale_model,qvalue_model,logits_model,temp = \
        get_models(vec_hidden_units,state_hidden_units,value_hidden_units,scale_hidden_units,qvalue_hidden_units,probs_hidden_units,activation,reg_scale=reg_scale)
    
    s_vec,a_vec,As_vec = transform_vec_tf(s_vec),transform_vec_tf(a_vec),transform_vec_tf(As_vec)
    s_vec_dense,a_vec_dense,As_vec_dense = vec_model(s_vec),vec_model(a_vec),vec_model(As_vec)
    s_vec_dense = tf.reshape(s_vec_dense,(-1,14*vec_hidden_units[-1]))
    As_vec_dense = tf.reshape(As_vec_dense,(-1,tf.shape(As)[1],vec_hidden_units[-1]))
    # print(s_vec_dense,a_vec_dense,As_vec_dense)
    
    s_total = tf.concat([s_vec_dense,s_scalar],axis=-1)
    a_total = tf.concat([a_vec_dense,a_scalar],axis=-1)
    As_total = tf.concat([As_vec_dense,As_scalar],axis=-1)
    
    s_total = state_model(s_total)
    sa_total = tf.concat([s_total,a_total],axis=-1)
    sAs_total = tf.concat([tf.tile(tf.expand_dims(s_total,axis=1),
                                   (1,tf.shape(As_total)[1],1)),As_total],axis=-1)
    
    v_s = value_model(s_total)[:,0]
    # scale_s = scale_model(s_total)[:,0]

    # q_sa = qvalue_model(sa_total)[:,0] * scale_s + v_s
    q_sa = qvalue_model(sa_total)[:,0] + v_s

    q_sAs = qvalue_model(sAs_total)[:,:,0]
    q_sAs = (q_sAs - tf.expand_dims(tf.reduce_mean(q_sAs,axis=-1),axis=-1)) + tf.expand_dims(v_s,axis=-1)
    # q_sAs = (q_sAs - tf.expand_dims(tf.reduce_mean(q_sAs,axis=-1),axis=-1)) * 400 * tf.expand_dims(scale_s,axis=-1) + tf.expand_dims(v_s,axis=-1)

    logits_sAs = logits_model(sAs_total)[:,:,0]
    return v_s*400,q_sa*400,q_sAs*400,logits_sAs,temp

def transform_q_tf(q_sAs,logits_sAs,direcs,mask,temp):
    q_sAs_direc = q_sAs * tf.expand_dims(direcs,axis=-1)
    q_sAs_direc = q_sAs_direc * mask + (-10000) * (1 - mask)
    v_s = tf.reduce_max(q_sAs_direc,axis=-1) * direcs
    max_idx_sAs = tf.argmax(q_sAs_direc,axis=-1)

    logits_sAs_direc = logits_sAs * tf.expand_dims(direcs,axis=-1)
    logits_sAs_direc = logits_sAs_direc * mask
    logits_sAs_direc += (tf.expand_dims(tf.reduce_min(logits_sAs_direc,axis=-1),axis=-1)-10000) * (1 - mask)
    logits_sAs_final = logits_sAs_direc * tf.nn.softplus(temp)
    probs_sAs = tf.nn.softmax(logits_sAs_final)
    
    return v_s,max_idx_sAs,logits_sAs_final,probs_sAs

class BaseModel(Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
            
        # self.activation = tf.nn.leaky_relu
        self.activation = tf.nn.tanh
        self.vec_hidden_units = [4,4,16]
        self.state_hidden_units = [64,32]
        self.value_hidden_units = self.scale_hidden_units = [32]
        self.qvalue_hidden_units = self.probs_hidden_units = [32,16]
        self.reg_scale = 1e-5
        self.q_net = kwargs.get('q_net',
            lambda s,a,As: q_net(s,a,As,self.vec_hidden_units,self.state_hidden_units,self.value_hidden_units,self.scale_hidden_units,self.qvalue_hidden_units,self.probs_hidden_units,self.activation,self.reg_scale))

        self.lr = kwargs.get('lr',0.001)
        self.learn_step_counter = 0
        
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.sess = tf.Session(graph=self.graph)
            ## inputs
            self.tf_direc = tf.placeholder(tf.float32, [None, ], 'direc')
            self.tfs = tf.placeholder(tf.float32, [None, 14 * 54 + 12 + 13 + 3 + 1], 'state')
            self.tfa = tf.placeholder(tf.float32, [None, 54 + 1], 'action')
            self.tfAs = tf.placeholder(tf.float32, [None, None, 54 + 1], 'actions')
            self.tfmask_len = tf.placeholder(tf.int64, [None, ], 'mask_len')
            self.tfmask = tf.sequence_mask(self.tfmask_len, dtype=tf.float32)
            
            with tf.variable_scope('eval'):
                self.v_s_eval,self.q_sa_eval,self.q_sAs_eval,self.logits_sAs_eval,self.temp_eval = \
                    self.q_net(self.tfs,self.tfa,self.tfAs)
                self.v_s_new_eval,self.idx_sAs_eval,self.logits_sAs_final_eval,self.probs_sAs_eval = \
                    transform_q_tf(self.q_sAs_eval,self.logits_sAs_eval,self.tf_direc,self.tfmask,self.temp_eval)
                    
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.temp_eval.assign(-5))
            self.saver = tf.train.Saver(max_to_keep=100000)

        self.predict_out = [t._as_tf_output() for t in [self.q_sAs_eval,self.probs_sAs_eval,self.idx_sAs_eval]]
        self.tfs_out = self.tfs._as_tf_output()
        self.tfAs_out = self.tfAs._as_tf_output()
        self.tfmask_len_out = self.tfmask_len._as_tf_output()
        self.tf_direc_out = self.tf_direc._as_tf_output()

    def partial_restore(self,fname=''):
        with self.graph.as_default():
            var_name_to_restore = set([name for name,shape in tf.train.list_variables(self.save_dir + fname) if 'eval' in name and 'Adam' not in name])
            var_to_restore = [var for var in tf.global_variables() if var.name.split(':')[0] in var_name_to_restore]
            tf.train.Saver(var_to_restore).restore(self.sess,self.save_dir + fname + '/model')

    def predict(self,state_vec,actions_vec,direc=1):
        qs,probs,action_idx = self.even_faster_run(self.predict_out,
                                      {self.tfs_out:state_vec,
                                       self.tfAs_out:actions_vec,
                                       self.tfmask_len_out:np.array([actions_vec.shape[1]],dtype=np.int64),
                                       self.tf_direc_out:np.array([direc],dtype=np.float32)})
        # qs,probs,action_idx = self.faster_run([self.q_sAs_eval,self.probs_sAs_eval,self.idx_sAs_eval],
        #                               {self.tfs:state_vec,
        #                                self.tfAs:actions_vec,
        #                                self.tfmask_len:np.array([actions_vec.shape[1]]),
        #                                self.tf_direc:np.array([direc])})
        return qs[0],probs[0],action_idx[0]

class QModel(BaseModel):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
            
        self.exp_sample_size = kwargs.get('exp_sample_size',256)
        self.update_target_iter = kwargs.get('update_iter',100)
        
        with self.graph.as_default():
            ## labels
            self.tfq_sa = tf.placeholder(tf.float32, [None], 'qvalue')

            with tf.variable_scope('target'):
                self.v_s_target,self.q_sa_target,self.q_sAs_target,self.logits_sAs_target,self.temp_target = \
                    self.q_net(self.tfs,self.tfa,self.tfAs)
                self.v_s_new_target,self.idx_sAs_target,self.logits_sAs_final_target,self.probs_sAs_target = \
                    transform_q_tf(self.q_sAs_target,self.logits_sAs_target,self.tf_direc,self.tfmask,self.temp_target)

            self.q_loss = tf.reduce_mean(tf.square(self.q_sa_eval - self.tfq_sa))
            self.train_q_op = tf.train.AdamOptimizer(self.lr).minimize(self.q_loss)

            self.v_bs_loss = tf.reduce_mean(tf.square(self.v_s_eval - self.v_s_new_eval))
            self.train_v_bs_op = tf.train.AdamOptimizer(self.lr).minimize(self.v_bs_loss)
            
            eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='eval')
            target_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope='target')
            self.copy_ops = [t.assign(e) for e,t in zip(eval_params,target_params)]
            self.alpha = 0.1
            self.update_ops = [t.assign(e*self.alpha+t*(1-self.alpha)) \
                               for e,t in zip(eval_params,target_params)]
            
            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.temp_eval.assign(-5))
            self.sess.run(self.copy_ops)

    def partial_restore(self,fname=''):
        super().partial_restore(fname)
        self.sess.run(self.copy_ops)

    def learn(self):
        for _ in range(self.learn_iter):
            exp = self.get_experience_sample()
            if exp is not None:
                if self.learn_step_counter % self.update_target_iter == 0:
                    self.sess.run(self.update_ops)
                    
                s_batch,a_batch,r_batch,e_batch = [],[],[],[]
                next_s_batch,next_As_batch,next_direc_batch,next_mask_len_batch = [],[],[],[]
                for sample in exp:
                    state_vec,action_vec,reward,is_terminated,next_direc,next_state_vec,next_actions_vec = sample
                    if next_actions_vec is not None:
                        next_mask_len_batch.append(next_actions_vec.shape[1])
                    else:
                        next_mask_len_batch.append(0)
                max_len = np.max(next_mask_len_batch)
                    
                for sample in exp:
                    state_vec,action_vec,reward,is_terminated,next_direc,next_state_vec,next_actions_vec = sample
                    
                    ## do rotation for better learning process
                    rotation = np.random.choice(4)
                    if rotation > 0:
                        rotate_direc = (1 if rotation % 2 == 0 else -1)
                        reward = reward * rotate_direc
                        next_direc = next_direc * rotate_direc
                        state_vec = rotate_state(state_vec,rotation)
                        if not is_terminated:
                            next_state_vec = rotate_state(next_state_vec,rotation)
                            
                    s_batch.append(state_vec)
                    a_batch.append(action_vec)
                    r_batch.append(reward)
                    e_batch.append(is_terminated)
                    next_direc_batch.append(next_direc)
                    next_s_batch.append(next_state_vec)
                    
                    if next_actions_vec is not None:
                        n_pad = max_len - next_actions_vec.shape[1]
                        next_actions_vec = np.concatenate([next_actions_vec,np.zeros((1,n_pad,55))],axis=1)
                    else:
                        next_actions_vec = np.zeros((1,max_len,55))
                    
                    next_As_batch.append(next_actions_vec)
                    
                # v_target, = self.faster_run([self.v_s_target,],
                #                            {self.tfs:np.concatenate(next_s_batch,axis=0),
                #                             self.tfAs:np.concatenate(next_As_batch,axis=0),
                #                             self.tfmask_len:np.array(next_mask_len_batch),
                #                             self.tf_direc:np.array(next_direc_batch),
                #                            })
                _,v_target = self.sess.run([self.train_v_bs_op,self.v_s_target],
                                            {self.tfs:np.concatenate(next_s_batch,axis=0),
                                            self.tfAs:np.concatenate(next_As_batch,axis=0),
                                            self.tfmask_len:np.array(next_mask_len_batch),
                                            self.tf_direc:np.array(next_direc_batch),
                                           })
                q_batch = np.array(r_batch) + (1 - np.array(e_batch)) * v_target
                    
                self.sess.run(self.train_q_op,{self.tfs:np.concatenate(s_batch,axis=0),
                                            self.tfa:np.concatenate(a_batch,axis=0),
                                            self.tfq_sa:q_batch,
                                            })
                self.learn_step_counter += 1

class MCTSModel(BaseModel):

    def __init__(self,**kwargs):
        super().__init__(**kwargs)
           
        self.exp_sample_size = kwargs.get('exp_sample_size',1024)

        with self.graph.as_default():
            ## labels
            self.tfv_s = tf.placeholder(tf.float32, [None, ], 'values')
            self.tfprobs_sAs = tf.placeholder(tf.float32, [None, None], 'probs')
            self.tfq_sAs = tf.placeholder(tf.float32, [None, None], 'q_values')

            self.v_loss = tf.reduce_mean(tf.square(self.v_s_eval - self.tfv_s))
            self.prob_loss = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.tfprobs_sAs,logits=self.logits_sAs_final_eval)
            )
            self.q_loss_all = self.q_sAs_eval - self.tfq_sAs
            self.q_loss_all = self.q_loss_all - tf.expand_dims(tf.reduce_mean(self.q_loss_all,axis=-1),axis=-1)
            self.q_loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.q_loss_all) * self.tfmask,axis=-1))
            self.v_bs_loss = tf.reduce_mean(tf.square(self.v_s_eval - self.v_s_new_eval))

            self.train_v_op = tf.train.AdamOptimizer(self.lr).minimize(self.v_loss)
            self.train_prob_op = tf.train.AdamOptimizer(self.lr).minimize(self.prob_loss)
            self.train_joint_op = tf.train.AdamOptimizer(self.lr).minimize(1e0 * self.prob_loss + 1e-4 * self.v_loss + 1e-8 * self.q_loss + 1e-10 * self.v_bs_loss)

            self.sess.run(tf.global_variables_initializer())
            self.sess.run(self.temp_eval.assign(-5))

            self.predict_batch_out = [t._as_tf_output() for t in [self.v_s_eval,self.probs_sAs_eval]]

    def predict_batch(self,state_vec_batch,actions_vec_batch,mask_len_batch,direc_batch):
        vs,probs = self.even_faster_run(self.predict_batch_out,
                                      {self.tfs_out:state_vec_batch,
                                       self.tfAs_out:actions_vec_batch,
                                       self.tfmask_len_out:mask_len_batch,
                                       self.tf_direc_out:direc_batch})
        return vs,probs

    def extract_batch(self,exp):
        if exp is not None:
            s_batch,As_batch,direc_batch,mask_len_batch,value_batch,probs_batch,q_batch = [],[],[],[],[],[],[]
            for sample in exp:
                state_vec,actions_vec,direc,value,a_probs,q_values = sample
                mask_len_batch.append(actions_vec.shape[1])
            max_len = np.max(mask_len_batch)
            
            for sample in exp:
                state_vec,actions_vec,direc,value,a_probs,q_values = sample

                # rotation = np.random.choice(4)
                # if rotation > 0:
                #     rotate_direc = (1 if rotation % 2 == 0 else -1)
                #     direc = direc * rotate_direc
                #     value = value * rotate_direc
                #     state_vec = rotate_state(state_vec,rotation)

                s_batch.append(state_vec)
                direc_batch.append(direc)
                value_batch.append(value)
                
                n_pad = max_len - actions_vec.shape[1]
                if n_pad > 0:
                    actions_vec = np.concatenate([actions_vec,np.zeros((1,n_pad,55))],axis=1)
                    a_probs = np.concatenate([a_probs,np.zeros((n_pad,))],axis=0)
                    q_values = np.concatenate([q_values,np.zeros((n_pad,))],axis=0)
                
                As_batch.append(actions_vec)
                probs_batch.append(a_probs)
                q_batch.append(q_values)

            return s_batch,As_batch,direc_batch,mask_len_batch,value_batch,probs_batch,q_batch
        else:
            return None,None,None,None,None,None,None

    def learn(self):
        prob_loss_batches,value_loss_batches = [],[]
        for _ in range(self.learn_iter):
            exp = self.get_experience_sample(exp_pool_id=0)
            if exp is not None:
                s_batch,As_batch,direc_batch,mask_len_batch,value_batch,probs_batch,q_batch = \
                    self.extract_batch(exp)
                _,value_loss_batch,prob_loss_batch = self.sess.run([self.train_joint_op,self.v_loss,self.prob_loss],{self.tfs:np.concatenate(s_batch,axis=0),
                                             self.tfAs:np.concatenate(As_batch,axis=0),
                                             self.tfprobs_sAs:np.stack(probs_batch,axis=0),
                                             self.tfq_sAs:np.stack(q_batch,axis=0),
                                             self.tfv_s:np.array(value_batch),
                                             self.tfmask_len:np.array(mask_len_batch),
                                             self.tf_direc:np.array(direc_batch),
                                            })
                prob_loss_batches.append(prob_loss_batch)
                value_loss_batches.append(value_loss_batch)
                self.learn_step_counter += 1

        return prob_loss_batches,value_loss_batches

    def eval(self):
        prob_loss_batches,value_loss_batches = [],[]
        exp = self.get_experience_sample(exp_pool_id=0)
        if exp is not None:
            s_batch,As_batch,direc_batch,mask_len_batch,value_batch,probs_batch,q_batch = \
                    self.extract_batch(exp)
            value_loss_batch,prob_loss_batch = self.sess.run([self.v_loss,self.prob_loss],{self.tfs:np.concatenate(s_batch,axis=0),
                                            self.tfAs:np.concatenate(As_batch,axis=0),
                                            self.tfprobs_sAs:np.stack(probs_batch,axis=0),
                                            self.tfq_sAs:np.stack(q_batch,axis=0),
                                            self.tfv_s:np.array(value_batch),
                                            self.tfmask_len:np.array(mask_len_batch),
                                            self.tf_direc:np.array(direc_batch),
                                        })
            prob_loss_batches.append(prob_loss_batch)
            value_loss_batches.append(value_loss_batch)

        return prob_loss_batches,value_loss_batches

