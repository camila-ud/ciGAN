#version T1 to 
from prepare import *
from model import *
from config import *

import tensorflow as tf
import numpy as np
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import matplotlib.pyplot as plt
class CiGAN:

    def __init__(self, save_name,load_name,patch_size,num_iterations,
                batch_size,new_model,train_vgg,load_vgg,load_weights,l1_factor,
                save_model,type = "lsgan",inside = True,learn_rate = 1e-4):
        self.save_name = save_name
        self.patch_size = patch_size
        self.load_name = load_name
        self.learn_rate = None
        self.num_iterations = num_iterations
        self.batch_size = batch_size
        self.new_model = new_model
        self.train_vgg = train_vgg
        self.load_vgg = load_vgg
        self.load_weights = load_weights
        self.learn_rate = learn_rate
        
        self.input_x = self.input_mask = self.input_real = self.input_boundary = None
        self.fake_image = self.global_step = None
        self.t_vars = self.d_vars = self.g_vars  = None
        self.D_real = self.D_fake = self.D_logits_fake = self.D_logits_real = self.discriminator = None
        self.saver = self.d_saver = self.g_saver = None
        self.G_loss = self.D_loss = None
        self.G_solver = self.D_solver = self.G_loss_vgg = None
        self.VGG_solver = self.boundary_loss = self.boundary_solver = self.L1_loss = self.L1_solver = None
        #juss VGG
        self.inside = inside
        

        # L1 and boundary loss params
        self.alpha = 0.95
        self.l1_factor = l1_factor
        self.boundary_factor = 1200.0
        self.save_model = save_model

        #type of training
        self.type = type

    def vgg_loss(self):
        #Features extraction and build loss function VGG model
        #----------- Build VGG networks -----------------
        
        vgg_real_c = build_vgg19(tf.multiply(self.input_real, 1 - self.input_mask))
        vgg_fake_c = build_vgg19(tf.multiply(self.fake_image, 1 - self.input_mask), reuse=True)

        #    Extract VGG weights
        G_loss_vgg = tf.reduce_mean(tf.abs(vgg_real_c['input'] - vgg_fake_c['input']))
        
        vgg_real = build_vgg19(tf.multiply(self.input_real, self.input_mask))
        vgg_fake = build_vgg19(tf.multiply(self.fake_image, self.input_mask), reuse=True)
        
        #add importance inside 100 no mejora
        G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real['input'] - vgg_fake['input']))
        
        for i in range(1, 4):
            conv_str = 'pool' + str(i)
            #add importance ?
            G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real[conv_str] - vgg_fake[conv_str]))

        vgg_real = build_vgg19(tf.multiply(self.input_real, self.input_boundary))
        vgg_fake = build_vgg19(tf.multiply(self.fake_image, self.input_boundary), reuse=True)
        
        G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real['input'] - vgg_fake['input']))
        for i in range(1, 4):
            conv_str = 'pool' + str(i)
            G_loss_vgg += tf.reduce_mean(tf.abs(vgg_real[conv_str] - vgg_fake[conv_str]))
        return G_loss_vgg
    
    def vgg_loss_inside(self):
        #Features extraction and build loss function VGG model       
        #----------- Build VGG networks -----------------     
        vgg_real = build_vgg19(tf.multiply(self.input_real, self.input_mask))
        vgg_fake = build_vgg19(tf.multiply(self.fake_image, self.input_mask), reuse=True)
        
        #First layer
        G_loss_vgg = tf.reduce_mean(tf.abs(vgg_real['input'] - vgg_fake['input']))
        
        for i in range(1, 4):
            conv_str = 'pool' + str(i)
            #get pool
            G_loss_vgg += (self.l1_factor)*tf.reduce_mean(tf.abs(vgg_real[conv_str] - vgg_fake[conv_str]))
        return G_loss_vgg


    def set_loss_function(self):
        #method for testing different loss functions
        #loss function pretraining
        if self.inside:
            self.G_loss_vgg = self.vgg_loss() 
        else:
            print("VGG just inside")
            self.G_loss_vgg = self.vgg_loss_inside()
        
        if self.type =="lsgan":
            #(lsgan)
            # test n4 smooth 0.8
            self.G_loss = tf.reduce_mean(tf.nn.l2_loss(self.D_logits_fake - tf.ones_like(self.D_logits_fake))) 
            D_loss_real = tf.reduce_mean(tf.nn.l2_loss(self.D_logits_real - tf.ones_like(self.D_logits_real))) 
            D_loss_fake = tf.reduce_mean(tf.nn.l2_loss(self.D_logits_fake - tf.zeros_like(self.D_logits_fake))) 
            self.D_loss = D_loss_real + D_loss_fake
            
            ##add article
            # L1 and boundary loss --loss for generator
            self.L1_loss = self.l1_factor * \
                        tf.reduce_mean(tf.abs(self.alpha *
                        tf.multiply(self.input_mask, self.fake_image - self.input_real)) +
                        tf.abs((1 - self.alpha) *
                        tf.multiply(1 - self.input_mask, self.fake_image - self.input_real)))
            
            self.boundary_loss = self.boundary_factor * tf.reduce_mean(tf.multiply(self.input_boundary,
                                                                                   tf.abs(self.fake_image - self.input_real)))
            
            #add all losses
            self.G_loss += self.G_loss_vgg
            self.G_loss += self.L1_loss
            self.G_loss += self.boundary_loss
            
        elif self.type == "dcgan":
             #advesarial loss (sigmoid cross entropy)
            self.G_loss = dcgan_function(self.D_logits_fake,tf.ones_like(self.D_logits_fake))
            self.D_loss = dcgan_function(self.D_logits_fake,tf.zeros_like(self.D_logits_fake)) 
            self.D_loss += dcgan_function(self.D_logits_real,tf.ones_like(self.D_logits_fake)*0.9) #smooth
            self.D_loss /= 2

        elif self.type == "mammo":      
            # In the article 4 loss functions are proposed      
             #advesarial loss (sigmoid cross entropy)
            self.G_loss = dcgan_function(self.D_logits_fake,tf.ones_like(self.D_logits_fake))
            self.D_loss = dcgan_function(self.D_logits_fake,tf.zeros_like(self.D_logits_fake)) 
            self.D_loss += dcgan_function(self.D_logits_real,tf.ones_like(self.D_logits_fake)*0.9) #smooth
            self.D_loss /= 2
                       
            # L1 and boundary loss --loss for generator
            self.L1_loss = self.l1_factor * \
                        tf.reduce_mean(tf.abs(self.alpha *
                        tf.multiply(self.input_mask, self.fake_image - self.input_real)) +
                        tf.abs((1 - self.alpha) *
                        tf.multiply(1 - self.input_mask, self.fake_image - self.input_real)))
            self.boundary_loss = self.boundary_factor * tf.reduce_mean(tf.multiply(self.input_boundary,  
                                                                                   tf.abs(self.fake_image - self.input_real)))
            
            #add all losses
            self.G_loss += self.G_loss_vgg
            self.G_loss += self.L1_loss
            self.G_loss += self.boundary_loss
        print("Losses {} have been configured".format(self.type))

           
    def set_optimizer(self):
        # Set pretrained Parameter by default ( articles)
        if self.type == 'dcgan':
            self.G_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.G_loss,
                                            var_list=self.g_vars,global_step=self.global_step)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=2e-4, beta1=0.5).minimize(self.D_loss,
                                            var_list=self.d_vars,global_step=self.global_step)
            self.VGG_solver = tf.train.AdamOptimizer(learning_rate=2e-4).minimize(self.G_loss_vgg, 
                                            global_step=self.global_step, var_list=self.g_vars)

        elif self.type == 'lsgan':
            #best after different tests
            self.G_solver = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.G_loss,
                                            var_list=self.g_vars,global_step=self.global_step) 
            self.D_solver = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.D_loss,
                                            var_list=self.d_vars,global_step=self.global_step)
            self.VGG_solver = tf.train.RMSPropOptimizer(learning_rate=self.learn_rate).minimize(self.G_loss_vgg, 
                                            global_step=self.global_step, var_list=self.g_vars)
        elif self.type == "mammo":
            self.G_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.G_loss, 
                                            var_list=self.g_vars, global_step=self.global_step)
            self.D_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.D_loss, 
                                            var_list=self.d_vars, global_step=self.global_step)
            self.VGG_solver = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(self.G_loss_vgg, 
                                            global_step=self.global_step, var_list=self.g_vars)
        print("Solver {} has been configured".format(self.type))
    
    def set_new_optimizer(self, optimizer):
        self.G_solver = optimizer.minimize(self.G_loss,
                                               var_list=self.g_vars,global_step=self.global_step)
        self.D_solver = optimizer.minimize(self.D_loss,
                                                var_list=self.d_vars,global_step=self.global_step)
        self.VGG_solver = optimizer.minimize(self.G_loss_vgg, 
                                            global_step=self.global_step, var_list=self.g_vars)
    
    def build_model(self,batch_normalization = False):
         # Learning rate params
        self.global_step = tf.Variable(0, trainable=False)        
        #self.learn_rate = learn_rate
        #input placeholders
        self.input_x = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_mask = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_real = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        self.input_boundary = tf.placeholder(tf.float32, [None, self.patch_size, self.patch_size, 1])
        
        #generator
        self.fake_image = build_generator(self.input_x, self.input_mask,batch_normalization = batch_normalization)

        #discriminator
        self.D_real, self.D_logits_real = build_discriminator(self.input_real)
        self.D_fake, self.D_logits_fake = build_discriminator(self.fake_image,reuse=True)
     
        #set training variables 
        self.t_vars = tf.trainable_variables()
        self.d_vars=[var for var in self.t_vars if 'dis' in var.name]
        self.g_vars=[var for var in self.t_vars if 'gen' in var.name]

        #savers
        self.saver = tf.train.Saver(self.g_vars + self.d_vars, max_to_keep=10000)
        self.g_saver = tf.train.Saver(self.g_vars)
        self.d_saver = tf.train.Saver(self.d_vars)       
         # Set losses
        self.set_loss_function()
        #set optimizer
        self.set_optimizer()
    
    
    def train_model(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            # If using existing model
            if not self.new_model:
                # Load the VGG loss trained model (model pretrain in model_wgan_vgg)
                if self.load_vgg and tf.train.checkpoint_exists(models_dir + self.load_name + '_vgg'):
                    print('Loading vgg')
                    self.g_saver.restore(self.sess, models_dir + self.load_name + '_vgg')
                # Load the GAN loss trained model
                elif self.load_name is not None and self.load_weights:
                    print('Loading model', self.load_name)
                    path = models_dir + self.load_name
                    self.g_saver.restore(self.sess, path)
                    self.d_saver.restore(self.sess, path)

            # Create data generators
            # Generators generate 256x256px patches
            print('Getting dataset')
            data_generator = generate_cpatches(self.batch_size)
            val_data_generator = generate_cpatches(1)

            # iteration number
            it = 0
            # Number of iterations per epoch for each type of loss
            d_iters = 1
            g_iters = 1
            vgg_iters = 10
            
           #history -------
            d_data = []
            g_data = []
            #------------
            print('Training model')
            # First train on VGG loss only
            if self.train_vgg:
                print('Pre-training on VGG')
                for i in range(vgg_iters):
                    print('Iteration', i,vgg_iters)
                    VGG_loss_cur = self.train(self.sess, self.VGG_solver, self.G_loss_vgg, generator=data_generator, iters=100)
                    print('VGG loss', VGG_loss_cur)
                    self.validate(i, val_data_generator, self.sess)
                    if self.save_model:
                        save(self.save_name + '_vgg', it, self.g_saver, self.sess)
                    
            else:
                print('VGG Pretrained')

            while it < int(self.num_iterations):
                it += 1
                for j in range(1):
                    D_loss_cur = self.train(self.sess, self.D_solver, self.D_loss, generator=data_generator, iters=d_iters)
                    print('D_loss', D_loss_cur,it)
                    it += d_iters
                    d_data.append([it,D_loss_cur])

                    print('========')
                    G_loss_cur = self.train(self.sess, self.G_solver, self.G_loss, generator=data_generator, iters=g_iters)
                    print('G loss', G_loss_cur)
                    it += g_iters
                    g_data.append([it,G_loss_cur])
                    print('========')

                if it % 100 == 0 and it < 5000:
                    #random or not, test with id 10
                    #___random
                    #self.validate(it, val_data_generator, self.sess)
                    #___id choix 10
                    self.validate(it, val_data_generator , self.sess, 
                                  randn = False, id_choix = 20)
                    print("end:",self.num_iterations -it)
                elif it % 3500 == 0 :
                    self.validate(it, val_data_generator , self.sess, 
                                  randn = False, id_choix = 20)
                    print("end:",self.num_iterations -it)
                    
                #self.save_loss(data = np.asarray([vgg_data,d_data,g_data]),legend = ['vgg','discriminator','generator'])
            if self.save_model:
                print("Saving model")
                save(self.save_name, it, self.saver, self.sess)     
            
            #self.plot_loss(np.stack([d_data,g_data]))
            #np.savez_compressed('loss{}'.format(self.type),loss = np.stack([d_data,g_data]))
        tf.reset_default_graph()
        return np.stack([d_data,g_data])


        #         # VGG LOSS
        #         for i in range(0):
        #             VGG_loss_cur = self.train(self.sess, self.VGG_solver, self.G_loss_vgg, generator=data_generator, iters=vgg_iters)
        #             it += vgg_iters
        #             print('VGG loss', VGG_loss_cur)
        #             vgg_data.append([it,VGG_loss_cur])

        #         # BOUNDARY LOSS      
        #         for i in range(0):
        #              boundary_loss_cur = self.train(self.sess, self.boundary_solver, self.boundary_loss, generator=data_generator, iters=boundary_iters)
        #              it += boundary_iters
        #              print('Boundary loss', boundary_loss_cur)
        #              boundary_data.append([it,boundary_loss_cur])
        #         # #L1 LOSS
        #         for i in range(0):
        #              L1_loss_cur = self.train(self.sess, self.L1_solver, self.L1_loss, generator=data_generator, iters=boundary_iters)
        #              it += boundary_iters
        #              print('L1 loss', L1_loss_cur)
        #              l1_data.append([it,L1_loss_cur])
        
        #         if it % 50 == 0:
        #            self.validate(it, val_data_generator, self.sess)

        #     print('Saving model')
        #     #self.save_loss(data = np.asarray([vgg_data,d_data,g_data]),legend = ['vgg','discriminator','generator'])
        #     save(self.save_name, it, self.saver, self.sess)     
        #     np.savez_compressed('loss',d = d_data,g=g_data)
        # tf.reset_default_graph()


    def plot_loss(self,data):
        fig,ax = plt.subplots(2,1,figsize = (15,5))
        ax[0].plot(data[0,:,0],data[0,:,1])
        ax[0].set_title('Discriminator loss')
        
        ax[1].plot(data[1,:,0],data[1,:,1])
        ax[1].set_title('Generator loss')
        
        plt.title("Loss functions ({})".format(self.type))
        fig.savefig('loss{}.png'.format(self.type))   # save the figure to file
        plt.close(fig)    # close the figure window


    def validate_model(self):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            self.g_saver.restore(self.sess, models_dir + self.load_name)
            self.d_saver.restore(self.sess, models_dir + self.load_name)
            generator_syn =  generate_cpatches(batch_size)
            self.validate(1, generator_syn, self.sess)
        

    def synthesis(self,i):
        with tf.Session() as self.sess:
            self.sess.run(tf.global_variables_initializer())
            self.g_saver.restore(self.sess, models_dir + self.load_name)
            self.d_saver.restore(self.sess, models_dir + self.load_name)
            
            generator_syn = generate_nc_patches(1)
            
            data_X  = next(generator_syn)
            data_x = data_X[0:1, :, :, 0:1]
            data_mask = data_X[0:1, :, :, 1:2]
            data_real = data_X[0:1, :, :, 2:3]
            input_image = data_x[0, :, :, 0]
            real_image = data_real[0, :, :, 0]
           
            pred_img = self.sess.run(self.fake_image, feed_dict={
                    self.input_x: data_x,
                    self.input_mask: data_mask
                   })
            pred_img = pred_img[0, :, :, 0]
            img = np.concatenate((real_image,input_image, pred_img), axis=1)
            img = scipy.ndimage.zoom(img, zoom=[0.75, 0.75])
            directory = './synthesis/' + self.save_name + '/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            print(np.unique(pred_img))
            scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(directory + self.save_name + '_' + str(self.patch_size) + '_' + str(i) + '.png')
        

    def synthesize_dataset(self, num, batch_size=batch_size):
        num = int(num)
        with tf.Session() as self.sess:
            # Load model
            self.sess.run(tf.global_variables_initializer())
            self.g_saver.restore(self.sess, models_dir + self.load_name)
            self.d_saver.restore(self.sess, models_dir + self.load_name)

             # Create patch generator
            generator_syn = generate_nc_patches(batch_size)

            X_train = np.zeros((num, batch_size, self.patch_size, self.patch_size, 1))

            for i in range(0, num):
                print('Num ', i)
                data_X = next(generator_syn)

                # Use normals to generate malignant, and vice versa
                data_x = data_X[:, :, :, 0:1]
                data_mask = data_X[:, :, :, 1:2]
                pred_img = self.sess.run(self.fake_image, feed_dict={
                    self.input_x: data_x,
                    self.input_mask: data_mask
                   })
                X_train[i] = pred_img

            X_train = X_train.reshape((-1, self.patch_size, self.patch_size, 1))
            directory = './synthesis/'
            if not os.path.exists(directory):
                os.makedirs(directory)
            np.savez_compressed('{}samples_sinthesizes{}'.format(directory,num), samples = X_train)
            print("samples_generated")
        

    def train(self, sess, solver, loss, generator, step=0, iters=10, return_acc=False):
        loss_avg = []
        accs = []
        for i in range(iters):
            data_X = next(generator)
            #pdb.set_trace()
            data_x = data_X[:, :, :, 0:1]
            data_mask = data_X[:, :, :, 1:2]
            data_real = data_X[:, :, :, 2:3]
            data_boundary = data_X[:, :, :, 3:4]
            _, loss_cur = sess.run([solver, loss],
               feed_dict={
               self.input_x: data_x,
               self.input_mask: data_mask,
               self.input_real: data_real,
               self.input_boundary: data_boundary
               })
            loss_avg.append(loss_cur)
            #if return_acc:
                #data_c = data_c.reshape((-1, c_dims))
                #attr_s = one_hot(attr)
                #accs.append(metrics.accuracy_score(data_c, attr_s))          

        if return_acc:
            return (np.mean(loss_avg), np.mean(accs))
        else:
            return np.mean(loss_avg)

    def validate(self, i, data_generator, sess,randn = True, id_choix = 0):
        print('Validating', i)
        if randn:
            data_X = next(data_generator)
        else:
            data_X = generate_patch_id(id_choix)
            
        data_x = data_X[0:1, :, :, 0:1]
        data_mask = data_X[0:1, :, :, 1:2]
        data_real = data_X[0:1, :, :, 2:3]
        mask_image = data_mask[0, :, :, 0]
        real_image = data_real[0, :, :, 0]
        
        output_mal_mass = sess.run(self.fake_image, feed_dict={
            self.input_x: data_x,
            self.input_mask: data_mask
           })
        #output_ben_mass = output_ben_mass[0, :, :, 0]
        output_mal_mass = output_mal_mass[0, :, :, 0]
        img = np.concatenate((real_image, mask_image, output_mal_mass), axis=1)
        img = scipy.ndimage.zoom(img, zoom=[0.75, 0.75])
        directory = './validation/' + self.save_name + '/'
        if not os.path.exists(directory):
            os.makedirs(directory)
        print(np.unique(output_mal_mass))
        scipy.misc.toimage(img, cmin=0.0, cmax=1.0).save(directory + self.save_name + '_' + str(self.patch_size) + '_' + str(i) + '.png')
