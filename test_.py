from train import *

patch_size = 256
#learn_rate = 1e-4
learn_rate = 1e-4
batch_size = 8
epochs =  10000
l1_factor = 600.0

if __name__ == '__main__':
    # model name to load?
    #first model 
    # only validating the model
    if '-loss' in sys.argv:
        loss = ["wgan","lsgan","dcgan","mammo"]
        for i in loss:        
            load_name = 'model_{}'.format(i)
            # new model save name
            save_name = 'model_{}'.format(i)
            # pretrain with VGG
<<<<<<< HEAD
<<<<<<< HEAD
            
            train_vgg = False
            # load VGG pre-trained model
            load_vgg = False
            
            if i == "wgan":
                print("Pretrain")
                #just pretrained one time
                train_vgg = True
                 # load VGG pre-trained model
                load_vgg = True
                
=======
            train_vgg = True
            # load VGG pre-trained model
            load_vgg = True
>>>>>>> c34e5058343bdda2385c0429ec88da172226af3c
=======
            train_vgg = True
            # load VGG pre-trained model
            load_vgg = True
>>>>>>> c34e5058343bdda2385c0429ec88da172226af3c
            # load GAN model weights
            load_weights = False
            # save new model?
            new_model = True
            GAN = CiGAN(save_name, load_name, patch_size, epochs,
                        batch_size, new_model, train_vgg=train_vgg, load_vgg=load_vgg,
<<<<<<< HEAD
<<<<<<< HEAD
                        load_weights=load_weights,l1_factor = 600.0, type = i,save_model = False)
=======
                        load_weights=load_weights,l1_factor = i, save_model = False)
>>>>>>> c34e5058343bdda2385c0429ec88da172226af3c
=======
                        load_weights=load_weights,l1_factor = i, save_model = False)
>>>>>>> c34e5058343bdda2385c0429ec88da172226af3c

            GAN.build_model()
            GAN.train_model()
            print("model",i)
    
    
    elif '-factor' in sys.argv:
        l1_test = [600.0,1200.0,1800.0]
        for i in l1_test:        
            load_name = 'model_gan{}'.format(i)
            # new model save name
            save_name = 'model_gan{}'.format(i)
            # pretrain with VGG
            train_vgg = True
            # load VGG pre-trained model
            load_vgg = True
            # load GAN model weights
            load_weights = False
            # save new model?
            new_model = True


            GAN = CiGAN(save_name, load_name, patch_size, epochs,
                        batch_size, new_model, train_vgg=train_vgg, load_vgg=load_vgg,
                        load_weights=load_weights,l1_factor = i, save_model = False)

            GAN.build_model()
            GAN.train_model()
            print("model",i)
    
    print("Please select -loss or -factor for testing")
