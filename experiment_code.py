from train import *
from experiment_conf import *

def build_cigan(type_,lr,name = "model", ot = False):
    if ot:
        load_name = "{}{:.1e}{}".format(type_,lr,name)
        # new model save name
        save_name = "{}{:.1e}{}".format(type_,lr,name)
     # load GAN model weights
    else: 
        load_name = "{}{:.1e}".format(type_,lr)
        # new model save name
        save_name = "{}{:.1e}".format(type_,lr)
        
    load_weights = False
    # save new model?
    new_model = True
    # pretrain with VGG            
    train_vgg = True
    load_vgg = True

    return CiGAN(save_name, load_name, patch_size, epochs,batch_size, 
                  new_model, train_vgg=train_vgg, load_vgg=load_vgg,
                  load_weights=load_weights,l1_factor = l1_factor, type = type_,
                  save_model = False)


def experiment_opt(type_,lr):
    data = []
    for learn_rate in lr:
        print("Begin : ",learn_rate)
        opt = tf.train.RMSPropOptimizer(learning_rate=learn_rate)
        model = build_cigan(type_,learn_rate)
        model.build_model()
        model.set_new_optimizer(opt)
        results = model.train_model()
        data.append(results)
    #end experiment
    data = np.stack(data)
    print("EXP {} is finished".format(type_))
    directory = './results/' + model.save_name + '/'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed("{}{}".format(directory,type_),loss = data)
    print("saved")

if __name__ == '__main__':
    # model name to load?
    #first model 
    # only validating the model
    #1 test : loss = ["dcgan","mammo"]
    #2 test : loss = ["wgan","lsgan"]
    #loss = ["wgan","lsgan"]
    #for i in loss: 
    #    lr = np.linspace(1e-5,1e-4,10)
        #test RMSPROP
    #    print(i)
    #    experiment_opt(i,lr)
    #3. experiment loss = ["wgan","lsgan","mammo"]
    #loss = ["mammo"]
    #print("mammo","exp3")
    #lr = np.linspace(1e-5,1e-4,10)
    #experiment_opt("mammo",lr)

    type_ = "lsgan"
    print("Test with batchnormalization : ")
    model = build_cigan(type_,5e-5,"rmsbn_bn",True)
    model.build_model(batch_normalization=True)
    results = model.train_model()    
    #end experiment
    print("EXP {} is finished".format(type_))
    directory = './results/' + model.save_name + '/'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed("{}{}".format(directory,type_),loss = results)
    print("saved")
    
    
    

