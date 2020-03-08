from train import *

"experiment"
patch_size = 256
#learn_rate = 1e-4
batch_size = 8
epochs =  4500
l1_factor = 1200.0


def build_cigan(type_,save_name,vgg = True,save_model = False):        
    load_weights = False
    # save new model?
    new_model = True
    # pretrain with VGG            
    train_vgg = vgg
    load_vgg = vgg
    load_name = save_name
    return CiGAN(save_name, load_name, patch_size, epochs,batch_size, 
                  new_model, train_vgg=train_vgg, load_vgg=load_vgg,
                  load_weights=load_weights,l1_factor = l1_factor, type = type_,
                  save_model = save_model)


def experiment_opt(type_,learning_rate,opt,vgg = True):
    data = []    
    for learn_rate in learning_rate:
        name = "{}_{}_{:.1e}{}".format(type_,opt,learn_rate,int(vgg))
        print("Begin : ", name)
        model = build_cigan(type_,name,vgg = vgg)
        model.build_model(batch_normalization=True)
        
        if opt == "adam":
            print("{} selected".format(opt))
            optimizer = tf.train.AdamOptimizer(learning_rate=learn_rate)
        
        elif opt == "rms":
            print("{} selected".format(opt))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learn_rate)
        
        model.set_new_optimizer(optimizer)
        results = model.train_model()  
        data.append(results)   
        
    data = np.stack(data)
    #end experiment #end experiment
    print("EXP {} is finished".format(type_))
    directory = './results/' + type_ + '/'
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    np.savez_compressed("{}{}".format(directory,opt),loss = data)
    print("saved")

if __name__ == '__main__':
    #1.
    """
    exp = ["lsgan","mammo"]
    for loss in exp:
        lr = [1e-5,5e-5,1e-4]
        experiment_opt(loss,lr,"rms")   
        experiment_opt(loss,lr,"adam")
    """
    #2.
    """
    exp = ["dcgan"]
    for loss in exp:
        lr = [1e-5,5e-5,1e-4]
        experiment_opt(loss,lr,"rms")   
        experiment_opt(loss,lr,"adam")  
    """
    #experimento no2 sin vggpre
    exp = ["lsgan","mammo","dcgan"]
    for loss in exp:
        lr = [1e-5,5e-5,1e-4]
        experiment_opt(loss,lr,"rms",vgg = False)   
        experiment_opt(loss,lr,"adam",vgg = False)
    #end experiment
    
    
    
    

