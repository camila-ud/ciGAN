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

    return CiGAN(save_name, load_name, patch_size, epochs,batch_size, 
                  new_model, train_vgg=train_vgg, load_vgg=load_vgg,
                  load_weights=load_weights,l1_factor = l1_factor, type = type_,
                  save_model = save_model)


def experiment_opt(type_,learning_rate,opt):
    data = []    
    for learn_rate in learning_rate:
        name = "{}_{}_{:.1e}".format(type_,opt,learn_rate)
        print("Begin : ", name)
        model = build_cigan(type_,save_name)
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
    exp = ["lsgan","mammo","dsgan"]
    for loss in exp:
        lr = [1e-5,5e-5,1e-4]
        experiment_opt(loss,lr,"rms")   
        experiment_opt(loss,lr,"adam")  
    #end experiment
    
    
    
    

