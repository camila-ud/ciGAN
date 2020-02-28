from train import *
from experiment_conf import *

def build_cigan(type_):
    load_name = type_
    # new model save name
    save_name = type_
     # load GAN model weights
    load_weights = False
    # save new model?
    new_model = True
    # pretrain with VGG            
    train_vgg = False
    load_vgg = False

    return CiGAN(save_name, load_name, patch_size, epochs,batch_size, 
                  new_model, train_vgg=train_vgg, load_vgg=load_vgg,
                  load_weights=load_weights,l1_factor = l1_factor, type = type_,
                  save_model = False)

def experiment_opt(type_):
    lr = np.linspace(1e-5,1e-4,10)
    data = []
    for learn_rate in lr:
        print("Begin : ",learn_rate)
        opt = tf.train.RMSPropOptimizer(learning_rate=learn_rate)
        model = build_cigan(type_)
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
    loss = ["dcgan","mammo"]
    for i in loss: 
        #test RMSPROP
        print(i)
        experiment_opt(i)
    

