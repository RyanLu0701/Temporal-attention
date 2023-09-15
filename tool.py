import h5py
import numpy as np
import pandas as pd
import tqdm

def time_embedding(timedata):

   date =  [(i.hour, i.minute) for i in timedata]

   return np.array(date)

def get_data(data,pv_data,time,time_stampe):

    data_length = len(np.array(data)) // time_stampe

    data_images_log = np.array(data)[:data_length*time_stampe]

    data_pv_log     = np.array(pv_data)[:data_length*time_stampe]

    data_images     = data_images_log.reshape(data_length,time_stampe,64,64,3)

    data_pv_shifted_data  = np.roll(data_pv_log, -(time_stampe), axis=0)[:data_length*time_stampe]
    data_pv_reshaped_data = data_pv_shifted_data.reshape(data_length, time_stampe)

    date = time_embedding(time)
    date_time_stampe = date[:data_length*time_stampe].reshape(data_length ,time_stampe,2)

    data_pv = np.array([i[0] for i in data_pv_reshaped_data])

    return data_images,data_pv,date_time_stampe

def read_data(time_stampe):# 記得檢查是否有兌到資料

    import h5py
    import numpy as np
    import pandas as pd

    test_time = np.load(r"Solar_standford\times_test.npy", allow_pickle=True)
    train_time = np.load(r"Solar_standford\times_trainval.npy", allow_pickle=True)

    hf = h5py.File('Solar_standford/2017_2019_images_pv_processed.hdf5', 'r')
    train = hf.get('trainval/')  # train:{"image_log:image_data", "pv_log":pv_data}
    test = hf.get('test/')


    train_images ,train_pv ,train_date = get_data(train["images_log"],train["pv_log"],train_time,time_stampe)
    test_images ,test_pv ,test_date     = get_data(test["images_log"],test["pv_log"],test_time,time_stampe)



    print(f"train images data shape : {train_images.shape}")
    print(f"train pv data shape : {train_pv.shape}")
    print(f"test images data shape : {test_images.shape}")
    print(f"test pv data shape : {test_pv.shape}")



    return train_images, test_images, train_pv , test_pv , train_date,test_date

