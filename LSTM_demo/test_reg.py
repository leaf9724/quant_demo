import os
import sys
sys.path.append('/home/pc/matrad/leaf/factor/quant_demo')
from feature_manuer import Feature_engine
from LSTM_demo.solo_LSTM import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CD = Creat_LSTM_data()


def timer(func):
    def func_wrapper(*args, **kwargs):
        from time import time

        time_start = time()
        result = func(*args, **kwargs)
        time_end = time()
        time_spend = time_end - time_start
        print("\n{0} cost time {1} s\n".format(func.__name__, time_spend))
        return result

    return func_wrapper

@timer
def train_fun(file_list_train,root, target ='regression'):
    model = LSTM_Regression(input_size=CD.feature_num, hidden_size=128, output_size=1, num_layers=2,days_for_train=CD.days_for_train).to(device) # 导入模型并设置模型的参数输入输出层、隐藏层等
    model_total = sum([param.nelement() for param in model.parameters()]) # 计算模型参数
    print("Number of model_total parameter: %.8fM" % (model_total/1e6))
    train_loss = []
    loss_function = nn.L1Loss().to(device)
    # loss_function = nn.BCELoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

### 模型训练
    for i in range(200):
        for file in file_list_train:
            # print(file)
            feature_root_file = os.path.join(root,'feature_data',file)
            label_file = 'label' + file[7:]
            label_root_file = os.path.join(root,'label_data',label_file)
            feature_data = pd.read_pickle(feature_root_file)
            label_data = pd.read_pickle(label_root_file)
            data_close = pd.merge(feature_data, label_data, on=['date','code'], how='outer')
            if data_close.shape[0]<= 600:
                continue
            train_x, train_y, _ = CD.data_standard(data_close,lable='label_week_lagRet_reg')
            train_x, train_y = CD.feature_reshape(train_x, train_y)
            t0 = time.time()
            out = model.forward(train_x)
            loss = loss_function(out, train_y)
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_loss.append(loss.item())
            print(' Loss:{:.5f}'.format( loss.cpu().item()))
            # 将训练过程的损失值写入文档保存，并在终端打印出来
        with open('/home/pc/matrad/leaf/factor/quant_demo/LSTM_demo/'+label+'log.txt', 'a+') as f:
            f.write('{} - {}\n'.format(i+1, loss.cpu().item()))
        if (i+1) % 1 == 0:
            print('Epoch: {}, Loss:{:.5f}'.format(i+1, loss.cpu().item()))


    # 画loss曲线
    plt.figure()
    plt.plot(train_loss, 'b', label='loss')
    plt.title("Train_Loss_Curve")
    plt.ylabel('train_loss')
    plt.xlabel('epoch_num')
    plt.savefig('loss.png', format='png', dpi=200)
    plt.close()
    torch.save(model.state_dict(), '/home/pc/matrad/leaf/factor/quant_demo/LSTM_demo/'+label+'model_params.pkl')  # 可以保存模型的参数供未来使用

    t1=time.time()
    T=t1-t0
    print('The training time took %.2f'%(T/60)+' mins.')
    tt0=time.asctime(time.localtime(t0))
    tt1=time.asctime(time.localtime(t1))
    print('The starting time was ',tt0)
    print('The finishing time was ',tt1)


# for test
@timer
def model_test(file_list_test, threshold=0.5, evaul_plot=True):
    model = LSTM_Regression(input_size=CD.feature_num, hidden_size=128, output_size=1, num_layers=2,days_for_train=CD.days_for_train).to(device) # 导入模型并设置模型的参数输入输出层、隐藏层等
    # model = model.eval() # 转换成测试模式
    model.load_state_dict(torch.load('/home/pc/matrad/leaf/factor/quant_demo/LSTM_demo/'+label+'model_params.pkl'))  # 读取参数
    df_all = pd.DataFrame()
    for file in file_list_test:
        feature_root_file = os.path.join(root,'feature_data',file)
        label_file = 'label' + file[7:]
        label_root_file = os.path.join(root,'label_data',label_file)
        feature_data = pd.read_pickle(feature_root_file)
        label_data = pd.read_pickle(label_root_file)
        data_close = pd.merge(feature_data, label_data, on=['date','code'], how='outer')
        test_x, label, date_and_code = CD.data_standard(data_close,lable='label_week_lagRet_reg')

        test_x, label = CD.feature_reshape(test_x, label )
        pred_test = (model(test_x.to(torch.float32))).detach().cpu().numpy() # 全量训练集

        assert len(pred_test) == len(label)
        pred_prob = pred_test.reshape(-1,1)

        # pred_signal = np.sign(np.maximum(pred_test-threshold,0))
        label = label.detach().cpu().numpy()
        df = pd.DataFrame()

        df['label'] = label.squeeze(1)
        df['pred_prob'] = pred_prob.squeeze(1)
        # df['pred_signal'] = pred_signal.squeeze(1)
        assert df.values.shape[0] == date_and_code.values.shape[0]
        df = pd.concat([date_and_code,df])


        if evaul_plot:
            mae = np.absolute(pred_prob - label).sum()/len(pred_prob)
            print('mae',mae)
            plt.figure(figsize=(30,10))
            plt.plot(df['pred_prob'].values, 'r', label='prediction')
            plt.plot(df['label'].values, 'b', label='real')
            # plt.plot((train_size, test_size), (0, 1), 'g--')  # 分割线 左边是训练数据 右边是测试数据的输出
            plt.legend(loc='best')
            plt.savefig('/home/pc/matrad/leaf/factor/quant_demo/'+file+'+label'+'result.png', format='png', dpi=200)
            plt.close()

        df_all = pd.concat([df,df_all],axis=0)
    df_all.to_csv('/home/pc/matrad/leaf/factor/strategy/mosel_result/'+label+'lstm_back_pre.csv',index=False)


fm = Feature_engine()
fm.forward_feature()
fm.forward_label()
root ='/home/pc/matrad/leaf/factor/daily_data/data_processed/daily_data'
file_list_ori = os.listdir(os.path.join(root,'feature_data'))
file_list =[]
for i in file_list_ori:
    if i[-3:] == 'pkl':
        file_list.append(i)
file_list.sort(key=lambda x:int(x[-10:-4]))
train_file_list = file_list[:750]
test_file_list = file_list[750:]

train_fun(train_file_list,root)
model_test(test_file_list)