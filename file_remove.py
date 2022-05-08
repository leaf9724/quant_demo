import os
# file_remove()
# if file[-2: ] == 'py':
#     continue   #过滤掉改名的.py文件
# name = file.replace('5day', '1day')   #去掉空格
def file_remove(main_path):
    os.chdir(main_path)
    for file in os.listdir(main_path):
        if file[-3:] == 'csv' :
            os.remove(os.path.join(main_path,file))

file_remove(main_path)
file_remove()
        # continue
        # st_name = file
        # for j in os.listdir(main_path+'/'+file):
        # if j in ['__pycache__','run_and_check.sh','tools.py','feature_test.txt']:
        #     continue
        # if j not in [file+'.py',file+'.txt']:
        #     # print(j)
        #     os.chdir(main_path+'/'+file)
        #     print(st_name)
        #     if j[-2: ] == 'py':
        #     os.rename(j, st_name +'.py')

        #     if j[-3:] == 'txt':
        #     os.rename(j, st_name +'.txt')