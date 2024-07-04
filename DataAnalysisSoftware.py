# !pip install pyinstaller
# !pyinstaller --noconsole --onefile --icon=DataAnaylsisIcon.ico DataAnaylsis.py
#%% 모듈
import numpy as np
import pandas as pd
from math import ceil
from os import path, getcwd
from platform import system
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.tree import plot_tree
import tkinter as tk
from tkinter import scrolledtext
from tkinter import filedialog #파일 탐색기 열기
import pyperclip # 파일 경로 복사
import cv2 #이미지 처리
from PIL import Image, ImageTk #이미지 처리
#%% 경고메시지 무시
import warnings
warnings.filterwarnings('ignore')
#%% 시각화 폰트 설정
plt.rc('font', size=20)        # 기본 폰트 크기
plt.rc('xtick', labelsize=20)  # x축 눈금 폰트 크기 
plt.rc('ytick', labelsize=20)  # y축 눈금 폰트 크기
plt.rc('figure', titlesize=30) # figure title 폰트 크기
plt.rc('font', family='malgun gothic') # 한글 폰트 오류 해결
rc('axes', unicode_minus = False) # 마이너스(-) 깨짐 해결
#%% 선형회귀분석
def Linear_Regression():
    global x_train, x_test, y_train, y_test
    global df_pred
    
    #모델 적용
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    y_pred_df = pd.DataFrame(y_pred, index = x_test.index.to_list(), columns=[y_col])
    df_pred = pd.concat([y_pred_df, x_test], axis = 1)
    
    #선형회귀식
    if type(model.coef_.tolist()[0]) == list: #다중 선형 회귀
        expression_list = [float(model.intercept_)] + [i for i in model.coef_.tolist()[0]]
    elif type(model.coef_) == np.ndarray:
        expression_list = [float(model.intercept_)] + [i for i in model.coef_.tolist()]
    else: #단순 선형 회귀
        expression_list = [float(model.intercept_), float(model.coef_)]
    
    #모델평가지표
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    return mse, rmse, mae, r2, expression_list

#%% 로지스틱 회귀분석 
def Logistic_Regression():
    global x_train, x_test, y_train, y_test
    global df_pred

    #모형 적용
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(max_iter = 100)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    y_pred_df = pd.DataFrame(y_pred, index = x_test.index.to_list(), columns=[y_col])
    df_pred = pd.concat([y_pred_df, x_test], axis = 1)

    #선형 수식
    if type(model.coef_.tolist()[0]) == list: #다중 선형 회귀
        expression_list = [float(model.intercept_)] + [i for i in model.coef_.tolist()[0]]
    elif type(model.coef_) == np.ndarray:
        expression_list = [float(model.intercept_)] + [i for i in model.coef_.tolist()]
    else: #단순 선형 회귀
        expression_list = [float(model.intercept_), float(model.coef_)]
    
    #Odds Ratio
    [[TP,FN], [FP,TN]] = confusion_matrix(y_test,y_pred)
    
    if (FN * FP) == 0 :
        Odds_Ratio = 'inf'
    else : 
        Odds_Ratio = (TP*TN)/(FN*FP)
    
    #모델 평가 지표
    accuracy = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    
    return accuracy, precision, recall, classification_report(y_test, y_pred), Odds_Ratio, confusion_matrix(y_test,y_pred), expression_list

#%% DT
def Decision_Tree():
    global x_train, x_test, y_train, y_test

    #모형 적용
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state = 0, max_depth=3)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    
    #모델 평가 지표
    accuracy = accuracy_score(y_test,y_pred)
    if len(df[y_col].unique())>=3:
        precision = precision_score(y_test,y_pred,average='weighted')
        recall = recall_score(y_test,y_pred,average='weighted')
    else:
        precision = precision_score(y_test,y_pred)
        recall = recall_score(y_test,y_pred)
    
    #Entropy
    entropy = model.tree_.impurity
    
    return accuracy, precision, recall, classification_report(y_test, y_pred), entropy, model
#%% 데이터 파일 선택
# 파일 탐색기 대화 상자 열기
root = tk.Tk()
data = filedialog.askopenfilename()
root.destroy()
# 파일 경로 복사하기
pyperclip.copy(data)
print("파일 경로가 복사되었습니다.")

if system()=='Windows':
    Encoding = 'cp949'
else:
    Encoding = 'utf-8'
    
if path.splitext(data)[1] =='.csv':
    df = pd.read_csv(data, encoding = Encoding)
elif path.splitext(data)[1] =='.xlsx':
    df = pd.read_excel(data, engine='openpyxl')

#%% GUI 데이터 입력
def Dependent_Variable():
    global y_col, df_col
    y_col = entry.get()
    if y_col in df.columns:
        y = df[[y_col]]
        result_label.config(text=f"{y_col} 변수가 입력되었습니다.\n")
        replace_y_col = '"' + y_col + '", '
        df_col = df_col.replace(replace_y_col, "")
    else:
        result_label.config(text="잘못된 변수명입니다. 다시 입력해주세요.\n")
    return y_col, y

def Independent_Variable(y_col):
    global x_col, df
    x_col = entry1.get()
    if x_col == "-1":
        x = df.drop(y_col, axis=1)
        result_label1.config(text=f"\n{list(x.columns)} 변수가 입력되었습니다.\n\n* Exit버튼을 눌러주세요. *")
    elif all(item in df.columns for item in x_col.split(', ')):
        x = df[x_col.split(', ')]
        result_label1.config(text=f"\n{x_col} 변수가 입력되었습니다.\n\n* Exit버튼을 눌러주세요. *")
    else:
        result_label1.config(text="\n잘못된 변수명입니다. 다시 입력해주세요.")
    x_col = x.columns
    y = df[y_col]
    df = pd.concat([y, x], axis = 1)
    return x

def update_label():
    y_col, y = Dependent_Variable()
    label1.config(text=f"변수명 : {df_col.rstrip().rstrip(',')}\n\n독립변수에 해당하는 변수명을 적어주세요\n(종속변수를 제외한 모든 변수라면 -1 입력)\n")
    label1.pack()
    entry1.pack()
    button1.pack()
    result_label1.pack()

root = tk.Tk()
root.title("Data Variable Selection")
root.geometry("500x450+200+200")
root.resizable(True, True)
Exit_button = tk.Button(root, text='Exit', command=root.destroy)
Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정

df_col = ""
for i, col in enumerate(df.columns):
    df_col = df_col + '"' + col + '", '
    if i % 10 == 9:
        df_col += '\n'

label = tk.Label(text=f"\n변수명 : {df_col.rstrip().rstrip(',')}\n종속변수에 해당하는 변수명을 적어주세요. ")
label.pack()
entry = tk.Entry(root)
entry.pack()
button = tk.Button(width=15, text="Enter", overrelief="solid", command=Dependent_Variable)
button.pack()
result_label = tk.Label(root, text="")
result_label.pack()

label1 = tk.Label(root, text="")
entry1 = tk.Entry(root)
button1 = tk.Button(root, width=15, text="Enter", overrelief="solid", command=lambda: Independent_Variable(y_col))
result_label1 = tk.Label(root, text="")

button.config(command=update_label)

root.mainloop()

#%% GUI
#Data View
def Show_Data():
    data_window = tk.Toplevel(root)
    data_window.geometry("600x500+200+200")
    data_window.title("Data_Preview")

    # row 생략 없이 출력
    pd.set_option('display.max_rows', None)
    # col 생략 없이 출력
    pd.set_option('display.max_columns', None)
    
    text = scrolledtext.ScrolledText(data_window)
    text.config(width = 600, height = 500)
    text.pack()
    
    text.insert(tk.END, df)
    
    #읽기 전용
    text.configure(state='disabled')

    Exit_button = tk.Button(data_window, text='Exit', command=data_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)
#Data Info.
def Show_Information():
    # 새로운 창 만들기
    info_window = tk.Toplevel(root)
    info_window.geometry("600x500+200+200")
    info_window.title("Data_Summary")
    
    # Text 위젯 생성
    text = tk.scrolledtext.ScrolledText(info_window)
    text.config(width = 600, height = 500)
    text.pack()
    
    # 데이터프레임 정보 출력
    text.insert(tk.END, "[Data Describe]\n")
    text.insert(tk.END, df.describe(include='all').to_string())
    text.insert(tk.END, "\n\n")
    text.insert(tk.END, "[Data Shape]\n")
    text.insert(tk.END, str(df.shape))
    
    #읽기 전용
    text.configure(state='disabled')
    
    Exit_button = tk.Button(info_window, text='Exit', command=info_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)

#Missing Data processing
def Show_Missing():
    missing_window = tk.Toplevel(root)
    missing_window.title("Missing_Data")
    
    text = tk.scrolledtext.ScrolledText(missing_window)
    text.pack()
    
    # 결측치 확인
    text.insert(tk.END, "Data Missing Value:\n\n")
    text.insert(tk.END, df.isnull().sum().to_string())
    
    #읽기 전용
    text.configure(state='disabled')

    Missing_Replace_button = tk.Button(missing_window, text='Replace', command=Missing_Replace)
    Missing_Replace_button.pack(side="top", anchor="nw", pady=10, padx=10)
    
    Exit_button = tk.Button(missing_window, text='Exit', command=missing_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정

def Missing_Replace():
    def Missing_Variable():
        col = entry_variable.get()
        if col in df.columns:
            missing_label.config(text=f"{col} 변수가 입력되었습니다.\n")
        else:
            missing_label.config(text="잘못된 변수명입니다. 다시 입력해주세요.\n")
        return col
    
    def Missing_Method(methods):
        col_name = Missing_Variable()
        if methods == 'Drop':
            df.dropna(subset=[col_name], inplace = True)
            df.reset_index(drop=True, inplace = True)
        
        if methods == 'Mean':
            df[col_name].fillna(df[col_name].mean(), inplace = True)
        
        if methods == 'Median':
            df[col_name].fillna(df[col_name].median(), inplace = True)
        
        if methods == 'Mode':
            df[col_name].fillna(df[col_name].mode()[0], inplace = True)
        
        if methods == 'Ffill':
            df[col_name].fillna(method='ffill', inplace = True)
        
        if methods == 'Bfill':
            df[col_name].fillna(method='bfill', inplace = True)
        
        label = tk.Label(replace_window,text=f"변수: {col_name}\n결측치({methods}) 처리 완료.")
        label.pack()
    
    def Values():
        def Value_Input():
            value = entry.get()
            if value.replace('.','').replace('-','').isdigit():
                value = float(value)
            if (value == 0) or value :
                value_label.config(text=f"{value}값이 입력되었습니다.\n")
                col_name = Missing_Variable()
                df[col_name].fillna(value, inplace = True)
            return value
        
        values_window = tk.Toplevel(replace_window)
        values_window.title("Enter_Value")
        
        values_window.geometry("400x200+100+100")
        values_window.resizable(True, True) # 창 크기 조절 여부
        
        label = tk.Label(values_window,text='[Option]\n결측치를 대체할 값을 적어주세요.')
        label.pack()
        
        entry = tk.Entry(values_window)
        entry.pack()
        button = tk.Button(values_window, width=15, text="Enter", overrelief="solid", command=Value_Input)
        button.pack()
        
        Exit_button = tk.Button(values_window, text='Exit', command=values_window.destroy)
        Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정
        value_label = tk.Label(replace_window, text="")
        value_label.pack()
    
    #결측치 대체 창 생성
    replace_window = tk.Toplevel(root)
    replace_window.title("Replace_Data")
    
    replace_window.geometry("400x400+100+100")
    replace_window.resizable(True, True) # 창 크기 조절 여부
    
    #결측치가 있는 열 출력
    col_na=[]
    for isna_i, isna_sum in enumerate(df.isna().sum().tolist()):
        if isna_sum != 0:
            col_na.append(df.columns[isna_i])
    #결측치 대체 열 선택
    label = tk.Label(replace_window,text=f"변수: {col_na}\n결측치 제거할 변수에 해당하는 column를 적어주세요. ")
    label.pack()
    entry_variable = tk.Entry(replace_window)
    entry_variable.pack()
    button = tk.Button(replace_window,width=15, text="Enter", overrelief="solid", command=Missing_Variable)
    button.pack()
    
    missing_label = tk.Label(replace_window, text="")
    missing_label.pack()
    
    #결측치 대체 코드
    label = tk.Label(replace_window, text='\n결측치 대체 방법을 선택하세요.')
    label.pack()
    
    button1 = tk.Button(replace_window, width=15, text='Drop', overrelief="solid", command=lambda: Missing_Method('Drop'))
    button1.pack()
    button2 = tk.Button(replace_window, width=15, text='Mean', overrelief="solid", command=lambda: Missing_Method('Mean'))
    button2.pack()
    button3 = tk.Button(replace_window, width=15, text='Median', overrelief="solid", command=lambda: Missing_Method('Median'))
    button3.pack()
    button4 = tk.Button(replace_window, width=15, text='Mode', overrelief="solid", command=lambda: Missing_Method('Mode'))
    button4.pack()
    button5 = tk.Button(replace_window, width=15, text='F-fill', overrelief="solid", command=lambda: Missing_Method('Ffill'))
    button5.pack()
    button6 = tk.Button(replace_window, width=15, text='B-fill', overrelief="solid", command=lambda: Missing_Method('Bfill'))
    button6.pack()
    button7 = tk.Button(replace_window, width=15, text='Values', overrelief="solid", command=Values)
    button7.pack()
    
    Exit_button = tk.Button(replace_window, text='Exit', command=replace_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정

#Hold_Out
def Show_Hold_Out():
    def Test_Ratio():
        from sklearn.model_selection import train_test_split
        Ratio = float(entry.get())
        Ratio /= 100
        
        x = df[x_col]
        y = df[y_col]

        global x_train, x_test, y_train, y_test
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=Ratio, random_state = 0)

        #종속변수가 2차원이 아닐 경우, 2차원으로 만드는 코드
        if type(x_train) == np.ndarray: 
            x_train = x_train.reshape(-1,1)
            x_test = x_test.reshape(-1,1)
        #종속변수를 1차원으로 만드는 코드
        if type(y_train) == pd.DataFrame: 
            y_train = np.ravel(y_train)
            y_test = np.ravel(y_test)
        
        hold_out_label.config(text=f'''\n{int(Ratio*100)}가 입력되었습니다.
                              \nx_train.shape: {x_train.shape}
                              \nx_test.shape: {x_test.shape}
                              \ny_train.shape: {y_train.shape}
                              \ny_test.shape: {y_test.shape}\n
                              ''')

    hold_window = tk.Toplevel(root)
    hold_window.title("Data_Separate")
    
    hold_window.geometry("400x300+100+100")
    hold_window.resizable(True, True) # 창 크기 조절 여부
    
    label = tk.Label(hold_window, text='test데이터 비율을 입력하세요.(0 ~ 100)')
    label.pack()
    entry = tk.Entry(hold_window)
    entry.pack()
    button = tk.Button(hold_window,width=15, text="Enter", overrelief="solid", command=Test_Ratio)
    button.pack()
    hold_out_label = tk.Label(hold_window, text="")
    hold_out_label.pack()
    
    Exit_button = tk.Button(hold_window, text='Exit', command=hold_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정

#Data Scaling
def Show_Scaling():
    def Scaling(scaler):
        global x_train, x_test, y_train, y_test
        
        #스케일러 모듈 설정
        if scaler == "Standard":
            from sklearn.preprocessing import StandardScaler
            scale_model = StandardScaler()
            
        if scaler == "MinMax":
            from sklearn.preprocessing import MinMaxScaler
            scale_model = MinMaxScaler()
            
        if scaler == "Robust":
            from sklearn.preprocessing import RobustScaler
            scale_model = RobustScaler()
            
        if scaler == "Normalizer": # Nan값이 있으면 실행 안됨
            from sklearn.preprocessing import Normalizer
            scale_model = Normalizer()
   
        #로그 변환
        if scaler == "Logit":
            x_train = np.log1p(x_train)
            x_test = np.log1p(x_test)
    
        #모듈 스케일링
        if scaler != "Logit":
            if type(x_train) == np.ndarray:
                x_train = scale_model.fit_transform(x_train)
                x_test = scale_model.transform(x_test)
                df[x_col] = scale_model.fit_transform(df[x_col])
            else:
                x_train = pd.DataFrame(scale_model.fit_transform(x_train), columns = x_col)
                x_test = pd.DataFrame(scale_model.transform(x_test), columns = x_col)
                df[x_col] = pd.DataFrame(scale_model.transform(df[x_col]), columns = x_col)


        result_label.config(text="{0} Scaler적용\n".format(scaler))

    scaling_window = tk.Toplevel(root)
    scaling_window.title("스케일링 방법 선택")
        
    scaling_window.geometry("400x300+100+100")
    scaling_window.resizable(True, True) # 창 크기 조절 여부
    
    label = tk.Label(scaling_window, text='스케일링 방법을 선택하세요.')
    label.pack()
    
    button1 = tk.Button(scaling_window, width=15, text="Standard_Scale", overrelief="solid", command=lambda: Scaling('Standard'))
    button1.pack()
    button2 = tk.Button(scaling_window, width=15, text="MinMax_Scale", overrelief="solid", command=lambda: Scaling('MinMax'))
    button2.pack()
    button3 = tk.Button(scaling_window, width=15, text="Robust_Scale", overrelief="solid", command=lambda: Scaling('Robust'))
    button3.pack()
    button4 = tk.Button(scaling_window, width=15, text="Normalizer_Scale", overrelief="solid", command=lambda: Scaling('Normalizer'))
    button4.pack()
    button5 = tk.Button(scaling_window, width=15, text="Logit_Scale", overrelief="solid", command=lambda: Scaling('Logit'))
    button5.pack()
    
    result_label = tk.Label(scaling_window, text="")
    result_label.pack()
    Exit_button = tk.Button(scaling_window, text='Exit', command=scaling_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)

#EDA
def Show_EDA():
    #EDA
    def EDA_Group(select):
        EDA_n = ceil(np.sqrt(len(df.columns)))
        di = getcwd()
        path_png = str(di)+"\\"+select+".png"

        if select == 'Histogram':
            fig, ax_lst = plt.subplots(EDA_n,EDA_n,figsize=(5*EDA_n,3*EDA_n))
            fig.suptitle('Histogram')
            
            EDA_loop_end = False
            for i in range(EDA_n):
                for j in range(EDA_n):
                    if i*EDA_n+j > len(x_col):
                        EDA_loop_end = True
                        break
                    ax_lst[i][j].hist(df[df.columns[i*EDA_n+j]])
                    ax_lst[i][j].set_title(df.columns[i*EDA_n+j])
                if EDA_loop_end:
                    break
            plt.tight_layout()
            plt.show()
            #저장
            fig.savefig(path_png)
        
        if select == 'Boxplot':
            fig, ax_lst = plt.subplots(EDA_n,EDA_n,figsize=(5*EDA_n,3*EDA_n))
            fig.suptitle('Boxplot')
            
            EDA_loop_end = False
            for i in range(EDA_n):
                for j in range(EDA_n):
                    if i*EDA_n+j > len(x_col):
                        EDA_loop_end = True
                        break
                    ax_lst[i][j].boxplot(df[df.columns[i*EDA_n+j]])
                    ax_lst[i][j].set_title(df.columns[i*EDA_n+j])
                if EDA_loop_end:
                    break
            plt.tight_layout()
            plt.show()
            #저장
            fig.savefig(path_png)

        if select == 'Scatter':
            EDA_n = ceil(np.sqrt(len(df[x_col].columns)))
            fig, ax_lst = plt.subplots(EDA_n,EDA_n,figsize=(5*EDA_n,3*EDA_n))
            fig.suptitle('Scatter')
            
            EDA_loop_end = False
            for i in range(EDA_n):
                for j in range(EDA_n):
                    if i*EDA_n+j >= len(x_col):
                        EDA_loop_end = True
                        break
                    ax_lst[i][j].plot(df[df[x_col].columns[i*EDA_n+j]], df[y_col], '+')
                    ax_lst[i][j].set_title(df[x_col].columns[i*EDA_n+j])
                if EDA_loop_end:
                    break
            plt.tight_layout()
            plt.show()
            #저장
            fig.savefig(path_png)
            
        if select == 'Heatmap':
            fig, ax = plt.subplots(1, 1, figsize=(5*EDA_n,3*EDA_n))
            fig.suptitle('Heatmap')
            ax.xaxis.tick_top()
            sns.heatmap(df.corr(), annot=True, cmap = 'OrRd')
            plt.tight_layout()
            plt.show() 
            #저장
            fig.savefig(path_png)

        if select == 'Pairplot': #종속변수가 범주형일 경우.
            sns_pair = sns.pairplot(df, hue=y_col, vars=x_col)
            #저장
            sns_pair.savefig(path_png)
        
        plot_window = tk.Toplevel(eda_window)
        plot_window.title(select)
        
        src = cv2.imread(path_png)
        src = cv2.resize(src, (640, 400))
        
        img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=img)
        
        label = tk.Label(plot_window, image=imgtk)
        label.pack(side="top")
        
        Exit_button = tk.Button(plot_window, text='Exit', command=plot_window.destroy)
        Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정
        
    eda_window = tk.Toplevel(root)
    eda_window.title("데이터 EDA")
    
    eda_window.geometry("400x200+100+100")
    eda_window.resizable(True, True) # 창 크기 조절 여부
    
    label = tk.Label(eda_window, text='시각화 방법을 선택하세요.')
    label.pack()
    
    button1 = tk.Button(eda_window, width=15, text="히스토그램", overrelief="solid", command=lambda: EDA_Group('Histogram'))
    button1.pack()
    button2 = tk.Button(eda_window, width=15, text="박스플롯", overrelief="solid", command=lambda: EDA_Group('Boxplot'))
    button2.pack()
    button3 = tk.Button(eda_window, width=15, text="산점도", overrelief="solid", command=lambda: EDA_Group('Scatter'))
    button3.pack()
    button4 = tk.Button(eda_window, width=15, text="히트맵", overrelief="solid", command=lambda: EDA_Group('Heatmap'))
    button4.pack()
    button5 = tk.Button(eda_window, width=15, text="Pairplot", overrelief="solid", command=lambda: EDA_Group('Pairplot'))
    button5.pack()
    
    Exit_button = tk.Button(eda_window, text='Exit', command=eda_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)

#Encoding 
def Show_Encoding():
    def Encoding(name):
        global df, x_col
        
        if df[y_col].dtypes == object or df[y_col].dtypes == bool:
            label_dict = dict(zip(sorted(list(df[y_col].unique())), list(range(len(df[y_col].unique())))))
            df[y_col] = df[y_col].map(label_dict)
            text.insert(tk.END, f'종속변수 {y_col}\t\t=> {label_dict}\n')

        if name=='Label':
            text.delete('1.0', 'end')
            for i, col in enumerate(df.columns):
                if df.dtypes[i] == object or df.dtypes[i] == bool:
                    label_dict = dict(zip(sorted(list(df[col].unique())), list(range(len(df[col].unique())))))
                    df[col] = df[col].map(label_dict)
                    text.insert(tk.END, f'{col}\t\t=> {label_dict}\n')

        if name=='Onehot':
            text.delete('1.0', 'end')
            encoded_df = pd.get_dummies(df, columns=df.select_dtypes(include=['object', 'bool']).columns)
            for col in encoded_df.columns:
                if col in df.columns:
                    continue
                text.insert(tk.END, f'{col}\n')
        
            df = pd.get_dummies(df, columns=df.select_dtypes(include=['object', 'bool']).columns)
        x_col = list(df.columns)[1:]
        
    encoding_window = tk.Toplevel(root)
    encoding_window.title('Encoding')
    
    text = tk.scrolledtext.ScrolledText(encoding_window)
    text.config(width = 180, height = 48)
    text.pack(side='bottom')
    
    button = tk.Button(encoding_window, text="Label_Encoding", command=lambda: Encoding('Label'))
    button.pack(side="top", anchor="nw", pady=10, padx=10)
    button1 = tk.Button(encoding_window, text="Onehot_Encoding", command=lambda: Encoding('Onehot'))
    button1.pack(side="top", anchor="nw", pady=10, padx=10)
    
#Modeling
def Show_Modeling():
    def Modeling(select):
        model_window = tk.Toplevel(modeling_window)
        model_window.title(select)
                 
        text = tk.scrolledtext.ScrolledText(model_window)
        text.pack()
        #선형회귀분석
        if select == 'Linear_Regression':
            mse, rmse, mae, r2, expression_list = Linear_Regression()

            def Linear_Regression_Plot():
                EDA_n = ceil(np.sqrt(len(df[x_col].columns)))
                di = getcwd()
                path_png = str(di)+"\\"+select+".png"
                # 선형회귀 시각화
                fig, ax_lst = plt.subplots(EDA_n,EDA_n,figsize=(5*EDA_n,3*EDA_n))
                fig.suptitle(select)
            
                EDA_loop_end = False
                for i in range(EDA_n):
                    for j in range(EDA_n):
                        if i*EDA_n+j >= len(x_col):
                            EDA_loop_end = True
                            break
                    # 회귀 그래프 그리기
                        sns.regplot(x=x_col[i * EDA_n + j], y=y_col, data=df, ax=ax_lst[i][j],fit_reg=False)
                        sns.regplot(x=x_col[i * EDA_n + j], y=y_col, data=df_pred, line_kws={'color': 'black'}, ax=ax_lst[i][j],scatter=False)
                        ax_lst[i][j].set_title(f'[{x_col[i * EDA_n + j]}]')
                    if EDA_loop_end:
                        break
                plt.tight_layout()
                plt.show()
                #저장
                fig.savefig(path_png)
                
                Linear_window = tk.Toplevel(model_window)
                Linear_window.title('Linear Regression Plot')
                src = cv2.imread(path_png)
                src = cv2.resize(src, (640, 400))
                
                img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                label = tk.Label(Linear_window, image=imgtk)
                label.pack(side="top")
                
            #분석 결과 출력
            text.insert(tk.END, '[선형회귀식]\nY = {0:.5f}'.format(expression_list[0]))
            for i, b_i in enumerate(expression_list[1:]):
                text.insert(tk.END, ' + {0:.5f}*[{1}]'.format(b_i, x_col[i]))
            
            text.insert(tk.END, '\n\n[모델평가지표]')
            text.insert(tk.END, '\nMean Squared Error(MSE) : {0:.5f}'.format(mse))
            text.insert(tk.END, '\nRoot Mean Squared Error(RMSE) : {0:.5f}'.format(rmse))
            text.insert(tk.END, '\nMean Absolute Error(MAE) : {0:.5f}'.format(mae))
            text.insert(tk.END, '\nR^2 Score : {0:.5f}'.format(r2))
            
            button = tk.Button(model_window, width=20, text="Linear Regression Plot", overrelief="solid", command=Linear_Regression_Plot)
            button.pack()

        #로지스틱 회귀분석
        if select == 'Logistic_Regression':
            accuracy, precision, recall, Classification_report, Odds_Ratio, Confusion_matrix, expression_list = Logistic_Regression()
            
            def Logistic_Regression_Plot():
                global df_pred
                EDA_n = ceil(np.sqrt(len(df[x_col].columns)))
                di = getcwd()
                path_png = str(di)+"\\"+select+".png"
                # 선형회귀 시각화
                fig, ax_lst = plt.subplots(EDA_n,EDA_n,figsize=(5*EDA_n,3*EDA_n))
                fig.suptitle(select)
                
                EDA_loop_end = False
                for i in range(EDA_n):
                    for j in range(EDA_n):
                        if i*EDA_n+j >= len(x_col):
                            EDA_loop_end = True
                            break
                        # 회귀 그래프 그리기
                        df_pred = df_pred.astype({y_col:'int'})
                        sns.regplot(x=x_col[i * EDA_n + j], y=y_col, data=df, ax=ax_lst[i][j],fit_reg=False,logistic=True)
                        sns.regplot(x=x_col[i * EDA_n + j], y=y_col, data=df_pred, line_kws={'color': 'black'}, ax=ax_lst[i][j],scatter=False,logistic=True)
                        ax_lst[i][j].set_title(f'[{x_col[i * EDA_n + j]}]')
                    
                    if EDA_loop_end:
                        break
                plt.tight_layout()
                plt.show()
                #저장
                fig.savefig(path_png)
                
                Logistic_window = tk.Toplevel(model_window)
                Logistic_window.title('Logistic Regression Plot')
                src = cv2.imread(path_png)
                src = cv2.resize(src, (640, 400))
                
                img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                label = tk.Label(Logistic_window, image=imgtk)
                label.pack(side="top")
                
            #분석 결과 출력
            text.insert(tk.END, '[로지스틱 회귀식]\nln(P/(1-P)) = {0:.5f}'.format(expression_list[0]))
            for i, b_i in enumerate(expression_list[1:]):
                text.insert(tk.END, ' + {0:.5f}*[{1}]'.format(b_i, x_col[i]))
            
            text.insert(tk.END, '\n\n[모델평가지표]')
            text.insert(tk.END, '\nAccuracy : {0:.5f}'.format(accuracy))
            text.insert(tk.END, '\nPrecision : {0:.5f}'.format(precision))
            text.insert(tk.END, '\nRecall : {0:.5f}'.format(recall))
            text.insert(tk.END, '\n\nClassification_report : \n{0}'.format(Classification_report))
            text.insert(tk.END, '\nConfusion_matrix : \n{0}'.format(Confusion_matrix))
            text.insert(tk.END, '\nOdds_Ratio : {0}'.format(Odds_Ratio))
            button = tk.Button(model_window, width=20, text="Logistic Regression Plot", overrelief="solid", command=Logistic_Regression_Plot)
            button.pack()

        #Decision Tree
        if select == 'Decision_Tree':
            accuracy, precision, recall, Classification_report, entropy, model = Decision_Tree()

            def Decision_Tree_Plot():
                di = getcwd()
                path_png = str(di)+"\\"+select+".png"
                
                features = x_col
                classes = list(map(str, list(df[y_col].unique())))
                
                fig = plt.figure(figsize=(10, 8))
                plot_tree(model,
                          feature_names=features,
                          class_names=classes,
                          rounded=True, # Rounded node edges
                          filled=True, # Adds color according to class
                          proportion=True); # Displays the proportions of class samples instead of the whole number of samples
                
                plt.tight_layout()
                plt.show()
                #저장
                fig.savefig(path_png)
                
                Decision_window = tk.Toplevel(model_window)
                Decision_window.title('Decision Tree Plot')
                src = cv2.imread(path_png)
                src = cv2.resize(src, (640, 400))
                
                img = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
                
                img = Image.fromarray(img)
                imgtk = ImageTk.PhotoImage(image=img)
                
                label = tk.Label(Decision_window, image=imgtk)
                label.pack(side="top")
                
            #Entropy
            text.insert(tk.END, '\nEntropy : \n{0}'.format(entropy))
            
            #분석 결과 출력
            text.insert(tk.END, '\n\n[모델평가지표]')
            text.insert(tk.END, '\nAccuracy : {0:.5f}'.format(accuracy))
            text.insert(tk.END, '\nPrecision : {0:.5f}'.format(precision))
            text.insert(tk.END, '\nRecall : {0:.5f}'.format(recall))
            text.insert(tk.END, '\nClassification_report : \n{0}'.format(Classification_report))
            
            button = tk.Button(model_window, width=20, text="Decision Tree Plot", overrelief="solid", command=Decision_Tree_Plot)
            button.pack()
        
        #읽기 전용
        text.configure(state='disabled')
        
        Exit_button = tk.Button(model_window, text='Exit', command=model_window.destroy)
        Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정

    # 새로운 창 만들기
    modeling_window = tk.Toplevel(root)
    modeling_window.title("분석 방법 선택")
    
    modeling_window.geometry("400x300+100+100")
    modeling_window.resizable(True, True) # 창 크기 조절 여부
    
    label = tk.Label(modeling_window, text='분석방법을 선택하세요.')
    label.pack()
    
    button1 = tk.Button(modeling_window, width=15, text="Linear Regression", overrelief="solid", command=lambda: Modeling('Linear_Regression'))
    button1.pack()
    button2 = tk.Button(modeling_window, width=15, text="Logistic Regression", overrelief="solid", command=lambda: Modeling('Logistic_Regression'))
    button2.pack()
    button3 = tk.Button(modeling_window, width=15, text="Decision Tree", overrelief="solid", command=lambda: Modeling('Decision_Tree'))
    button3.pack()

    Exit_button = tk.Button(modeling_window, text='Exit', command=modeling_window.destroy)
    Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정

#__Main__
root = tk.Tk()
root.title("Data Analysis")
root.geometry("600x500+100+100")
root.resizable(True, True) # 창 크기 조절 여부

#데이터 보기 버튼
View_button = tk.Button(root, text='Data Preview', command=Show_Data)
View_button.pack(side="top", anchor="nw", pady=10, padx=10)

#데이터 정보 버튼
Info_button = tk.Button(root, text='Data Summary', command=Show_Information)
Info_button.pack(side="top", anchor="nw", pady=10, padx=10)

# 데이터 결측치 버튼
Missing_button = tk.Button(root, text='Missing Data', command=Show_Missing)
Missing_button.pack(side="top", anchor="nw", pady=10, padx=10)

# 데이터 인코딩 버튼
Encoding_button = tk.Button(root, text='Encoding Data', command=Show_Encoding)
Encoding_button.pack(side="top", anchor="nw", pady=10, padx=10)

# 데이터 EDA 버튼
Eda_button = tk.Button(root, text='Data EDA', command=Show_EDA)
Eda_button.pack(side="top", anchor="nw", pady=10, padx=10)


# 데이터 Hold Out 버튼
Hold_Out_button = tk.Button(root, text='Data Separate', command=Show_Hold_Out)
Hold_Out_button.pack(side="top", anchor="nw", pady=10, padx=10)

# 데이터 스케일링 버튼     
Scale_button = tk.Button(root, text='Data Scaling', command=Show_Scaling)
Scale_button.pack(side="top", anchor="nw", pady=10, padx=10)

# 분석 버튼    
Analysis_button = tk.Button(root, text='Data Analysis', command=Show_Modeling)
Analysis_button.pack(side="top", anchor="nw", pady=10, padx=10)

# 종료 버튼
Exit_button = tk.Button(root, text='Exit', command=root.destroy)
Exit_button.pack(side="bottom", anchor="sw", pady=10, padx=10)  # 윈도우 창의 크기가 변해도 위치 고정
root.mainloop()