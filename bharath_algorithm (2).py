from tkinter import *
from tkinter import Label
import tkinter 
from tkinter import Canvas
import cv2
import PIL.Image, PIL.ImageTk
from tkinter import filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from numpy import reshape
from pandas import DataFrame
from pandastable import Table
from tkinter import ttk
window = Tk()
scrollbar = Scrollbar(window)
scrollbar.pack( side = RIGHT, fill = Y ) 
mylist = Listbox(window, yscrollcommand = scrollbar.set)
window.title("BHARATH SOLAR POWER ESTIMATOR")
x2 = "WITH OUT NIGHT DATA"
Ax1 = "THE ACCURACY OF PREDICTED VALUES: "
Ax2 = "THE ACCURACY OF PREDICTED VALUES of ONLY DAY: "
x1 = "WITH NIGHT DATA"
lb1 = Label(window, text="TESTING THE ALGORITHM", font=("Arial Bold", 35), fg='white', underline=0, bg='black')
lb2 = Label(window, text="USING THE ALGORITHM", font=("Arial Bold", 35), fg='white', underline=0, bg='black')
lbl=Label(window, text=Ax1)
lbl1=Label(window, text=Ax2)
lbl2=Label(window, text="ACCURACY OF THE TEST FILE DATA: ")
def getExcel ():
    global weth
    
    import_file_path = filedialog.askopenfilename()
    weth = pd.read_excel (import_file_path)
    btn1.configure(text="FILE IMPORTED", fg='green')
def getExcel2 ():
    global test
    
    import_file_path = filedialog.askopenfilename()
    test = pd.read_excel (import_file_path)
    btn7.configure(text="FILE IMPORTED")
def getExcel3 ():
    global test1
    
    import_file_path = filedialog.askopenfilename()
    test1 = pd.read_excel (import_file_path)
    btn10.configure(text="FILE IMPORTED")
def nightdata():
    df = DataFrame(weth, columns=weth.columns)
    X=df.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
    y=weth['solar_power_gen']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)    
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    plt.figure(figsize=(20,12))
    plt.scatter(y_test, prediction)
    for i in range(len(prediction)):
        if(prediction[i]<1):
            prediction[i]=0
    from sklearn import metrics
    mean_sq_er = metrics.mean_squared_error(y_test, prediction)
    root_mean_sq = np.sqrt(mean_sq_er)
    df = pd.DataFrame({'Actual': y_test, 'prediction': prediction, })
    Ax1=model.score(X_test, y_test)
    r = str(round((model.score(X_test, y_test))*100, 1))
    lbl.configure(text="Accuracy of predicted values: " + r + "%")

    
class nightt(Frame):
        """Basic test frame for the table"""
        def __init__(self, parent=None):
            self.parent = parent
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('window')
            f = Frame(self.main)
            f.pack(fill=BOTH,expand=1)
            df = DataFrame(weth, columns=weth.columns)
            X_train=df.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
            y_train=weth['solar_power_gen']
            X_test=test   
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            for i in range(len(prediction)):
                if(prediction[i]<1):
                    prediction[i]=0
            df1 = pd.DataFrame({'Predicted': prediction})
            self.table = pt = Table(f, dataframe=df1,
                                    showtoolbar=True, showstatusbar=True)
            pt.show()
            return
class NNt(Frame):
        """Basic test frame for the table"""
        def __init__(self, parent=None):
            self.parent = parent
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('window')
            f = Frame(self.main)
            f.pack(fill=BOTH,expand=1)
            df = DataFrame(weth, columns=weth.columns)
            we = df.loc[weth.solar_power_gen > 0]
            X_train=we.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
            y_train=we['solar_power_gen']
            X_test=test
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            for i in range(len(prediction)):
                if(prediction[i]<1):
                    prediction[i]=0
            df1 = pd.DataFrame({'Predicted': prediction})
            

            self.table = pt = Table(f, dataframe=df1,
                                    showtoolbar=True, showstatusbar=True)
            pt.show()
            return

class NNt1(Frame):
        """Basic test frame for the table"""
        def __init__(self, parent=None):
            self.parent = parent
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('window')
            f = Frame(self.main)
            f.pack(fill=BOTH,expand=1)
            df = DataFrame(weth, columns=weth.columns)
            we = df.loc[weth.solar_power_gen > 0]
            X_train=we.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
            y_train=we['solar_power_gen']
            X_test=test
            y_test=test1
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            for i in range(len(prediction)):
                if(prediction[i]<200):
                    prediction[i]=0
            r = str(round((model.score(X_test, y_test))*100, 1))
            lbl2.configure(text="ACCURACY OF THE TEST FILE DATA: " + r + "%")
            df1 = pd.DataFrame({'ACTUAL': test1['solar_power_gen'], 'Predicted': prediction})
            self.table = pt = Table(f, dataframe=df1,
                                    showtoolbar=True, showstatusbar=True)
            pt.show()
            return
   
class TestApp(Frame):
        """Basic test frame for the table"""
        def __init__(self, parent=None):

            time.sleep(1)
            window.update_idletasks()
            self.parent = parent
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('window')
            f = Frame(self.main)
            f.pack(fill=BOTH,expand=1)
            df = DataFrame(weth, columns=weth.columns)
            X=df.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
            y=weth['solar_power_gen']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y)    
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            plt.figure(figsize=(20,12))
            plt.scatter(y_test, prediction)
            for i in range(len(prediction)):
                if(prediction[i]<1):
                    prediction[i]=0
            from sklearn import metrics
            mean_sq_er = metrics.mean_squared_error(y_test, prediction)
            root_mean_sq = np.sqrt(mean_sq_er)
            df1 = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
            self.table = pt = Table(f, dataframe=df1,
                                    showtoolbar=True, showstatusbar=True)
            pt.show()
            return
class TestApp1(Frame):
        """Basic test frame for the table"""
        def __init__(self, parent=None):
            self.parent = parent
            Frame.__init__(self)
            self.main = self.master
            self.main.geometry('600x400+200+100')
            self.main.title('window')
            f = Frame(self.main)
            f.pack(fill=BOTH,expand=1)
            df = DataFrame(weth, columns=weth.columns)
            we = df.loc[weth.solar_power_gen > 0]
            X=we.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
            y=we['solar_power_gen']
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y)    
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
            model.fit(X_train, y_train)
            prediction = model.predict(X_test)
            for i in range(len(prediction)):
                if(prediction[i]<1):
                    prediction[i]=0
            from sklearn import metrics
            mean_sq_er = metrics.mean_squared_error(y_test, prediction)
            root_mean_sq = np.sqrt(mean_sq_er)
            df1 = pd.DataFrame({'Actual': y_test, 'Predicted': prediction})
            self.table = pt = Table(f, dataframe=df1,
                                    showtoolbar=True, showstatusbar=True)
            pt.show()
            return
def withoutnight():
    
    df = DataFrame(weth, columns=weth.columns)
    we = df.loc[weth.solar_power_gen > 0]
   
    X=we.drop(['solar_power_gen', 'Solar_irradiation'], axis=1)
    y=we['solar_power_gen']
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators = 1000,min_samples_split=2,min_samples_leaf= 1,max_features='sqrt',max_depth=20,bootstrap= True)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)
    for i in range(len(prediction)):
        if(prediction[i]<1):
            prediction[i]=0
    df1 = pd.DataFrame({'Actual': y_test, 'prediction': prediction})
    r = str(round((model.score(X_test, y_test))*100, 1))
    lbl1.configure(text="Accuracy of predicted values: " + r + "%")
def closewin():
    window.destroy()
    
btn1=Button(window, text="IMPORT TRAINING DATA",fg="red", command=getExcel, font=("Arial Bold", 25))
btn=Button(window, text="OBTAIN ACCURACY OF FULLDAY", command=nightdata, font=("Arial Bold", 20), fg='maroon2')
btn3=Button(window, text="OBTAIN ACCURACY OF ONLYDAY", command=withoutnight, font=("Arial Bold", 20), fg='black')
btn4=Button(window, text='OBTAIN ORIGINAL AND PREDICTED VALUES', command=TestApp, font=("Arial Bold", 20), fg='blue')
btn6=Button(window, text='OBTAIN ORIGINAL AND PREDICTED VALUES for only day', command=TestApp1, font=('Arial Bold', 20), fg='purple3')

btn7=Button(window, text="IMPORT TESTING DATA",fg="red", command=getExcel2, font=('Arial Bold', 25))
btn10=Button(window, text="IMPORT result DATA",fg="red", command=getExcel3, font=('Arial Bold', 25))

btn11=Button(window, text="OBTAIN RESULT FOR WHOLE DAY",fg="orange", command=NNt1, font=('Arial Bold', 25))

btn8=Button(window, text='PREDICT the SOLAR POWER GENERATED', command=nightt, font=('Arial Bold', 20), fg='maroon2')
btn9=Button(window, text='PREDICT the SOLAR POWER GENERATED EXCLUDING NIGHT', command=NNt, font=('Arial Bold', 20), fg='purple3')
btn5=Button(window, text='QUIT', command=closewin, font=('Arial BOld', 15))


lb1.pack()

btn1.pack()
btn.pack()
lbl.pack()
btn3.pack()
lbl1.pack()
btn4.pack()
btn6.pack()

lb2.pack()

btn8.pack()
btn7.pack()
btn9.pack()
btn10.pack()
btn11.pack()
lbl2.pack()
btn5.pack()
progress_var = tkinter.IntVar()
pb = ttk.Progressbar(window, orient="horizontal",
                        length=200, maximum=10,
                        mode="determinate",
                        var=progress_var)
pb.pack(side="left")
pb["value"] = 0
window.mainloop()
