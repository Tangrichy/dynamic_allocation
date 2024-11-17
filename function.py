import pandas as pd
import numpy as np
import exchange_calendars as xcals
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
# Tree Visualisation
from sklearn.tree import export_graphviz
from IPython.display import Image
import graphviz

# match trading time
xshg = xcals.get_calendar("XSHG")


## label the forward n perfomance
def calculate_rolling_return(data, rolling_days, columns, thres,handle_missing='drop'):

    # Ensure 'Date' is in datetime format
    data['Date'] = pd.to_datetime(data['Date'])

    # Handle missing data based on the selected method
    if handle_missing == 'drop':
        data = data.dropna(subset=columns)
    elif handle_missing == 'ffill':
        data[columns] = data[columns].fillna(method='ffill')
    elif handle_missing == 'bfill':
        data[columns] = data[columns].fillna(method='bfill')
    elif handle_missing == 'interpolate':
        data[columns] = data[columns].interpolate()

    # Sort the data by date
    data = data.sort_values(by='Date')

    for column in columns:
        # Calculate rolling returns for each specified column
        return_col = f"{column}_Rolling_Return"
        mark_col = f"{column}_Return_Mark"

        data[return_col] = data[column].pct_change(periods=rolling_days)
        data[mark_col] = data[return_col].apply(lambda x: 1 if x > thres else (0 if x < thres else None))
    
    return data



## demo data read
equity_df = pd.read_excel("Data/dynamic demo.xlsx", sheet_name = "Price")
bond_df = pd.read_excel("Data/dynamic demo.xlsx", sheet_name = "Bond")
df = equity_df.merge(bond_df, on = "Date")

macro_df = pd.read_excel("Data/dynamic demo.xlsx")
select = ["Date",'炼焦煤库存:六港口合计','OPEC:一揽子原油价格','Myspic综合钢价指数','30大中城市:商品房成交面积:一线城市', '柯桥纺织:价格指数:总类','义乌中国小商品指数:总价格指数', '中国沿海散货运价指数(CCBFI)', '螺纹钢:主要钢厂开工率:全国', '产能利用率:电炉:全国', 'PTA产业链负荷率:PTA工厂', '食用农产品价格指数', '水泥价格指数:全国']
select = ["Date",'OPEC:一揽子原油价格','30大中城市:商品房成交面积:一线城市', 'PTA产业链负荷率:PTA工厂', '水泥价格指数:全国']

macro_df = macro_df[select]
macro_df = macro_df.dropna()

## macro data match trading time
## weekly data or monthly

for i in range(macro_df.shape[0]):
    temp_date = macro_df.iloc[i, 0]
    temp_date = str(temp_date)[:10]

    if xshg.is_session(temp_date):
        pass
    else:
        macro_df.iloc[i, 0] = xshg.next_close(temp_date)
        print("Next trading")




# forward 12 week performance
df_mark = calculate_rolling_return(data = df, rolling_days = 12, columns = ["ZZ1000", "AU"], thres = 0, handle_missing='drop')
df_mark = df_mark[["Date", "ZZ1000", "AU", "ZZ1000_Return_Mark", "AU_Return_Mark"]]

# pre-process
df = df_mark.merge(macro_df, on="Date")
df = df.drop_duplicates()

df.iloc[:,5:] = df.iloc[:,5:].shift(12)
df = df.dropna()

# label y
df["Label"] = df["ZZ1000_Return_Mark"].astype(int).astype(str) + "," + df["AU_Return_Mark"].astype(int).astype(str)

# random forest
X = df.iloc[:,5:9]
y = df["Label"]

model = RandomForestClassifier(n_estimators=100,
                               max_features = 'sqrt',
                               n_jobs=-1, verbose = 1)

# Fit on training data
model.fit(X, y)

train_rf_predictions = model.predict(X)
train_rf_probs = model.predict_proba(X)[:, 1]
