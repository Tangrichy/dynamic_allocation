import pandas as pd
import numpy as np


def calculate_rolling_return(data, rolling_days, columns, handle_missing='drop', thres):

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

## demo
equity_df = pd.read_excel("Data/dynamic demo.xlsx", sheet_name = "Price")
bond_df = pd.read_excel("Data/dynamic demo.xlsx", sheet_name = "Bond")
df = equity_df.merge(bond_df, on = "Date")

macro_df = pd.read_excel("Data/dynamic demo.xlsx")
select = ["Date",'炼焦煤库存:六港口合计','OPEC:一揽子原油价格','Myspic综合钢价指数','30大中城市:商品房成交面积:一线城市', '柯桥纺织:价格指数:总类','义乌中国小商品指数:总价格指数', '中国沿海散货运价指数(CCBFI)', '螺纹钢:主要钢厂开工率:全国', '产能利用率:电炉:全国', 'PTA产业链负荷率:PTA工厂', '食用农产品价格指数', '水泥价格指数:全国']
macro_df[select]