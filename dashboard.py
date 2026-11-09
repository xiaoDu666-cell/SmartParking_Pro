import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
from datetime import datetime

# 设置网页的标题和图标
st.set_page_config(page_title="智慧停车场数据大屏", page_icon="🅿️", layout="wide")

# 数据库路径
DB_PATH = 'parking.db'

def get_data(query):
    """从数据库查询数据并返回 DataFrame"""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# --- 页面顶部：标题与大指标 ---
st.title("🅿️ 智慧停车场实时数据监控大屏")

# 获取核心指标数据
current_cars_df = get_data("SELECT COUNT(*) as count FROM current_vehicles")
current_cars = current_cars_df['count'][0] if not current_cars_df.empty else 0

today = datetime.now().strftime('%Y-%m-%d')
today_income_df = get_data(f"SELECT SUM(fee) as total FROM parking_records WHERE exit_time LIKE '{today}%'")
today_income = today_income_df['total'][0] if not today_income_df.empty and pd.notna(today_income_df['total'][0]) else 0.0

total_records_df = get_data("SELECT COUNT(*) as count FROM parking_records")
total_records = total_records_df['count'][0] if not total_records_df.empty else 0

# 使用列布局展示指标卡片
col1, col2, col3 = st.columns(3)
col1.metric("当前在场车辆", f"{current_cars} 辆")
col2.metric("今日营收", f"¥ {today_income:.2f}")
col3.metric("历史累计服务车次", f"{total_records} 次")

st.markdown("---")

# --- 页面中下部：图表分析 ---
row1_left, row1_right = st.columns(2)

with row1_left:
    st.subheader("📊 近期停车时长分布")
    # 查最近的 50 条记录
    duration_df = get_data("SELECT duration_hours FROM parking_records ORDER BY id DESC LIMIT 50")
    if not duration_df.empty:
        # 制作直方图
        fig = px.histogram(duration_df, x="duration_hours", nbins=10,
                           labels={'duration_hours': '停车时长 (小时)'},
                           title="近期车辆停车时长直方图", color_discrete_sequence=['#4F8BF9'])
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无历史流水数据，请去摄像头端模拟进出场。")

with row1_right:
    st.subheader("📈 历史营收趋势")
    income_df = get_data("SELECT substr(exit_time, 1, 10) as date, SUM(fee) as revenue FROM parking_records GROUP BY date")
    if not income_df.empty:
        # 制作折线图
        fig2 = px.line(income_df, x='date', y='revenue', markers=True,
                       title="每日营收趋势图", labels={'date': '日期', 'revenue': '收入 (元)'})
        st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("暂无营收数据。")

st.markdown("---")

# --- 页面底部：流水明细表 ---
st.subheader("📋 停车场最新流水明细")
latest_records = get_data("SELECT plate_number as '车牌号', entry_time as '入场时间', exit_time as '出场时间', duration_hours as '停车时长(小时)', fee as '费用(元)' FROM parking_records ORDER BY id DESC LIMIT 20")

if not latest_records.empty:
    st.dataframe(latest_records, use_container_width=True, hide_index=True)
else:
    st.info("暂无流水记录。")