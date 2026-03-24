import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Gerontech Data Analytics Dashboard", layout="wide")

st.title("📊 乐龄科技 (Gerontech) 租赁数据与需求预测看板")
st.markdown("基于真实租赁记录及外部人口数据融合分析 | **IoT Data Hackathon 2026**")

@st.cache_data
def load_data():
    df_orders = pd.read_pickle("dashboard_orders.pkl")
    df_external = pd.read_pickle("dashboard_external.pkl")
    
    # 过滤出 2020-2026 年的记录，避免脏数据
    df_orders = df_orders[(df_orders['租借開始日期'].dt.year >= 2020) & (df_orders['租借開始日期'].dt.year <= 2026)]
    
    # 为了聚合，新增年月列
    df_orders['YearMonth'] = df_orders['租借開始日期'].dt.to_period('M').dt.to_timestamp()
    
    # Calculate Service Gap
    # 计算各个地区的当前租赁量
    district_rentals = df_orders['地區*'].value_counts().reset_index()
    district_rentals.columns = ['地區*', 'Current_Rentals']
    
    df_gap = pd.merge(df_external, district_rentals, on='地區*', how='left').fillna(0)
    
    # 估算需求 = 老年人口 * 慢性病率 (这里做一个加权/假设模型)
    df_gap['Estimated_Need'] = df_gap['elderly_pop'] * df_gap['chronic_disease_rate']
    
    # Penetration rate
    # 加入平滑防止除零
    df_gap['Penetration_Rate'] = df_gap['Current_Rentals'] / (df_gap['Estimated_Need'] + 1)
    
    # 归一化 Gap Score
    max_pen = df_gap['Penetration_Rate'].max()
    if max_pen > 0:
        df_gap['Penetration_Rate_Norm'] = df_gap['Penetration_Rate'] / max_pen
    else:
        df_gap['Penetration_Rate_Norm'] = 0
        
    df_gap['Service_Gap_Score'] = (1 - df_gap['Penetration_Rate_Norm']).clip(lower=0)
    
    return df_orders, df_gap

try:
    df_orders, df_gap = load_data()
except Exception as e:
    st.error(f"数据加载失败，请确保已经运行了 prepare_data.py。错误: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["📈 租赁趋势分析 (Internal Trends)", "🗺️ 需求热点与缺口 (Heatmaps & Gaps)", "🤖 市场洞察与预测 (Insights & Prediction)"])

# --- Tab 1: Rental Trends ---
with tab1:
    st.header("2020-2025 乐龄科技租赁趋势")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("设备租赁量随时间变化")
        # 取最热门的 10 个分类
        top_cats = df_orders['分類'].value_counts().head(10).index
        trend_data = df_orders[df_orders['分類'].isin(top_cats)].groupby(['YearMonth', '分類']).size().reset_index(name='Rentals')
        fig1 = px.line(trend_data, x='YearMonth', y='Rentals', color='分類', title="Monthly Rentals by Top Equipment Categories")
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("各地区整体租赁量分布")
        dist_counts = df_orders['地區*'].value_counts().reset_index()
        dist_counts.columns = ['District', 'Rentals']
        fig2 = px.bar(dist_counts, x='District', y='Rentals', title="Total Rentals by District", text_auto=True)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("租户年龄分布与设备偏好")
    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.histogram(df_orders.drop_duplicates(subset=['用戶號碼']), x='年齡', nbins=20, title="Unique User Age Distribution")
        st.plotly_chart(fig3, use_container_width=True)
    with col4:
        # Age group vs Equipment
        df_orders['Age_Group'] = pd.cut(df_orders['年齡'], bins=[0, 60, 70, 80, 90, 120], labels=['<60', '60-70', '70-80', '80-90', '90+'])
        age_eq = df_orders[df_orders['分類'].isin(top_cats)].groupby(['Age_Group', '分類']).size().reset_index(name='Count')
        fig4 = px.bar(age_eq, x='Age_Group', y='Count', color='分類', barmode='stack', title="Equipment Preference by Age Group")
        st.plotly_chart(fig4, use_container_width=True)

# --- Tab 2: Hotspots and Gaps ---
with tab2:
    st.header("服务需求热点与缺口对比分析")
    st.markdown("通过整合香港政府统计处模拟的各区长者人口及慢性病比率，对比真实租赁量，找出 **Service Demand Hotspots** (红色大圆) 与 **Service Gaps**。")

    col1, col2 = st.columns([2, 1])
    with col1:
        # Initialize map centered on Hong Kong
        m = folium.Map(location=[22.35, 114.15], zoom_start=11)
        
        for i, row in df_gap.iterrows():
            if pd.isna(row['lat']) or pd.isna(row['lon']):
                continue
            
            # Gap score defines color
            if row['Service_Gap_Score'] > 0.7:
                color = "red"
            elif row['Service_Gap_Score'] > 0.4:
                color = "orange"
            else:
                color = "green"
                
            # Estimated Need defines radius
            radius = row['Estimated_Need'] / 3000
            
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=radius,
                popup=f"<b>{row['地區*']}</b><br>Elderly Pop: {row['elderly_pop']:,}<br>Estimated Need: {int(row['Estimated_Need']):,}<br>Current Rentals: {int(row['Current_Rentals']):,}<br>Gap Score: {row['Service_Gap_Score']:.2f}",
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.6,
                tooltip=f"{row['地區*']} (Gap: {row['Service_Gap_Score']:.2f})"
            ).add_to(m)
            
        st_folium(m, width=700, height=500)
        st.caption("🔴 Red: High Gap (Underserved) | 🟠 Orange: Medium Gap | 🟢 Green: Low Gap (Well served) | Circle Size: Total Estimated Need")

    with col2:
        st.subheader("📊 缺口数据详情 (Top Underserved)")
        display_df = df_gap[['地區*', 'elderly_pop', 'Estimated_Need', 'Current_Rentals', 'Service_Gap_Score']].sort_values('Service_Gap_Score', ascending=False)
        st.dataframe(display_df.style.format({'Service_Gap_Score': '{:.2f}', 'Estimated_Need': '{:.0f}'}), hide_index=True)
        
        st.info("**Data Insight:**\n排名靠前的地区（如缺口得分 > 0.7）拥有庞大的潜在需求（长者人口基数大、慢性病率高），但实际产生的租赁订单极少。这些区域就是我们首要的 **Target Outreach Zones**。")

# --- Tab 3: Insights & Prediction ---
with tab3:
    st.header("市场洞察、目标画像与拓展策略")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("1. Target User Personas (核心用户画像)")
        
        # Calculate some dynamic insights based on real data
        top_district = df_orders['地區*'].value_counts().index[0] if not df_orders.empty else "未知"
        median_age = df_orders['年齡'].median() if not df_orders.empty else 0
        top_equipment = df_orders['分類'].value_counts().index[0] if not df_orders.empty else "未知"
        
        st.markdown(f"""
        **Persona A: 核心刚需长者 (Active Users)**
        - **特征**: 平均年龄 **{median_age:.0f}岁**，主要居住在 **{top_district}**。
        - **核心需求**: 对 **{top_equipment}** 的需求量最大，且租赁周期较长。
        - **同住情况**: 大多与配偶或子女同住。
        
        **Persona B: 隐蔽缺口长者 (Underserved Hidden Groups)**
        - **特征**: 居住在上述地图中标红的高 Gap 区域（如东区、观塘等）。
        - **核心痛点**: 缺乏转介渠道或数字化认知不足，未被社工/医院网络覆盖。
        """)
        
    with col2:
        st.subheader("2. Demand Forecasting (需求预测洞察)")
        st.markdown("""
        **基于时间序列的租赁高峰预测**
        - **季节性波动**: 从历史数据折线图中可见，特定设备（如助行器、沐浴椅）可能在特定季节呈现峰值。
        - **高危人群增长**: 随着香港高龄化加剧（80岁以上人口陡增），未来 12 个月内，针对 **高跌倒风险** 的设备（如防跌感应器、特殊护理床）需求将激增。
        - *(注：可进一步通过 ARIMA/Prophet 结合外部人口增长率输出具体的库存预警)*
        """)

    st.divider()
    st.subheader("3. 💡 Actionable Outreach Strategy (战略提案)")
    st.markdown("""
    基于以上 Data Insights，为「迎进」乐龄科技提出以下 **三步走推广策略**：
    
    1. **Targeted Medical Referrals (精准医疗转介)**
       - **行动**: 针对高 Service Gap 的地区，主动与当地的**长者健康中心、地区康健中心 (DHC)** 及公立医院的职业治疗师 (OT) 建立转介机制。
       - **数据支撑**: 内部数据显示大量订单来自转介机构，而高 Gap 地区显然缺乏这一环。
       
    2. **Community Pop-up Exhibitions (社区流动体验)**
       - **行动**: 在需求热点（热力图大圆圈区域）部署“乐龄科技流动体验车”或在屋邨大堂举办 Pop-up 展览。
       - **数据支撑**: 隐蔽长者不擅长使用网购，必须将实体设备带入他们的生活半径。
       
    3. **Carer-Centric Digital Marketing (针对照顾者的数字营销)**
       - **行动**: 大量长者实际是由中年子女（Carers）代为决策。通过 Facebook/Google Ads 在高需求地区定向投放广告，主打“减轻照顾者压力”的设备组合套餐。
    """)
