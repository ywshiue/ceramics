import pandas as pd
import numpy as np
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from PIL import Image
import io
import requests
import base64
import ternary
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


# =========================
# ① Seger 計算核心
# =========================
def calculate_seger_moles_from_excel(df, selected_ingredients, amounts):

    total_moles = {}

    for ing in selected_ingredients:
        amt = amounts[ing]
        rows = df[df["名稱"] == ing]

        for _, row in rows.iterrows():
            ox_list = [ox.strip() for ox in str(row["氧化物組成"]).split('+')]

            try:
                percent = float(row["百分比"])
                mw = float(row["分子量（g/mol）"])
                category = str(row["類別"]).strip()
            except:
                continue

            moles_per_ox = (amt * percent / 100) / mw / len(ox_list)

            for ox in ox_list:
                if ox not in total_moles:
                    total_moles[ox] = {"moles": 0.0, "category": category}

                total_moles[ox]["moles"] += moles_per_ox

    total_moles_df = pd.DataFrame([
        {"氧化物": ox, "摩爾數": v["moles"], "類別": v["category"]}
        for ox, v in total_moles.items()
    ]).sort_values("氧化物").reset_index(drop=True)

    ro_sum = total_moles_df[total_moles_df["類別"] == "RO"]["摩爾數"].sum()
    r2o3_sum = total_moles_df[total_moles_df["類別"] == "R2O3"]["摩爾數"].sum()
    ro2_sum = total_moles_df[total_moles_df["類別"] == "RO2"]["摩爾數"].sum()

    norm = ro_sum if ro_sum > 0 else 1.0

    seger = {
        "RO": ro_sum / norm,
        "R2O3": r2o3_sum / norm,
        "RO2": ro2_sum / norm
    }

    return total_moles_df, seger


# =========================
# ② 三軸點生成（66點）
# =========================
def generate_ternary_points(n=11, step=10):
    max_val = step * (n - 1)

    data = []
    num = 1

    for i in reversed(range(n)):
        for j in range(n - i):
            z = i * step
            y = j * step
            x = max_val - y - z
            data.append([num, x, y, z])
            num += 1

    return pd.DataFrame(data, columns=["編號", "RO", "RO2", "R2O3"])


# =========================
# ③ 三角圖畫圖
# =========================
def draw_ternary(df, title="Ternary"):
    fig, ax = plt.subplots(figsize=(6, 6))

    n = int(np.sqrt(len(df))) + 1
    den = n - 1

    for _, row in df.iterrows():

        i = int((df["RO"].max() - row["RO"] - row["RO2"]) // 10)
        j = int(row["RO2"] // 10)

        x0, y0 = (j + 0.5 * i) / den, i * np.sqrt(3) / (2 * den)
        x1, y1 = x0 + 1 / den, y0
        x2, y2 = x0 + 0.5 / den, y0 + np.sqrt(3) / (2 * den)

        tri = Polygon([[x0, y0], [x1, y1], [x2, y2]],
                      closed=True, edgecolor="lightgray", facecolor="none", lw=0.6)
        ax.add_patch(tri)

        cx = (x0 + x1 + x2) / 3
        cy = (y0 + y1 + y2) / 3

        ax.text(cx, cy, str(row["編號"]),
                ha="center", va="center", fontsize=6)

    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title)

    return fig


# =========================
# ④ 主 App
# =========================
def glaze_app(excel_path="glaze_ingredients.xlsx"):

    st.title("三軸 + Seger 整合釉料系統")

    # 讀資料
    df = pd.read_excel(excel_path)

    # 材料選擇
    materials = st.multiselect("選擇材料", df["名稱"].tolist())

    # 克重
    total_weight = st.number_input("總克重", 100.0)

    # 建立 66 點
    ternary = generate_ternary_points()

    st.subheader("三軸設計空間 (RO / RO2 / R2O3)")
    st.pyplot(draw_ternary(ternary))

    # 選點
    idx = st.selectbox("選擇點位編號", ternary["編號"])

    row = ternary[ternary["編號"] == idx].iloc[0]

    st.subheader("該點克重分配")

    st.write({
        "RO": row["RO"],
        "RO2": row["RO2"],
        "R2O3": row["R2O3"]
    })

    # 材料克重（簡化：平均分配）
    if materials:

        weights = {m: total_weight / len(materials) for m in materials}

        st.subheader("Seger 計算")

        seger_df, seger = calculate_seger_moles_from_excel(
            df, materials, weights
        )

        st.dataframe(seger_df)
        st.write("Seger Ratio:", seger)

    
def calculate_plaster(length, width, height, proportion):
    if length  is not None and width is not None and height is not None and proportion is not None:
        original_size = length * width * height
        plaster_weight = round(original_size /((1/ proportion)+(1/2.21)),2)
    else:
        original_size = None
        plaster_weight = None
        water_weight = None
    return plaster_weight
    
def calculate_water(length, width, height, proportion):
    if length  is not None and width is not None and height is not None and proportion is not None:
        original_size = length * width * height
        plaster_weight = original_size /((1/ proportion)+(1/2.21))
        water_weight = round(plaster_weight/proportion,2)
    else:
        original_size = None
        plaster_weight = None
        water_weight = None
    return water_weight
    
    
# 計算收縮前尺寸的函數
def calculate_original_size(length, shrinkage):
    """
    計算收縮前的尺寸
    """
    if length is not None:
        original_length = round(length / (1 - shrinkage), 2)
    else:
        original_length = None
    return original_length

# 計算收縮後尺寸的函數
def calculate_shrinked_size(length, shrinkage):
    """
    計算收縮後的尺寸
    """
    if length is not None:
        shrinked_length = round(length * (1 - shrinkage), 2)
    else:
        shrinked_length = None
    return shrinked_length

    
# 定義計算圓形和方形成本的函數
def calculate_cost(dimensions):
    material_cost_per_unit = 0.15  # 假設每個單位的材料成本為 0.15 (這個可以根據實際情況調整)
    
    length = dimensions['長']
    width = dimensions['寬']
    height = dimensions['高度']
    volume = length * width * height  # 長方體積公式
    cost = volume * material_cost_per_unit
    return cost

# 釉料預測的邏輯
def glaze_forecast():
    # 加載數據集
    file_path = '/Users/xueyiwen/ywshiue/Python project/Glaze Data/Train data/glaze data.xlsx'
    data = pd.read_excel(file_path)

    # 數據處理
    X = data[['R', 'G', 'B']]  # RGB 顏色
    y = data[['長石 (%)', '矽石 (%)', '鋁土礦 (%)', '氧化鈉 (%)', '氧化鈣 (%)', '氧化鎂 (%)', 
              '氧化鐵 (%)', '氧化鈷 (%)', '氧化銅 (%)', '氧化錳 (%)', '氧化鉻 (%)', '氧化鈦 (%)']]  # 釉料成分比例

    # 數據標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 分割數據集
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 訓練模型
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.title("釉料配方預測")
    st.write("根據顏色的 RGB 值預測釉料的成分比例")

    # 圖片上傳
    uploaded_file = st.file_uploader("上傳圖片以提取顏色", type=["jpg", "jpeg", "png"])

    # 預設顏色為 RGB(100, 50, 30)
    r, g, b = 0, 0, 0
    # 顯示圖片、顏色區塊和 RGB 顏色在同一行
    col1, col2, col3 = st.columns([1, 2, 3])
    # 如果上傳了圖片，提取顏色並顯示
    if uploaded_file is not None:

        # 顯示圖片
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="上傳的圖片", use_column_width=False, width=100,)

        # 計算圖片中心的顏色
        image = image.convert('RGB')
        image = np.array(image)
        
        # 確保圖片是RGB格式
        if image.ndim == 3 and image.shape[2] == 3:
            center_x = image.shape[0] // 2
            center_y = image.shape[1] // 2
            size = 50  # 提取周圍50x50像素範圍
            region = image[center_x-size:center_x+size, center_y-size:center_y+size]
            avg_color = np.mean(region, axis=(0, 1))
            r, g, b = avg_color
            
            with col2:               
                # with col3:
                st.markdown(f'<div style="background-color: rgb({int(r)}, {int(g)}, {int(b)}); height: 100px; width: 100px; text-align: center; display: block; margin: 0 auto;"></div>', unsafe_allow_html=True)
                st.markdown(f'<div style="text-align: center;">RGB ({int(r)}, {int(g)}, {int(b)})</div>', unsafe_allow_html=True)
    # RGB 拉桿選擇
    st.subheader("選擇顏色 (可選擇 RGB 拉桿)：")
    r = st.slider('紅色成分 (R)', min_value=0, max_value=255, value=int(r))
    g = st.slider('綠色成分 (G)', min_value=0, max_value=255, value=int(g))
    b = st.slider('藍色成分 (B)', min_value=0, max_value=255, value=int(b))

    # 顯示選擇的顏色
    
    with col3:
        st.markdown(f'<div style="background-color: rgb({r}, {g}, {b}); height: 100px; width: 100px; text-align: center; display: block; margin: 0 auto;"></div>', unsafe_allow_html=True)
        st.markdown(f'<div style="text-align: center;">slider RGB ({int(r)}, {int(g)}, {int(b)})</div>', unsafe_allow_html=True)
    # 使用選擇的顏色進行預測
    new_rgb = np.array([[r, g, b]])  # 新的 RGB 顏色
    new_rgb_scaled = scaler.transform(new_rgb)  # 標準化處理

    # 預測
    predicted_glaze = model.predict(new_rgb_scaled)

    # 顯示預測結果
    st.write("預測的釉料配方：")
    glaze_df = pd.DataFrame(predicted_glaze, columns=['長石 (%)', '硅石 (%)', '鋁土礦 (%)', '氧化鈉 (%)', '氧化鈣 (%)', 
                                                     '氧化鎂 (%)', '氧化鐵 (%)', '氧化鈷 (%)', '氧化銅 (%)', 
                                                     '氧化錳 (%)', '氧化鉻 (%)', '氧化鈦 (%)'])
    st.write(glaze_df)

# 主頁面設置
st.sidebar.title("選擇功能")
page = st.sidebar.radio("選擇頁面", ("收縮率計算", "石膏板材料計算","賽格式計算","釉料三軸表"))

if page == "石膏板材料計算":
    st.subheader("石膏板材料計算(石膏比重採2.21)")
    length_input = st.text_input("長 (cm):")
    width_input = st.text_input("寬 (cm):")
    height_input = st.text_input("高 (cm):")
    proportion_input = st.text_input("水與石膏的重量比例 (1.2~1.6):")
    calculate_button = st.button("材料計算")
    if calculate_button:
            if length_input is not None and width_input is not None and height_input is not None and proportion_input is not None:
                length = float(length_input)
                width=float(width_input)
                height=float(height_input)
                proportion=float(proportion_input)

                plaster_weight = calculate_plaster(length, width, height, proportion)
                water_weight = calculate_water(length, width, height, proportion)
                st.markdown("<h5>石膏重量</h5>", unsafe_allow_html=True)
                st.write(f"克重: {plaster_weight} g")
            
                st.markdown("<h5>水重量</h5>", unsafe_allow_html=True)
                st.write(f"克重: {water_weight} g")
            
            
elif page == "收縮率計算":
    st.subheader("請輸入尺寸與收縮率")

    col1, col2 = st.columns(2)
    
    with col1:
        length_input = st.text_input("尺寸 (cm):")
        try:
            length = float(length_input)
            if length < 1.0 or length > 100.0:
                st.error("請輸入 1～100 的數字")
                length = None
        except ValueError:
            length = None
    
    with col2:
        shrinkage_rate = st.text_input("收縮率 (%):")
        try:
            shrinkage = float(shrinkage_rate)
            if shrinkage < 1.0 or shrinkage > 100.0:
                st.error("請輸入 1～100 的數字")
                shrinkage = None
            else:
                shrinkage = shrinkage * 0.01
        except ValueError:
            shrinkage = None

    # ===== 單一計算 =====
    calculate_button = st.button("單一收縮率計算")
    if calculate_button and length is not None and shrinkage is not None:
        shrinked_length = calculate_shrinked_size(length, shrinkage)
        original_length = calculate_original_size(length, shrinkage)    
        st.markdown("<h5>收縮前的坯體尺寸</h5>", unsafe_allow_html=True)
        st.write(f"尺寸: {original_length:.2f} cm")   # 保留兩位
        st.markdown("<h5>收縮後的成品尺寸</h5>", unsafe_allow_html=True)
        st.write(f"尺寸: {shrinked_length:.2f} cm")  # 保留兩位

    # ===== 自動對照表 =====
    if length is not None:
        shrinkage_rates = list(range(8, 16))  # 5%~15%
        results = []
        for rate in shrinkage_rates:
            s = rate / 100
            shrinked_length = calculate_shrinked_size(length, s)
            original_length = calculate_original_size(length, s)

            # 土料對應
            if rate == 8:
                clay_type = "雕塑土"
            elif rate == 10:
                clay_type = "黃陶土,信樂土"
            else:
                clay_type = ""

            results.append({
                "收縮率 (%)": f"{rate}%",
                "收縮前尺寸 (cm)": f"{original_length:.2f}",  # 兩位小數
                "輸入尺寸 (cm)": f"{length:.2f}",             # 兩位小數
                "收縮後尺寸 (cm)": f"{shrinked_length:.2f}",  # 兩位小數
                "對應土料": clay_type
            })

        df = pd.DataFrame(results)

        # 顏色標示
        highlight_rates = ["8%", "10%", "13%", "15%"]

        def highlight_rows(row):
            if row["收縮率 (%)"] in highlight_rates:
                return ["background-color: yellow"] * len(row)
            else:
                return [""] * len(row)

        styled_df = df.style.apply(highlight_rows, axis=1)

        st.markdown("### 收縮率對照表 (8% ~ 15%)")
        st.dataframe(styled_df, use_container_width=True)

# --- 賽格式計算器頁面 ---
elif page == "賽格式計算":
    glaze_app()

elif page == "釉料三軸表":
    glaze_ternary_21points_numbered()
    
