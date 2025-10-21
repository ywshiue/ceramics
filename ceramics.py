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



# -----------------------
def calculate_seger_moles_from_excel(df, selected_ingredients, amounts):
    total_moles = {}

    for ing in selected_ingredients:
        amt = amounts[ing]
        rows = df[df["成分名稱"] == ing]

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
        {"氧化物": ox, "摩爾數": info["moles"], "類別": info["category"]}
        for ox, info in total_moles.items()
    ]).sort_values(by="氧化物").reset_index(drop=True)

    ro_sum = total_moles_df[total_moles_df["類別"]=="RO"]["摩爾數"].sum()
    r2o3_sum = total_moles_df[total_moles_df["類別"]=="R2O3"]["摩爾數"].sum()
    ro2_sum = total_moles_df[total_moles_df["類別"]=="RO2"]["摩爾數"].sum()

    norm_factor = ro_sum if ro_sum > 0 else 1.0
    seger_ratios = {
        "RO 鹼類": ro_sum / norm_factor,
        "R2O3 中性物": r2o3_sum / norm_factor,
        "RO2 酸類": ro2_sum / norm_factor
    }

    seger_ratio_text = " + ".join([f"{round(v,2)} {k}" for k,v in seger_ratios.items()])
    return total_moles_df, seger_ratios, seger_ratio_text

# 賽格式計算
def glaze_ternary_app(excel_path="glaze_materials_streamlit.xlsx"):
    """
    Streamlit 賽格式計算器函式
    1. 讀取原料 Excel
    2. 選擇原料或配料
    3. 輸入總克重
    4. 選擇 21 點編號
    5. 計算賽格式克重
    6. 顯示三角形圖與表格
    """
    
    st.title("賽格式計算器")
    
    # 讀取 Excel
    df = pd.read_excel(excel_path)
    
    # 選擇原料
    selected_materials = st.multiselect(
        "選擇原料或配料",
        options=df['名稱'].tolist()
    )
    
    df_selected = df[df['名稱'].isin(selected_materials)]
    st.subheader("選擇的原料")
    st.dataframe(df_selected)
    
    # 輸入總克重
    total_weight = st.number_input("輸入總克重 (克)", min_value=0.0, value=100.0, step=1.0)
    
    # 21 點比例表 (示範，可改為 Excel 匯入)
    max_val = 50
    step = 10
    n = 6
    data = []
    number = 1
    for i in reversed(range(n)):
        for j in range(n-i):
            z_val = i * step
            y_val = j * step
            x_val = max_val - y_val - z_val
            data.append([number, x_val, y_val, z_val])
            number += 1
    df_ratio = pd.DataFrame(data, columns=['編號','X_ratio','Y_ratio','Z_ratio'])
    
    # 選擇編號
    selected_number = st.selectbox("選擇 21 點編號", df_ratio['編號'].tolist())
    row = df_ratio[df_ratio['編號']==selected_number].iloc[0]
    
    # 計算賽格式克重
    factor = total_weight / max_val
    X_weight = int(row['X_ratio'] * factor)
    Y_weight = int(row['Y_ratio'] * factor)
    Z_weight = int(row['Z_ratio'] * factor)
    
    st.subheader(f"選擇編號 {selected_number} 的賽格式克重")
    st.write(f"X: {X_weight} g, Y: {Y_weight} g, Z: {Z_weight} g")
    
    # 畫三角形比例圖
    fig, ax = plt.subplots(figsize=(6,6))
    for idx, r in df_ratio.iterrows():
        i = int((50 - r['X_ratio'] - r['Y_ratio']) // step)
        j = int(r['Y_ratio'] // step)
        x0, y0 = (j + 0.5*i)/n, i*np.sqrt(3)/(2*n)
        x1, y1 = x0 + 1/n, y0
        x2, y2 = x0 + 0.5/n, y0 + np.sqrt(3)/(2*n)
        tri = Polygon([[x0,y0],[x1,y1],[x2,y2]], closed=True,
                      edgecolor='lightgray', facecolor='none', lw=0.8)
        ax.add_patch(tri)
        x_center = (x0 + x1 + x2)/3
        y_center = (y0 + y1 + y2)/3
        number_text = str(int(r['編號']))
        xyz_text = f"{int(r['X_ratio']*factor)},{int(r['Y_ratio']*factor)},{int(r['Z_ratio']*factor)}"
        ax.text(x_center, y_center + 0.01, number_text, ha='center', va='bottom', fontsize=6, color='blue', weight='bold')
        ax.text(x_center, y_center - 0.01, xyz_text, ha='center', va='top', fontsize=6, color='black')

    triangle = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2],[0,0]])
    ax.plot(triangle[:,0], triangle[:,1], color='black', lw=2)
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)
    

def glaze_ternary_21points_numbered():
    st.title("釉料三軸表")
    
    # 使用者輸入總克重（單一欄位）
    total_weight = st.number_input("總克重 (克)", min_value=0.0, value=50.0, step=1.0)
    
    
    # 參數設定
    max_val = 50
    step = 10
    n = 6  # 每邊 6 等分
    
    # 建立小三角形 XYZ 比例表
    data = []
    number = 1
    for i in reversed(range(n)):   # 從上到下
        for j in range(n-i):       # 每排從左到右
            z_val = i * step
            y_val = j * step
            x_val = max_val - y_val - z_val
            data.append([number, x_val, y_val, z_val])
            number += 1
    
    df_ratio = pd.DataFrame(data, columns=['編號','X_ratio','Y_ratio','Z_ratio'])
    
    # 計算比例因子
    factor = total_weight / max_val  # 總克重除以 max_val，按比例分配
    
    # 計算每個小三角形克重
    df_ratio['X (克)'] = (df_ratio['X_ratio'] * factor).round(0).astype(int)
    df_ratio['Y (克)'] = (df_ratio['Y_ratio'] * factor).round(0).astype(int)
    df_ratio['Z (克)'] = (df_ratio['Z_ratio'] * factor).round(0).astype(int)

    
    # 畫圖
  
    fig, ax = plt.subplots(figsize=(6,6))
    
    for idx, row in df_ratio.iterrows():
        # 對應 i,j
        i = int((50 - row['X_ratio'] - row['Y_ratio']) // step)
        j = int(row['Y_ratio'] // step)
    
        # 小三角形頂點
        x0, y0 = (j + 0.5*i)/n, i*np.sqrt(3)/(2*n)
        x1, y1 = x0 + 1/n, y0
        x2, y2 = x0 + 0.5/n, y0 + np.sqrt(3)/(2*n)
    
        # 畫三角形
        tri = Polygon([[x0,y0],[x1,y1],[x2,y2]], closed=True,
                      edgecolor='lightgray', facecolor='none', lw=0.8)
        ax.add_patch(tri)
    
        # 計算重心
        x_center = (x0 + x1 + x2)/3
        y_center = (y0 + y1 + y2)/3
    
        # 編號文字顏色（例如紅色），XYZ 克重文字黑色
        number_text = f"{int(row['編號'])}"
        xyz_text = f"{row['X (克)']},{row['Y (克)']},{row['Z (克)']}"
        
        # 編號
        ax.text(x_center, y_center + 0.01, number_text, ha='center', va='bottom', fontsize=6, color='blue', weight='bold')
        # XYZ 克重
        ax.text(x_center, y_center - 0.01, xyz_text, ha='center', va='top', fontsize=6, color='black')

    
    # 畫正三角形邊界
    triangle = np.array([[0,0],[1,0],[0.5,np.sqrt(3)/2],[0,0]])
    ax.plot(triangle[:,0], triangle[:,1], color='black', lw=2)
    
    ax.set_aspect('equal')
    ax.axis('off')
    st.pyplot(fig)
    
    # 顯示表格

    st.dataframe(df_ratio[['編號','X (克)','Y (克)','Z (克)']])
    
# -----------------------
def glaze_app(excel_path="glaze_ingredients.xlsx"):
    st.title("Seger")

    df = pd.read_excel(excel_path)
    df.columns = df.columns.str.strip()

    selected_ingredients = st.multiselect(
        "選擇原料",
        df["成分名稱"].unique().tolist(),
        default=df["成分名稱"].unique().tolist()[:3]
    )

    amounts = {}
    for ing in selected_ingredients:
        amounts[ing] = st.number_input(
            f"{ing} 的重量 (g)",
            min_value=0.0,
            max_value=1000.0,
            value=10.0,
            step=1.0
        )

    if st.button("計算並繪圖"):
        total_moles_df, seger_ratios, seger_ratio_text = calculate_seger_moles_from_excel(df, selected_ingredients, amounts)
        st.subheader("各氧化物原始摩爾數")
        st.dataframe(total_moles_df)
        st.subheader("Seger 比例 ( RO 為1)")
        st.code(seger_ratio_text)

    
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
page = st.sidebar.radio("選擇頁面", ("收縮率計算", "石膏板材料計算", "電窯成本計算", "釉料預測","賽格式計算器","釉料三軸表"))

if page == "電窯成本計算":
    st.subheader("請輸入尺寸")

    # 使用 columns() 將長、寬、高的輸入框顯示在同一行
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # 允許用戶留空，並檢查是否是有效的數字
        length_input = st.text_input("長 (cm):")
        length = None
        if length_input:
            try:
                length = float(length_input)
                if length <= 0:
                    st.warning("請輸入大於0的數字。")
                    length = None
            except ValueError:
                st.warning("請輸入有效的數字。")

    with col2:
        width_input = st.text_input("寬 (cm):")
        width = None
        if width_input:
            try:
                width = float(width_input)
                if width <= 0:
                    st.warning("請輸入大於0的數字。")
                    width = None
            except ValueError:
                st.warning("請輸入有效的數字。")

    with col3:
        height_input = st.text_input("高度 (cm):")
        height = None
        if height_input:
            try:
                height = float(height_input)
                if height <= 0:
                    st.warning("請輸入大於0的數字。")
                    height = None
            except ValueError:
                st.warning("請輸入有效的數字。")
        
    # 檢查是否填寫了所有必要的尺寸
    if length and width and height:
        dimensions = {'長': length, '寬': width, '高度': height}
    else:
        dimensions = {}

    # 設置計算價格的按鈕
    calculate_button = st.button("計算價格")
    reset_button = st.button("清零")  # 用於清空輸入

    # 初始化總成本和作品計數變數
    if 'total_cost' not in st.session_state:
        st.session_state.total_cost = 0  # 初始化總成本為 0
    if 'work_counter' not in st.session_state:
        st.session_state.work_counter = 1  # 初始化作品計數器，從第1個作品開始
    if 'costs_history' not in st.session_state:
        st.session_state.costs_history = []  # 初始化歷史成本記錄

    # 當用戶點擊計算價格按鈕後，顯示價格
    if calculate_button:
        if dimensions:
            cost = calculate_cost(dimensions)  # 假設你已經有這個 calculate_cost 函數
            cost = round(cost, -1)  # 四捨五入
            
            # 保存這次的計算結果到歷史，包含尺寸與價格
            st.session_state.costs_history.append({
                'dimensions': dimensions,
                'cost': cost
            })
            
            # 累加成本
            st.session_state.total_cost += cost
            
            # 增加作品計數器
            st.session_state.work_counter += 1
        else:
            st.warning("請填寫尺寸資訊以計算價格。")

    # 當用戶點擊清空按鈕，重置輸入框並計算下一個作品
    if reset_button:
        # 清空已經輸入的尺寸值
        length = width = height = 0
        dimensions = {}
        # 清空歷史成本
        st.session_state.costs_history = []
        st.session_state.total_cost = 0  # 清空累計總成本
        st.session_state.work_counter = 1  # 重置作品計數器，從第1個作品開始

    # 顯示歷史計算的價格和尺寸
    if st.session_state.costs_history:
        st.markdown("<h5>歷史價格</h5>", unsafe_allow_html=True)
        for idx, record in enumerate(st.session_state.costs_history, 1):
            dimensions = record['dimensions']
            cost = record['cost']
            st.write(f"第 {idx} 個作品價格: {cost} 元, 尺寸: {dimensions['長']} x {dimensions['寬']} x {dimensions['高度']} cm")
        
    # 顯示總成本
    st.subheader(f"總成本: {st.session_state.total_cost} 元")

elif page == "石膏板材料計算":
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
                clay_type = "黃陶土"
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

elif page == "釉料預測":
    st.subheader(f"施工中")

# --- 賽格式計算器頁面 ---
elif page == "賽格式計算器":
    glaze_app()

elif page == "釉料三軸表":
    glaze_ternary_21points_numbered()
    
