import pandas as pd
import numpy as np
import folium
import matplotlib
import matplotlib.pyplot as plt
import japanize_matplotlib
import seaborn as sns
sns.set_style('darkgrid')
sns.set(font="IPAexGothic")
matplotlib.style.use('ggplot')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import streamlit as st
from streamlit_folium import folium_static

# ページ設定
st.set_page_config(layout="wide")

# サイドバーに画像を表示
st.sidebar.image(
    'data/hanrei.jpg',
    use_column_width=True
)

###タイトル
#st.info("TOLES α版")
st.image('data/TOLES.svg')

### Data Load
##旧犯罪データ
# df_hanzai = pd.read_csv('./data/R5_hanzai.csv', encoding='sjis')
# # df_hanzai = df_hanzai.iloc[:5172]
# # df_hanzai = df_hanzai.iloc[:116]#千代田区のみ
# df_hanzai = df_hanzai[df_hanzai['市区町丁'].str.contains('千代田区|板橋区')]#252
# df_hanzai = df_hanzai[~df_hanzai['市区町丁'].isin(["区計","市計"])][['市区町丁','総合計']]

##犯罪データ変更後
df_hanzai = pd.read_csv('./data/df_hanzai_add_nearby_loc_add_hinanjo.csv', encoding='cp932')

##街灯データ
df_gaitou = pd.read_csv('./data/LED_街灯_2021作成データ.csv', encoding='sjis')
df_gaitou = df_gaitou.iloc[:4000:5]#全4121件、処理重いので一旦800件、板橋区のデータのみ

##騒音データ
df_R4 = pd.read_csv('./data/自動車_常時監視測定地点_令和4年_add_location.csv', encoding='cp932')
df_R4 = df_R4[df_R4['測定地点の住所'].str.contains('千代田区|中央区|港区|板橋区|練馬区')]

## 板橋区_避難所データ
df_hinanjo = pd.read_csv('./data/板橋区_避難所データ.csv', encoding='cp932')
df_hinanjo = df_hinanjo[['施設名','緯度','経度']]
df_hinanjo = df_hinanjo.rename({'緯度':'LATITUDE', '経度':'LONGITUDE'}, axis=1)

def safe_literal_eval(value):
    try:
        return ast.literal_eval(value)
    except (ValueError, SyntaxError):
        return None

###0.住環境スコア化
import ast
def get_each_score(row):
    try:
        list_lamp = ast.literal_eval(row['nearby_locat_街灯'])
    except (ValueError, SyntaxError) as e:
        list_lamp = []
    len_hinanjo = len(row['nearby_locat_避難所'])    
    try:
        list_noise = ast.literal_eval(row['noise_levels'])
        if isinstance(list_noise, list) and len(list_noise) > 0:
            list_noise = list_noise[0]
        else:
            list_noise = []
    except (ValueError, SyntaxError, IndexError) as e:
        list_noise = []
    
    if list_noise and isinstance(list_noise, list) and len(list_noise) > 0:
        avg_noise_level = sum(list_noise) / len(list_noise)
    else:
        avg_noise_level = np.nan
    return [len(list_lamp), avg_noise_level, len_hinanjo]
df_hanzai[['街灯の数', '騒音の平均値', '避難所の数']] = df_hanzai.apply(lambda x: pd.Series(get_each_score(x)), axis=1)

# ---------------------
# 正規化
# ---------------------
from sklearn.preprocessing import StandardScaler, MinMaxScaler
print(df_hanzai.shape)
score_list = ['総合計','街灯の数','騒音の平均値','避難所の数']
df_score = df_hanzai.replace(0, np.nan)
for col in score_list:
  scaler = MinMaxScaler()
  # NaNを削除して1次元の配列に変換
  data = df_score[col].dropna().values.reshape(-1, 1)
  df_values = scaler.fit_transform(data)
  # 標準化された値を元のデータフレームに格納
  df_score.loc[df_score[col].dropna().index, f"{col}_normal"] = df_values.flatten()
    
# Nanがあるとenv_scoreが計算できていないので0で欠損値を補間
# score_normal_list = ['総合計_normal','街灯の数_normal','騒音の平均値_normal']
score_normal_list = ['総合計_normal','街灯の数_normal','騒音の平均値_normal','避難所の数_normal']
df_score[score_normal_list] = df_score[score_normal_list].fillna(0)

# =======================================================================================
# パラメータ
# =======================================================================================
WEIGHT_CRIME = -0.3
WEIGHT_NOISE = -0.3
WEIGHT_LAMP = 0.3
WEIGHT_ESCAPE = 0.5
df_score['env_score'] = df_score['総合計_normal'] * WEIGHT_CRIME + df_score['騒音の平均値_normal'] * WEIGHT_NOISE + df_score['街灯の数_normal'] * WEIGHT_LAMP

# ----------------------------------------
# スコアがマイナスになるので正規化しておく
# ----------------------------------------
scaler = MinMaxScaler()
# NaNを削除して1次元の配列に変換
data = df_score['env_score'].dropna().values.reshape(-1, 1)
df_values = scaler.fit_transform(data)
# 元の処理
df_score.loc[df_score['env_score'].dropna().index, "env_score_normal"] = df_values.flatten()
# 小数点以下2桁に丸めてから100倍し、整数に変換
df_score["env_score_normal"] = (df_score["env_score_normal"].round(2) * 100).astype(int)

###1.住所から緯度経度情報を取得する
import random
from geopy.geocoders import Nominatim
def get_coordinates(address):
  user_agent = f"geo_{random.randint(1000, 9999)}"
  geolocator = Nominatim(user_agent=user_agent)
  location = geolocator.geocode(address)
  if location is None:
    return np.nan, np.nan
  return location.latitude, location.longitude

### 2. 地理情報システム (GIS) データを取得する（例: GeoPandasを使用）
# GISデータから市区町村の境界データを取得
import geopandas as gpd
from shapely.geometry import Point
gdf = gpd.read_file('./data/r2ka13.shp')

### 3. 住所が含まれる地域のポリゴンを抽出
def get_polygon_for_address(address):
  latitude, longitude = get_coordinates(address)
  if latitude is np.nan or longitude is np.nan:
    return np.nan, np.nan, np.nan
  else:
    point = Point(longitude, latitude)
    if len(gdf[gdf.geometry.contains(point)]) == 0:
      return np.nan, np.nan, np.nan
    polygon = gdf[gdf.geometry.contains(point)].geometry.values[0]
  return polygon, latitude, longitude

### 4. ポリゴン情報を取得
# サンプルスタート位置情報
address = "東京都千代田区丸の内１丁目"
# 住所を入力してMAPを移動
# address = st.text_input("MAP上で移動したい住所を入力してください：", "千代田区麹町４丁目")
# サンプル位置情報のポリゴン情報を取得
polygon, latitude, longitude = get_polygon_for_address(address)
# サンプル位置情報をスタート位置としてMapを作成
# map = folium.Map(location=[latitude, longitude], zoom_start=15)

###スタート位置を板橋区に変更
start_lat = df_score[df_score['市区町丁'].str.contains('板橋区')]['LATITUDE'].values[0]
start_long = df_score[df_score['市区町丁'].str.contains('板橋区')]['LONGITUDE'].values[0]
map = folium.Map(location=[start_lat, start_long], zoom_start=16) # zoom:大きいと解像度が高い


def map_crime(map, df):
  for index, row in df.iterrows():
    address = row['市区町丁']
    crime = row['総合計']
    # ポリゴン情報を取得
    polygon, latitude, longitude = get_polygon_for_address(address)
    if polygon is np.nan or latitude is np.nan or longitude is np.nan:
      continue
    # ポリゴンを地図上に表示
    style_function = lambda x: {
      'fillColor': 'purple',
      'color': 'purple', # エッジの色
      'weight': 1.5 # エッジの太さ
    }
    folium.GeoJson(polygon, style_function=style_function).add_to(map)
    # 文字列を表示するカスタムアイコンを定義
    html = f'<div style="font-size: 10pt; color: gray; width: 300px;">住所エリア : {address}<br><br>犯罪件数 : {crime}</div>'
    folium.Marker(
      [latitude, longitude],
      icon=folium.Icon(color='purple'),
      popup=html
    ).add_to(map)

    # スタイルを追加してポップアップのウィンドウサイズを変更
    map.get_root().header.add_child(
      folium.Element('<style>.leaflet-popup-content-wrapper {width: 300px !important; height: auto !important;}</style>')
      )

  return map

### 騒音
def map_noise(map, df_R4):
  for index, row in df_R4.iterrows():
    # 住所情報
    address = row['測定地点の住所']
    road = row['評価対象道路①']
    value = row['騒音レベル_昼']

    # サークルをプロット
    folium.Circle(
      location=[row['LATITUDE'], row['LONGITUDE']],
      radius=value * 2, # サークルの大きさ調整
      color='', # サークルの囲い無し
      fill=True,
      fill_color='gray',
      fill_opacity=0.3,
    ).add_to(map)

    # 文字列を表示するカスタムアイコンを定義
    html = f'<div style="font-size: 10pt; color: gray;width: 200px;">道路 : {road}<br>騒音レベル : {value}</div>'
    folium.Marker(
      [row['LATITUDE'], row['LONGITUDE']],
      icon=folium.Icon(color='gray'),
      popup=html
    ).add_to(map)

  return map

###街灯
def map_lights(map, df_gaitou):
    value = 15  # とりあえず街灯は固定値
    for index, row in df_gaitou.iterrows():
        # 街灯のサークルをプロット（アイコンは無し）
        folium.Circle(
            location=[row['緯度'], row['経度']],
            radius=value,
            color='',  # サークルの囲い無し
            fill=True,
            fill_color='gold',
            fill_opacity=0.5,
        ).add_to(map)

        # 文字列を表示するカスタムアイコンを定義
        # html = f'<div style="font-size: 10pt; color: gray; width: 200px;">所在地 : {row["所在地"]}<br>街灯形式 :{row["街灯形式"]}<</div>'
        # folium.Marker(
        #     [row['緯度'], row['経度']],
        #     icon=folium.Icon(color='yellow'),
        #     popup=html
        # ).add_to(map)
    return map

import folium

### 避難所
def map_fixed_circles(map, df_hinanjo):
    for index, row in df_hinanjo.iterrows():
        value = 80  # サークルの固定値

        # サークルをプロット
        folium.Circle(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=value,
            color='',  # サークルの囲い無し
            fill=True,
            fill_color='aqua',
            fill_opacity=0.4,
        ).add_to(map)

        # 文字列を表示するカスタムアイコンを定義
        # html = f'<div style="font-size: 10pt; color: aqua;">サークル半径 : {value}</div>'
        # folium.Marker(
        #     [row['LATITUDE'], row['LONGITUDE']],
        #     icon=folium.Icon(color='lightblue'),
        #     popup=html
        # ).add_to(map)
    return map

import folium
import math

###スコア
def map_environmental_scores(map, df_score):
    for index, row in df_score[df_score['市区町丁'].str.contains('板橋区')].iterrows():
        value = row['env_score_normal'] * 2 #適切な半径に変更

        if math.isnan(row['LATITUDE']) or math.isnan(row['LONGITUDE']):
            continue

        folium.Circle(
            location=[row['LATITUDE'], row['LONGITUDE']],
            radius=value,
            color='',
            fill=True,
            fill_color='blue',
            fill_opacity=0.3,
        ).add_to(map)

        # Add a popup with additional information (optional)
        html = f'<div style="font-size: 10pt; color: blue;">スコア: {row["env_score_normal"]}</div>'
        folium.Marker(
            [row['LATITUDE'], row['LONGITUDE']],
            icon=folium.Icon(color='blue'),
            popup=html
        ).add_to(map)
    return map

###ポリゴン生成（選択した板橋区）
## (手法A)固定
# map = map_crime(map, df_hanzai[:1])

##（手法A'）板橋区全選択※API上限エラー
# filtered_df = df_hanzai[df_hanzai['市区町丁'].str.contains('板橋区')]

## (手法B)表示件数選択
# num_records = st.slider("表示する犯罪記録の数を選択してください：", min_value=1, max_value=len(df_hanzai), value=1)
# map = map_crime(map, df_hanzai[:num_records])

## (手法C)板橋区の市区町丁を選択
fil_df = df_hanzai[df_hanzai['市区町丁'].str.contains('板橋区')]
cho_list = fil_df['市区町丁'].unique().tolist()
selected_cho = st.multiselect(
    "住所を選択してください（α版は板橋区のみ）：",
    cho_list,
    default=cho_list[:1]
)
filtered_df = df_hanzai[df_hanzai['市区町丁'].isin(selected_cho)]


###マッピング
map = map_crime(map, filtered_df)
map = map_noise(map, df_R4)
map = map_lights(map, df_gaitou)
map = map_fixed_circles(map, df_hinanjo)
map = map_environmental_scores(map, df_score)
# Streamlitでマップを表示する
folium_static(map, width=None, height=500)#ワイド版でない時は725


### 選択した板橋区の市区町丁の詳細情報
df_syousai= df_score[df_score['市区町丁'].isin(selected_cho)]
# st.dataframe(df_syousai.style.highlight_max(axis=0)) #dfで表示
# 住環境スコアの降順でソート
sorted_df = df_syousai.sort_values(by='env_score_normal', ascending=False).reset_index(drop=True)
# 選択件数を取得
total_entries = len(sorted_df)
# 詳細表示
num_columns = 4
columns = st.columns(num_columns)
# データを一行ずつ処理して表示
for idx, row in sorted_df.iterrows():
    col_idx = idx % num_columns  # 現在の列を計算
    with columns[col_idx]:  # 該当する列に内容を表示
        st.markdown(f"#### {row['市区町丁']}")
        st.markdown(f"- **住環境スコア**：{row['env_score_normal']}（{total_entries} 件中 {idx + 1} 位）")
        streetlights = int(row['街灯の数']) if not pd.isna(row['街灯の数']) else 'データ無し'
        st.markdown(f"- **街灯の数**：{streetlights}")
        st.markdown(f"- **騒音の平均値(dB)**：{row['騒音の平均値'] if not pd.isna(row['騒音の平均値']) else 'データ無し'}")
        evacuation_centers = int(row['避難所の数']) if not pd.isna(row['避難所の数']) else 'データ無し'
        st.markdown(f"- **避難所の数**：{evacuation_centers}")
        st.markdown("---")

st.dataframe(df_hanzai)
st.dataframe(df_syousai)
###変更後
# # df_syousai を df_hanzai から必要な情報を取得するようにマージ
# df_syousai = df_score[df_score['市区町丁'].isin(selected_cho)]
# # df_hanzai からの情報を取得
# df_syousai = df_syousai.merge(df_hanzai[['市区町丁', '街灯の数', '騒音の平均値', '避難所の数']], on='市区町丁', how='left')
# st.dataframe(df_syousai.style.highlight_max(axis=0))
# # 住環境スコアの降順でソート
# sorted_df = df_syousai.sort_values(by='env_score_normal', ascending=False).reset_index(drop=True)
# # 選択件数を取得
# total_entries = len(sorted_df)
# # 詳細表示
# num_columns = 4
# columns = st.columns(num_columns)
# # データを一行ずつ処理して表示
# for idx, row in sorted_df.iterrows():
#     col_idx = idx % num_columns  # 現在の列を計算
#     with columns[col_idx]:  # 該当する列に内容を表示
#         st.markdown(f"#### {row['市区町丁']}")
#         st.markdown(f"- **住環境スコア**：{row['env_score_normal']}（{total_entries} 件中 {idx + 1} 位）")
#         # 街灯の数を整数に変換
#         streetlights = int(row['街灯の数']) if '街灯の数' in row and not pd.isna(row['街灯の数']) else 'データ無し'
#         st.markdown(f"- **街灯の数**：{streetlights}")
#         # 騒音の平均値を表示
#         noise_avg = row['騒音の平均値'] if '騒音の平均値' in row and not pd.isna(row['騒音の平均値']) else 'データ無し'
#         st.markdown(f"- **騒音の平均値(dB)**：{noise_avg}")
#         # 避難所の数を整数に変換
#         evacuation_centers = int(row['避難所の数']) if '避難所の数' in row and not pd.isna(row['避難所の数']) else 'データ無し'
#         st.markdown(f"- **避難所の数**：{evacuation_centers}")
#         st.markdown("---")


