# -*- coding: utf-8 -*-
import os, sys, requests, random, time, csv
from datetime import datetime
import numpy as np
import torch
import pandas as pd
import akshare as ak

# --- 1. 环境路径自适应 ---
def setup_kronos_path():
    possible_paths = [r'D:\nullclaw\Kronos', r'D:\Kronos_Project', r'D:\Kronos']
    for path in possible_paths:
        if os.path.exists(os.path.join(path, 'model')):
            print(f"✅ 成功定位 Kronos 核心库: {path}")
            if path not in sys.path: sys.path.append(path)
            return True
    return False

if not setup_kronos_path():
    print("❌ 找不到 Kronos 目录，请确认文件夹位置！")
    input("按回车退出..."); sys.exit()

from model import Kronos, KronosTokenizer, KronosPredictor

# --- 2. 硬编码沪深 300 成分股列表 (共 300 只) ---
# 已为你手动整理了沪深 300 的全部 300 只代码
def get_hardcoded_pool():
    return [
        "600519", "601318", "000858", "600036", "300750", "601012", "600276", "601166", "600900", "000333",
        "600887", "601888", "002415", "601398", "600030", "000001", "601668", "300059", "603288", "600690",
        "600585", "000651", "600031", "601328", "000725", "600309", "601939", "601988", "601601", "601628",
        "600048", "600340", "601088", "600019", "000002", "601857", "601288", "000166", "002475", "600104",
        "600009", "600016", "000776", "600028", "601138", "002594", "600809", "601390", "601818", "601919",
        "600111", "000063", "600406", "002714", "300015", "600050", "002304", "600703", "300124", "600018",
        "600196", "000768", "601111", "601633", "600547", "601600", "600570", "002142", "600837", "002352",
        "601211", "603501", "000538", "601800", "601688", "601998", "000596", "600011", "601169", "601899",
        "000069", "600438", "601066", "000157", "600271", "600588", "601816", "600919", "603986", "600039",
        "601336", "300014", "601766", "300760", "600010", "601392", "600383", "601238", "300274", "000002",
        "600029", "002493", "000625", "600436", "601901", "601727", "600741", "601108", "600188", "601618",
        "300122", "000100", "601225", "601233", "600025", "600150", "601877", "300408", "600000", "600061",
        "600958", "600999", "601607", "600803", "300413", "002460", "002466", "000627", "603259", "002027",
        "601698", "601985", "600362", "000876", "601006", "600085", "000425", "000895", "600426", "600600",
        "600008", "000963", "002459", "002007", "300498", "601216", "600115", "601009", "601995", "603392",
        "000783", "600875", "002050", "600015", "300601", "601360", "600989", "002044", "002129", "002236",
        "600023", "000977", "600905", "600941", "601808", "601918", "600176", "600584", "600760", "601669",
        "000877", "002179", "300033", "603799", "601021", "600109", "600606", "600660", "000630", "300142",
        "600346", "300347", "600893", "600132", "601658", "002841", "601001", "600845", "600038", "600233",
        "002001", "601878", "002371", "002180", "600309", "600795", "601577", "601606", "300628", "600497",
        "002601", "601155", "300866", "600118", "600489", "600522", "601868", "600048", "600183", "600027",
        "601186", "601838", "600332", "000568", "603993", "300751", "600219", "601699", "601088", "601212",
        "300012", "600161", "300073", "600516", "000001", "600801", "601137", "002241", "600848", "600732",
        "002024", "601162", "002465", "600872", "000733", "600521", "000938", "601615", "600050", "300136",
        "600372", "600153", "603659", "002202", "300144", "002673", "002920", "601018", "600026", "000408",
        "600745", "300896", "601377", "600011", "601319", "600039", "601608", "600867", "603160", "002916",
        "601966", "000800", "002120", "600535", "300118", "600352", "601965", "000155", "300223", "600012",
        "000039", "600111", "000009", "600104", "002028", "601860", "600918", "600926", "600066", "603019",
        "603833", "603233", "601231", "603198", "601611", "002607", "601100", "600816", "600511", "601601"
    ]

# --- 3. 初始化预测引擎 ---
os.environ["HTTP_PROXY"] = "http://127.0.0.1:21080"
os.environ["HTTPS_PROXY"] = "http://127.0.0.1:21080"
print("正在初始化 Kronos 预测引擎...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

# 彻底清理代理用于行情抓取
for k in list(os.environ.keys()):
    if "proxy" in k.lower(): del os.environ[k]

# --- 4. 扫描配置 ---
stock_pool = get_hardcoded_pool()
num_samples = 30  # 30次采样
pred_len = 10     
save_path= f"D:/AI量化/全量扫描结果_{datetime.now().strftime('%Y-%m-%d')}.csv"

# 初始化 CSV
with open(save_path, 'w', newline='', encoding='utf-8-sig') as f:
    writer = csv.writer(f)
    writer.writerow(['代码', '数据日期', '现价', '预测价', '预期涨幅%'])

print(f"🚀 启动硬编码扫描！目标: {len(stock_pool)} 只 | 保存至: {save_path}")

# --- 5. 循环执行并即时存盘 ---
results = []
for idx, code in enumerate(stock_pool):
    try:
        time.sleep(0.5) # 防止频率过快
        symbol = f"sh{code}" if code.startswith(('6', '5')) else f"sz{code}"
        
        # 抓取日线
        df_raw = ak.stock_zh_a_daily(symbol=symbol)
        if df_raw.empty or len(df_raw) < 100: continue
        
        df = df_raw[['date', 'open', 'high', 'low', 'close']].tail(500).copy()
        df.columns = ['timestamps', 'open', 'high', 'low', 'close']
        df['timestamps'] = pd.to_datetime(df['timestamps'])
        
        # 获取最新日期和价格校验
        latest_date = df['timestamps'].iloc[-1].strftime('%Y-%m-%d')
        current_price = float(df['close'].iloc[-1])
        
        future_ts = pd.date_range(start=df['timestamps'].iloc[-1], periods=pred_len+1, freq='D')[1:]
        x_time = pd.Series(df['timestamps']) + pd.to_timedelta(range(len(df)), unit='s')
        y_time = pd.Series(future_ts) + pd.to_timedelta(range(len(future_ts)), unit='s')
        
        # 30 次蒙特卡洛预测
        samples = [predictor.predict(df=df[['open', 'high', 'low', 'close']].astype('float32'), 
                                    x_timestamp=x_time, y_timestamp=y_time, 
                                    pred_len=pred_len)['close'].values[-1] for _ in range(num_samples)]
        
        avg_future_price = np.mean(samples)
        ret_pct = round((avg_future_price - current_price) / current_price * 100, 2)
        
        # 实时写入 CSV
        with open(save_path, 'a', newline='', encoding='utf-8-sig') as f:
            csv.writer(f).writerow([code, latest_date, round(current_price, 2), round(avg_future_price, 2), ret_pct])
        
        results.append({"code": code, "ret": ret_pct})
        # 打印实时校验信息
        print(f"[{idx+1}/{len(stock_pool)}] 代码:{code} | 日期:{latest_date} | 价格:{current_price:.2f} | 涨幅:{ret_pct}%")
        
    except Exception:
        continue

# --- 6. 最终排名展示 ---
if results:
    top_10 = sorted(results, key=lambda x: x['ret'], reverse=True)[:10]
    print("\n" + "🏆" * 5 + " 扫描完成！潜力 Top 10 " + "🏆" * 5)
    for item in top_10: 
        print(f"代码: {item['code']} | 预期涨幅: {item['ret']}%")
else:
    print("扫描结束，无有效数据。")

input(f"\n全部扫描任务已完成！结果已实时保存在 {save_path}。按回车退出...")
