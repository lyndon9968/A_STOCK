import os
import sys
import json
import torch
# 强制移除可能导致 GitHub Actions 下载失败的本地代理变量
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)
os.environ.pop("http_proxy", None)
os.environ.pop("https_proxy", None)

import akshare as ak

# --- 1. 硬编码沪深 300 成分股列表 ---
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
        "603833", "603233", "601231", "603198", "601611", "000001", "601668", "300059", "603288", "600690"
    ]

# ---2. 初始化预测引擎 ---
print("正在初始化 Kronos 预测引擎...")
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-mini")
# GitHub Actions 通常没有 GPU，使用 cpu 运行模型即可
predictor = KronosPredictor(model, tokenizer, device="cpu", max_context=512)

try:
    from model import Kronos, KronosTokenizer, KronosPredictor
except ModuleNotFoundError:
    print("❌ 找不到 Kronos 的 'model' 模块！请确保 Kronos 的源码存在于当前目录或环境变量中。")
    # 不要使用 input()，直接用 sys.exit 退出，适配 CI/CD 自动化环境
    sys.exit(1)

def main():
    # --- 4. 扫描配置 ---
    stock_pool = get_hardcoded_pool()
    num_samples = 30  # 30次采样
    pred_len = 10     
    save_path = "latest_scan.json"
    
    print(f"🚀 启动 GitHub Actions 云端扫描任务！目标: {len(stock_pool)} 只")
    
    # --- 5. 循环执行并输出 JSON ---
    results = []
    for idx, code in enumerate(stock_pool):
        try:
            # 防限流机制
            time.sleep(0.5) 
            symbol = f"sh{code}" if code.startswith(('6', '5')) else f"sz{code}"
            
            # 抓取日线
            df_raw = ak.stock_zh_a_daily(symbol=symbol)
            if df_raw.empty or len(df_raw) < 100: continue
            
            df = df_raw[['date', 'open', 'high', 'low', 'close']].tail(500).copy()
            df.columns = ['timestamps', 'open', 'high', 'low', 'close']
            df['timestamps'] = pd.to_datetime(df['timestamps'])
            
            latest_date = df['timestamps'].iloc[-1].strftime('%Y-%m-%d')
            current_price = float(df['close'].iloc[-1])
            
            future_ts = pd.date_range(start=df['timestamps'].iloc[-1], periods=pred_len+1, freq='D')[1:]
            x_time = pd.Series(df['timestamps']) + pd.to_timedelta(range(len(df)), unit='s')
            y_time = pd.Series(future_ts) + pd.to_timedelta(range(len(future_ts)), unit='s')
            
            # 30 次蒙特卡洛预测
            samples = [predictor.predict(df=df[['open', 'high', 'low', 'close']].astype('float32'), 
                                        x_timestamp=x_time, y_timestamp=y_time, 
                                        pred_len=pred_len)['close'].values[-1] for _ in range(num_samples)]
            
            avg_future_price = float(np.mean(samples))
            ret_pct = float(round((avg_future_price - current_price) / current_price * 100, 2))
            
            # 将结果格式化为前端界面匹配的对象
            results.append({
                "code": code,
                "latestDate": latest_date,
                "currentPrice": round(current_price, 2),
                "predictedPrice": round(avg_future_price, 2),
                "retPct": ret_pct,
                "status": "success"
            })
            
            print(f"[{idx+1}/{len(stock_pool)}] {code} | 日期:{latest_date} | 现价:¥{current_price:.2f} | 目标:¥{avg_future_price:.2f} | 涨幅:{ret_pct}%")
            
        except Exception as e:
            print(f"[{idx+1}/{len(stock_pool)}] 代码 {code} 报错: {e}")
            continue
    
    # 排序并提取 Top，保存给大屏直接用
    if results:
        results = sorted(results, key=lambda x: x['retPct'], reverse=True)
    
    # 写入前端所需的 JSON 数据格式
    output_data = {
        "lastUpdated": datetime.now(timezone.utc).isoformat(),
        "results": results
    }
    
    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    print(f"\n全部计算资源处理完毕，文件已落盘至: {save_path}")

if __name__ == "__main__":
    main()
