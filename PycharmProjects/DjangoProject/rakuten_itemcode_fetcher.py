import requests
from bs4 import BeautifulSoup
import time
import random
import win32com.client

# ===========================
# 楽天ページから商品番号を取得する関数
# ===========================
def get_item_number(url, retries=3, wait=5):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    for i in range(retries):
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, "html.parser")

            elem = soup.find("span", class_="normal_reserve_item_number")
            if elem:
                return elem.text.strip()

            return None

        except Exception as e:
            print(f"[{i+1}回目失敗] {url} → {e}")
            time.sleep(wait + random.uniform(0, 3))

    return None

# ===========================
# Excel操作準備
# ===========================
EXCEL_PATH = r"C:\Users\DT2008-3\PycharmProjects\DjangoProject\26SS NINT 商品分析 GREGORY.xlsx"
START_ROW = 16
G_COLUMN = 7    # G列 (URL)
AE_COLUMN = 31 # AE列 (結果出力)

excel = win32com.client.Dispatch("Excel.Application")
excel.Visible = True  # Excelを開いたまま確認
wb = excel.Workbooks.Open(EXCEL_PATH)
ws = wb.Sheets("情報元 業種分析")

last_row = ws.Cells(ws.Rows.Count, G_COLUMN).End(-4162).Row  # G列の最終行
print(f"処理対象: {START_ROW}行目～{last_row}行目")

# ===========================
# URLループ処理
# ===========================
for row in range(START_ROW, last_row + 1):
    url = ws.Cells(row, G_COLUMN).Value
    if not url:
        continue

    # 文字列化＆空白除去
    url = str(url).strip()
    print(f"[{row}] 処理中: {url}")

    item_no = get_item_number(url)

    if item_no:
        ws.Cells(row, AE_COLUMN).Value = item_no
        print(f" → 取得成功: {item_no}")
    else:
        ws.Cells(row, AE_COLUMN).Value = "取得できず"
        print(" → 取得失敗")

    # 楽天サーバーに優しく 2秒待機
    time.sleep(2)

# ===========================
# 完了メッセージ
# ===========================
print("=== 処理完了 ===")
print("※上書き保存は不要なら、Excelを手動で保存してください。")
