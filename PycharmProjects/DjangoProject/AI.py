import streamlit as st
import pandas as pd
import random
import spacy
from spacy.training import Example
from spacy.util import minibatch, compounding
import re
from pathlib import Path
import shutil
from difflib import SequenceMatcher
import io

# ===============================
# 📂 モデル保存ディレクトリ
# ===============================
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# ===============================
# 📦 ブランド一覧取得関数
# ===============================
def get_brand_list():
    return [p.name for p in MODEL_DIR.iterdir() if (p / "product_code_model").exists()]

# ===============================
# ✨ ランダムテンプレート（品番学習用）
# ===============================
TEMPLATES = [
    "{} スニーカー メンズ レディース 人気モデル",
    "NIKE {} AIR FORCE 1 ナイキ エアフォース ワン",
    "adidas {} スーパースター アディダス 定番モデル",
    "New Balance {} ニューバランス ランニングシューズ",
    "CONVERSE {} オールスター コンバース ハイカット",
    "PUMA {} プーマ スポーツシューズ",
    "Reebok {} リーボック トレーニングシューズ",
    "THE NORTH FACE {} ノースフェイス ジャケット メンズ",
    "UNIQLO {} ユニクロ シャツ 長袖 メンズ",
    "GU {} レディース ワンピース 春 夏 新作",
    "{} Tシャツ 半袖 綿100%",
    "{} バッグ トート レディース ブランド 人気",
    "{} キャップ 帽子 メンズ レディース",
    "型番 {} スニーカー 靴 送料無料",
    "品番 {} デニム パンツ ジーンズ",
    "{} ベルト メンズ ビジネス 本革",
    "商品コード {} リュック 通勤 通学",
]

# ===============================
# 🧠 学習データ生成
# ===============================
def create_training_data(codes, n_variants=10):
    data = []
    for code in codes:
        for _ in range(n_variants):
            template = random.choice(TEMPLATES)
            text = template.format(code)
            for match in re.finditer(re.escape(code), text):
                start, end = match.span()
                data.append((text, {"entities": [(start, end, "PRODUCT_CODE")]}))
    return data

# ===============================
# 🧠 ブランド別モデル学習
# ===============================
def train_model_for_brand(codes, brand_name, continue_training=False):
    brand_path = MODEL_DIR / brand_name / "product_code_model"
    brand_path.parent.mkdir(exist_ok=True, parents=True)

    if continue_training and brand_path.exists():
        nlp = spacy.load(brand_path)
        ner = nlp.get_pipe("ner")
        st.info(f"✅ {brand_name} の既存モデルを読み込んで追加学習")
    else:
        try:
            nlp = spacy.blank("ja")
        except Exception:
            nlp = spacy.blank("xx")
        ner = nlp.add_pipe("ner")
        ner.add_label("PRODUCT_CODE")
        st.info(f"🆕 {brand_name} の新規モデルを作成")

    optimizer = nlp.resume_training() if continue_training else nlp.initialize()
    train_data = create_training_data(codes)

    progress = st.progress(0)
    for epoch in range(5):
        random.shuffle(train_data)
        losses = {}
        batches = minibatch(train_data, size=compounding(4.0, 32.0, 1.5))
        for batch in batches:
            examples = [Example.from_dict(nlp.make_doc(text), ann) for text, ann in batch]
            nlp.update(examples, sgd=optimizer, losses=losses)
        st.write(f"Epoch {epoch+1} Losses: {losses}")
        progress.progress((epoch + 1) / 5)

    nlp.to_disk(brand_path)

    # 品番リスト保存
    codes_file = brand_path / "codes.txt"
    with open(codes_file, "w", encoding="utf-8") as f:
        for code in codes:
            f.write(f"{code}\n")

    st.success(f"🎉 {brand_name} モデルの学習完了！")

    # モデル一覧を即時更新
    st.session_state.brands = get_brand_list()

# ===============================
# 🔍 品番抽出（ハイブリッド）
# ===============================
def extract_codes_with_brand(text, brand_name, similarity_threshold=0.2):
    brand_path = MODEL_DIR / brand_name / "product_code_model"
    if not brand_path.exists():
        st.warning(f"❌ {brand_name} モデルがありません")
        return []

    nlp = spacy.load(brand_path)
    doc = nlp(text)

    codes_path = brand_path / "codes.txt"
    if codes_path.exists():
        with open(codes_path, "r", encoding="utf-8") as f:
            trained_codes = [line.strip() for line in f if line.strip()]
    else:
        trained_codes = []

    results = set()
    # NER抽出結果（3文字以下は無視）
    for ent in doc.ents:
        if (
            ent.label_ == "PRODUCT_CODE"
            and re.fullmatch(r"[A-Za-z0-9]+", ent.text)
            and len(ent.text) > 3
        ):
            results.add(ent.text)

    # 類似度チェック
    for match in re.findall(r"\b[A-Za-z0-9]+\b", text):
        if len(match) <= 3:
            continue
        for code in trained_codes:
            ratio = SequenceMatcher(None, match, code).ratio()
            if ratio >= similarity_threshold:
                results.add(match)

    return sorted(results)

# ===============================
# ⚙️ Streamlit UI
# ===============================
st.title("🧩 ブランド別 品番抽出AI（Excel対応）")
tab1, tab2, tab3 = st.tabs(["📘 モデル学習", "🧠 品番抽出", "🧹 モデル管理"])

# ====== モデル学習タブ ======
with tab1:
    st.header("モデル学習・再学習")
    uploaded = st.file_uploader("📤 品番リストExcelをアップロード（列名『品番』）", type="xlsx", key="train")
    brand_name = st.text_input("ブランド名を入力", key="train_brand")
    if uploaded and brand_name.strip():
        try:
            df = pd.read_excel(uploaded)
        except Exception as e:
            st.error(f"❌ Excel ファイルの読み込みに失敗しました: {e}")
        else:
            if "品番" not in df.columns:
                st.error("❌ Excelに『品番』列がありません。")
            else:
                codes = df["品番"].dropna().astype(str).unique().tolist()
                st.write(f"✅ 読み込んだ品番数: {len(codes)}")
                st.dataframe(df.head())
                mode = st.radio("学習モードを選択", ["新規学習", "追加学習"], horizontal=True)
                if st.button("🚀 モデル学習開始"):
                    train_model_for_brand(
                        codes, brand_name.strip(), continue_training=(mode=="追加学習")
                    )
    else:
        st.info("📂 Excelをアップロードし、ブランド名を入力してください。")

# ====== 品番抽出タブ ======
with tab2:
    st.header("Excel から商品名列を読み込んで品番抽出")
    uploaded_extract = st.file_uploader(
        "📤 Excel ファイルをアップロード（列名『商品名』）",
        type="xlsx",
        key="extract"
    )

    st.subheader("③ 品番抽出")

    # 既存モデル一覧を取得
    if "brands" not in st.session_state:
        st.session_state.brands = get_brand_list()

    brand_name = st.selectbox(
        "抽出に使うブランド名を選択",
        st.session_state.brands if st.session_state.brands else ["モデルがまだありません"]
    )

    similarity_threshold = st.slider(
        "類似度閾値を設定（0に近いほどゆるく抽出）",
        0.0, 1.0, 0.2, 0.05
    )

    if uploaded_extract and brand_name and brand_name != "モデルがまだありません":
        try:
            df_extract = pd.read_excel(uploaded_extract)
        except Exception as e:
            st.error(f"❌ Excel ファイルの読み込みに失敗しました: {e}")
        else:
            if "商品名" not in df_extract.columns:
                st.error("❌ Excelに『商品名』列がありません。")
            else:
                if st.button("🔍 抽出実行"):
                    extracted_list = []
                    for text in df_extract["商品名"]:
                        codes = extract_codes_with_brand(
                            str(text),
                            brand_name.strip(),
                            similarity_threshold
                        )
                        extracted_list.append(", ".join(codes))
                    df_extract["抽出品番"] = extracted_list
                    st.success(f"✅ 品番抽出完了！ 類似度閾値={similarity_threshold}")
                    st.dataframe(df_extract)

                    output = io.BytesIO()
                    df_extract.to_excel(output, index=False)
                    st.download_button(
                        "💾 抽出結果をダウンロード",
                        data=output.getvalue(),
                        file_name="抽出結果.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
    else:
        st.info("📂 Excelをアップロードし、ブランド名を選択してください。")

# ====== モデル管理タブ ======
if "brands" not in st.session_state:
    st.session_state.brands = get_brand_list()

with tab3:
    st.header("ブランド別モデル管理")
    st.write("📦 存在するブランドモデル一覧:")
    st.write(", ".join(st.session_state.brands) if st.session_state.brands else "（まだありません）")

    brand_name_manage = st.text_input("管理対象ブランド名を入力", key="manage_brand")
    if brand_name_manage.strip():
        brand_path = MODEL_DIR / brand_name_manage.strip() / "product_code_model"
        if brand_path.exists():
            st.success(f"✅ {brand_name_manage.strip()} のモデルが存在します")
            if st.button(f"🗑️ {brand_name_manage.strip()} モデル削除（初期化）", key="delete"):
                shutil.rmtree(brand_path.parent)
                st.warning("モデルを削除しました。")
                st.session_state.brands = get_brand_list()
        else:
            st.info(f"ℹ️ {brand_name_manage.strip()} モデルはまだ存在しません")
