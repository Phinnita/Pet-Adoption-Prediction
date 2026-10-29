"""
Pet Adoption Prediction : ทำนายโอกาสที่สัตว์เลี้ยงจะถูกรับเลี้ยง
Dataset : PetFinder.my (mini) - สุนัขและแมวในมาเลเซีย
Models  : Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting
"""

import io
import time
import urllib.request
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, plot_tree

# =====================================================================
# ข้อมูลผู้พัฒนา (แก้ตรงนี้)
# =====================================================================
DEVELOPER = {
    "name": "นางสาวภิญนิตา ซิ่วมณี",
    "student_id": "664245042",
    "section": "หมู่เรียน 66/44",
    "photo": "profile.jpg",
}

# =====================================================================
# ลิงก์ผลงาน (วางลิงก์ของตัวเองในเครื่องหมายคำพูด)
# =====================================================================
PROJECT_LINKS = {
    "youtube": "https://youtu.be/YeIjViR0vwg",   # ลิงก์วิดีโอ YouTube
    "colab": "https://colab.research.google.com/drive/1Tw8yeoepe2v7u75fSAbNOO92xQxaUw3m?usp=sharing",     # ลิงก์ Google Colab (Share > Copy link)
    "slides": "Pet_Adoption_Presentation.pptx",    # ลิงก์สไลด์ออนไลน์ เช่น Google Slides / Canva (ไม่บังคับ)
    "github": "https://github.com/Phinnita/Pet-Adoption-Prediction",
}

BASE_DIR = Path(__file__).parent
DATA_URL = "https://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip"
DATA_FILE = "petfinder-mini.csv"
TARGET = "Adopted"
CLASS_NAMES = ["ไม่ถูกรับเลี้ยง", "ถูกรับเลี้ยง"]
CLASS_NAMES_EN = ["Not adopted", "Adopted"]

COLUMN_INFO = {
    "Type": "ชนิดสัตว์ (Dog / Cat)",
    "Age": "อายุ (เดือน)",
    "Breed1": "สายพันธุ์หลัก",
    "Gender": "เพศ",
    "Color1": "สีหลัก",
    "Color2": "สีรอง",
    "MaturitySize": "ขนาดเมื่อโตเต็มที่ (Small / Medium / Large)",
    "FurLength": "ความยาวขน (Short / Medium / Long)",
    "Vaccinated": "ฉีดวัคซีนแล้วหรือไม่",
    "Sterilized": "ทำหมันแล้วหรือไม่",
    "Health": "สุขภาพ (Healthy / Minor Injury / Serious Injury)",
    "Fee": "ค่าธรรมเนียมรับเลี้ยง (ริงกิต, 0 = ฟรี)",
    "Description": "ข้อความบรรยายในประกาศ",
    "PhotoAmt": "จำนวนรูปในประกาศ",
    "AdoptionSpeed": "ความเร็วในการถูกรับเลี้ยง 0-4 (4 = ไม่ถูกรับเลี้ยงภายใน 100 วัน)",
}
SPEED_INFO = {0: "วันเดียวกับที่ลงประกาศ", 1: "1-7 วัน", 2: "8-30 วัน", 3: "31-90 วัน",
              4: "ไม่ถูกรับเลี้ยงภายใน 100 วัน"}

ORDINAL_MAPS = {
    "MaturitySize": {"Small": 1, "Medium": 2, "Large": 3, "Extra Large": 4},
    "FurLength": {"Short": 1, "Medium": 2, "Long": 3},
    "Health": {"Healthy": 0, "Minor Injury": 1, "Serious Injury": 2},
}
NUM_COLS = ["Age", "Fee", "PhotoAmt", "DescLength", "MaturitySize", "FurLength", "Health",
            "IsDog", "IsMale", "IsMixedBreed", "HasColor2", "IsFree"]
CAT_COLS = ["Color1", "Vaccinated", "Sterilized", "BreedGroup"]
CLIP_COLS = ["Age", "Fee", "PhotoAmt", "DescLength"]
TOP_BREEDS = 10

MENU = [
    "หน้าแรก / ผู้พัฒนา",
    "1. ปัญหาและ Dataset",
    "2. Data Preprocessing",
    "3. ทฤษฎีโมเดล ML",
    "4. ประเมินและเปรียบเทียบโมเดล",
    "5. Streamlit Application",
    "ไฟล์ประกอบ: Colab / YouTube / PowerPoint",
]

st.set_page_config(page_title="Pet Adoption Prediction", layout="wide")


# =====================================================================
# Utilities
# =====================================================================
def find_file(name, folders=("", "data", "images")):
    for f in folders:
        p = BASE_DIR / f / name
        if p.exists():
            return p
    return None


def show_photo(container, width):
    path = find_file(DEVELOPER["photo"])
    if path is None:
        container.warning(f"ไม่พบไฟล์รูป {DEVELOPER['photo']}")
        return
    try:
        from PIL import Image, ImageOps
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
        side = min(img.size)
        left, top = (img.width - side) // 2, (img.height - side) // 2
        container.image(img.crop((left, top, left + side, top + side)), width=width)
    except Exception as e:
        container.warning(f"เปิดรูปไม่ได้: {e}")


def fig_show(fig):
    st.pyplot(fig, clear_figure=True)
    plt.close(fig)


@st.cache_data(show_spinner="กำลังโหลดข้อมูล PetFinder ...")
def load_raw():
    path = find_file(DATA_FILE)
    if path is None:
        raw = urllib.request.urlopen(DATA_URL, timeout=60).read()
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            name = [n for n in z.namelist() if n.endswith(".csv")][0]
            data = z.read(name)
        path = BASE_DIR / DATA_FILE
        try:
            path.write_bytes(data)
        except OSError:
            return pd.read_csv(io.BytesIO(data))
    return pd.read_csv(path)


def engineer(raw, top_breeds):
    """Feature engineering: แปลงข้อมูลดิบเป็น features ที่โมเดลใช้ได้"""
    d = pd.DataFrame(index=raw.index)
    d["Age"] = raw["Age"].astype(float)
    d["Fee"] = raw["Fee"].astype(float)
    d["PhotoAmt"] = raw["PhotoAmt"].astype(float)
    d["DescLength"] = raw["Description"].fillna("").astype(str).str.split().str.len().astype(float)
    for col, mp in ORDINAL_MAPS.items():
        d[col] = raw[col].map(mp).fillna(0).astype(float)
    d["IsDog"] = (raw["Type"] == "Dog").astype(int)
    d["IsMale"] = (raw["Gender"] == "Male").astype(int)
    d["IsMixedBreed"] = raw["Breed1"].astype(str).str.contains("Mixed", case=False).astype(int)
    d["HasColor2"] = (raw["Color2"].astype(str) != "No Color").astype(int)
    d["IsFree"] = (raw["Fee"] == 0).astype(int)
    d["Color1"] = raw["Color1"].astype(str)
    d["Vaccinated"] = raw["Vaccinated"].astype(str)
    d["Sterilized"] = raw["Sterilized"].astype(str)
    d["BreedGroup"] = raw["Breed1"].astype(str).where(raw["Breed1"].isin(top_breeds), "Other")
    return d


@st.cache_data(show_spinner=False, max_entries=2)
def prepare(clip):
    raw = load_raw()
    raw = raw.drop_duplicates().reset_index(drop=True)
    top_breeds = raw["Breed1"].value_counts().head(TOP_BREEDS).index.tolist()
    X = engineer(raw, top_breeds)
    bounds = {}
    if clip:
        for c in CLIP_COLS:
            lo, hi = X[c].quantile(0.01), X[c].quantile(0.99)
            bounds[c] = (lo, hi)
            X[c] = X[c].clip(lo, hi)
    y = (raw["AdoptionSpeed"] < 4).astype(int)
    return raw, X, y, top_breeds, bounds


def make_preprocessor():
    return ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
    ])


def make_model(name, params=None, balanced=False, rs=42):
    p = params or {}
    cw = "balanced" if balanced else None
    if name == "Logistic Regression":
        est = LogisticRegression(C=p.get("C", 1.0), max_iter=2000, class_weight=cw)
    elif name == "KNN":
        est = KNeighborsClassifier(n_neighbors=p.get("k", 15), weights=p.get("weights", "uniform"))
    elif name == "Decision Tree":
        est = DecisionTreeClassifier(max_depth=p.get("max_depth", 6), min_samples_leaf=p.get("min_leaf", 20),
                                     criterion=p.get("criterion", "gini"), class_weight=cw, random_state=rs)
    elif name == "Random Forest":
        est = RandomForestClassifier(n_estimators=p.get("n_estimators", 200), max_depth=p.get("max_depth", 12),
                                     min_samples_leaf=p.get("min_leaf", 2), class_weight=cw,
                                     n_jobs=2, random_state=rs)
    elif name == "Gradient Boosting":
        est = HistGradientBoostingClassifier(learning_rate=p.get("lr", 0.1), max_iter=p.get("n_iter", 200),
                                             max_depth=p.get("max_depth", 6), class_weight=cw, random_state=rs)
    else:
        est = DummyClassifier(strategy="most_frequent")
    return Pipeline([("prep", make_preprocessor()), ("model", est)])


MODEL_NAMES = ["Logistic Regression", "KNN", "Decision Tree", "Random Forest", "Gradient Boosting"]


# =====================================================================
# Load data + sidebar
# =====================================================================
try:
    _probe = load_raw()
except Exception as e:
    st.error(f"โหลดข้อมูลไม่สำเร็จ: {e}")
    st.info("รัน `python download_data.py` ในโฟลเดอร์โปรเจกต์ เพื่อดาวน์โหลดไฟล์ petfinder-mini.csv ก่อน")
    st.stop()

with st.sidebar:
    st.title("Pet Adoption ML")
    st.caption("ทำนายโอกาสที่สัตว์เลี้ยงจะได้บ้านใหม่")
    page = st.radio("เลือกหัวข้อ", MENU)
    st.divider()
    st.subheader("ตั้งค่า")
    clip_outliers = st.checkbox("Clip outlier (percentile 1-99)", value=True)
    balanced = st.checkbox("ถ่วงน้ำหนักคลาส (class_weight = balanced)", value=False,
                           help="ช่วยให้โมเดลสนใจคลาสส่วนน้อย (ไม่ถูกรับเลี้ยง) มากขึ้น")
    test_size = st.slider("สัดส่วน Test set", 0.1, 0.4, 0.2, step=0.05)
    random_state = int(st.number_input("Random state", 0, 9999, 42))
    st.divider()
    st.subheader("ผู้พัฒนา")
    show_photo(st.sidebar, 120)
    st.markdown(f"**{DEVELOPER['name']}**  \nรหัสนักศึกษา: `{DEVELOPER['student_id']}`  \n{DEVELOPER['section']}")

raw, X, y, top_breeds, clip_bounds = prepare(bool(clip_outliers))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


# =====================================================================
# Pages
# =====================================================================
def page_home():
    st.title("Pet Adoption Prediction")
    st.subheader("ทำนายโอกาสที่สุนัขและแมวจะถูกรับเลี้ยงด้วย Machine Learning")
    st.write("เว็บแอปพลิเคชันสร้างและเปรียบเทียบโมเดล Machine Learning 5 แบบ ด้วยข้อมูลจริงจาก PetFinder.my "
             "พัฒนาด้วย **Python + scikit-learn + Streamlit**")
    st.header("ข้อมูลผู้พัฒนา")
    c1, c2 = st.columns([1, 3])
    show_photo(c1, 220)
    c2.markdown(
        f"""
| รายการ | ข้อมูล |
|---|---|
| **ชื่อ-นามสกุล** | {DEVELOPER['name']} |
| **รหัสนักศึกษา** | {DEVELOPER['student_id']} |
| **หมู่เรียน** | {DEVELOPER['section']} |
"""
    )
    st.header("หัวข้อในเว็บไซต์")
    st.table(pd.DataFrame({
        "หัวข้อ": ["การกำหนดปัญหาและ Dataset", "Data Preprocessing", "การสร้างโมเดล ML (ทฤษฎี)",
                   "การประเมินและเปรียบเทียบโมเดล", "Streamlit Application", "ไฟล์ประกอบ"],
        "เนื้อหา": [
            "ปัญหาที่ต้องการแก้ เหตุผลที่เลือกข้อมูล และการสำรวจข้อมูล (EDA)",
            "ตรวจข้อมูล, Feature Engineering, จัดการ Outlier, Encoding, Scaling, แบ่ง Train/Test",
            "แนวคิดและสมการของ Logistic Regression, KNN, Decision Tree, Random Forest, Gradient Boosting",
            "ตารางและกราฟเปรียบเทียบ Accuracy, Precision, Recall, F1, ROC-AUC และ Cross-validation",
            "กรอกข้อมูลสัตว์เลี้ยงเพื่อทำนายโอกาสถูกรับเลี้ยง และทดลองปรับโมเดล",
            "วิดีโอ YouTube, โค้ด Google Colab, สไลด์ PowerPoint และ GitHub",
        ],
    }, index=[1, 2, 3, 4, 5, "+"]))
    c = st.columns(4)
    c[0].metric("จำนวนสัตว์เลี้ยง", f"{len(raw):,}")
    c[1].metric("สุนัข", f"{(raw['Type'] == 'Dog').sum():,}")
    c[2].metric("แมว", f"{(raw['Type'] == 'Cat').sum():,}")
    c[3].metric("ถูกรับเลี้ยงภายใน 100 วัน", f"{y.mean():.1%}")


def rate_chart(ax, series, title, order=None):
    rate = y.groupby(series, observed=True).mean()
    if order is not None:
        rate = rate.reindex([o for o in order if o in rate.index])
    ax.bar([str(i) for i in rate.index], rate.values, color="#e07a5f")
    ax.axhline(y.mean(), color="k", ls="--", lw=1)
    ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="x", labelsize=8, rotation=30)


def page_problem():
    st.title("1. การกำหนดปัญหาและ Dataset")
    st.header("1.1 การกำหนดปัญหา (Problem Definition)")
    st.write(
        "สัตว์เลี้ยงจรจัดจำนวนมากรอคนรับเลี้ยงอยู่ในศูนย์พักพิง ยิ่งอยู่นานยิ่งใช้ทรัพยากรมาก "
        "และบางตัวไม่เคยได้บ้านเลย ถ้ารู้ล่วงหน้าว่าตัวไหนมีโอกาสถูกรับเลี้ยงต่ำ "
        "ศูนย์พักพิงจะช่วยได้ตรงจุด เช่น ถ่ายรูปเพิ่ม เขียนประกาศให้ละเอียดขึ้น ฉีดวัคซีน หรือลดค่าธรรมเนียม"
    )
    c1, c2 = st.columns(2)
    c1.info("**คำถาม:** สัตว์เลี้ยงตัวนี้จะถูกรับเลี้ยงภายใน 100 วันหรือไม่  \n"
            "**ประเภทงาน:** Binary Classification (Supervised Learning)")
    c2.info("**Target:** `Adopted` = 1 ถ้า AdoptionSpeed เป็น 0-3 (ได้บ้านภายใน 100 วัน)  \n"
            "`Adopted` = 0 ถ้า AdoptionSpeed = 4 (ไม่ได้บ้านภายใน 100 วัน)")

    st.header("1.2 ทำไมเลือกชุดข้อมูลนี้")
    reasons = [
        ("ข้อมูลจริง", "มาจากประกาศหาบ้านจริงบนเว็บ PetFinder.my ประเทศมาเลเซีย"),
        ("มีประโยชน์ต่อสังคม", "ช่วยให้ศูนย์พักพิงรู้ว่าปัจจัยใดทำให้สัตว์ได้บ้านเร็วขึ้น"),
        ("ขนาดเหมาะสม", f"{len(raw):,} ตัว มากพอให้โมเดลเรียนรู้ และฝึกบนเว็บได้เร็ว"),
        ("มีข้อมูลหลายชนิด", "มีทั้งตัวเลข หมวดหมู่ และข้อความ จึงได้ฝึก Preprocessing ครบทุกแบบ"),
        ("ตีความผลได้ง่าย", "ปัจจัยอย่างอายุ จำนวนรูป หรือค่าธรรมเนียม เข้าใจได้โดยไม่ต้องเป็นผู้เชี่ยวชาญ"),
        ("เป็นที่รู้จัก", "ใช้ในการแข่งขัน Kaggle PetFinder Adoption Prediction และบทเรียนของ TensorFlow"),
    ]
    cols = st.columns(3)
    for i, (h, d) in enumerate(reasons):
        cols[i % 3].success(f"**{h}**  \n{d}")

    st.header("1.3 รายละเอียดชุดข้อมูล")
    st.write(f"**{len(raw):,} แถว × {raw.shape[1]} คอลัมน์** (1 แถว = สัตว์ 1 ตัว)  \n"
             "แหล่งที่มา: PetFinder.my Adoption Prediction (Kaggle) ฉบับย่อที่ TensorFlow เผยแพร่ "
             f"({DATA_URL})")
    st.dataframe(pd.DataFrame({
        "คอลัมน์": list(COLUMN_INFO),
        "ความหมาย": list(COLUMN_INFO.values()),
        "ชนิด": [str(raw[c].dtype) if c in raw else "-" for c in COLUMN_INFO],
        "ตัวอย่าง": [str(raw[c].iloc[0])[:40] if c in raw else "-" for c in COLUMN_INFO],
    }), hide_index=True, width="stretch")

    st.header("1.4 สำรวจข้อมูล (EDA)")
    t1, t2, t3, t4 = st.tabs(["ตัวอย่างข้อมูล", "Target", "ปัจจัยที่มีผล", "ตัวเลข"])
    with t1:
        st.dataframe(raw.head(30), width="stretch")
    with t2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sp = raw["AdoptionSpeed"].value_counts().sort_index()
            ax.bar(sp.index.astype(str), sp.values, color=["#81b29a"] * 4 + ["#e07a5f"])
            ax.set_xlabel("AdoptionSpeed")
            ax.set_ylabel("Count")
            ax.set_title("AdoptionSpeed (0 = fastest, 4 = not adopted)")
            fig.tight_layout()
            fig_show(fig)
            st.dataframe(pd.DataFrame({"ความหมาย": SPEED_INFO, "จำนวน": sp}), width="stretch")
        with c2:
            fig, ax = plt.subplots(figsize=(5, 3.5))
            cnt = y.value_counts().sort_index()
            ax.pie(cnt.values, labels=CLASS_NAMES_EN, autopct="%1.1f%%", colors=["#e07a5f", "#81b29a"],
                   startangle=90)
            ax.set_title("Target: Adopted within 100 days")
            fig_show(fig)
            st.caption("ข้อมูลไม่สมดุล (Imbalanced) ถูกรับเลี้ยงมากกว่าไม่ถูกรับเลี้ยง")
    with t3:
        st.caption("แท่ง = สัดส่วนที่ถูกรับเลี้ยงในแต่ละกลุ่ม, เส้นประ = ค่าเฉลี่ยรวม")
        fig, axes = plt.subplots(2, 3, figsize=(13, 7))
        rate_chart(axes[0, 0], raw["Type"], "Type")
        rate_chart(axes[0, 1], pd.cut(raw["Age"], [-1, 3, 6, 12, 36, 300],
                                      labels=["0-3", "4-6", "7-12", "13-36", "37+"]), "Age (months)")
        rate_chart(axes[0, 2], pd.cut(raw["PhotoAmt"], [-1, 0, 2, 5, 100], labels=["0", "1-2", "3-5", "6+"]),
                   "Number of photos")
        rate_chart(axes[1, 0], raw["Vaccinated"], "Vaccinated")
        rate_chart(axes[1, 1], raw["MaturitySize"], "MaturitySize", ["Small", "Medium", "Large", "Extra Large"])
        rate_chart(axes[1, 2], (raw["Fee"] == 0).map({True: "Free", False: "Has fee"}), "Fee")
        fig.tight_layout()
        fig_show(fig)
    with t4:
        st.dataframe(raw.describe().T.round(2), width="stretch")
        col = st.selectbox("Histogram", ["Age", "Fee", "PhotoAmt"])
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.hist(raw[col], bins=50, color="#3d405b")
        ax.set_xlabel(col)
        fig.tight_layout()
        fig_show(fig)


def page_preprocess():
    st.title("2. Data Preprocessing")
    st.write("ขั้นตอนเตรียมข้อมูลก่อนเข้าโมเดล ทุกตารางและกราฟในหน้านี้คำนวณจากข้อมูลจริง")
    st.table(pd.DataFrame({
        "ขั้นตอน": ["ตรวจ Missing values", "ตัดข้อมูลซ้ำ", "สร้าง Target", "Feature Engineering",
                    "จัดการ Outlier", "Encoding", "Feature Scaling", "แบ่ง Train/Test", "จัดการ Imbalanced"],
        "สิ่งที่ทำ": [
            "Description มีค่าว่างบางแถว แทนด้วยข้อความว่าง",
            "ใช้ drop_duplicates()",
            "AdoptionSpeed 0-3 → 1 (ถูกรับเลี้ยง), 4 → 0",
            "นับคำใน Description, จัดกลุ่มสายพันธุ์, สร้างตัวแปร 0/1",
            "Clip Age, Fee, PhotoAmt, DescLength ที่ percentile 1-99",
            "Ordinal encoding (ขนาด ขน สุขภาพ) และ One-Hot encoding (สี วัคซีน ทำหมัน พันธุ์)",
            "StandardScaler ให้ทุก feature มี mean 0, std 1",
            f"Stratified split, Test {test_size:.0%}",
            "เลือก class_weight = balanced ได้จากแถบด้านซ้าย",
        ],
    }, index=range(1, 10)))

    st.header("2.1 Missing values และข้อมูลซ้ำ")
    full = load_raw()
    c1, c2 = st.columns([2, 1])
    c1.dataframe(pd.DataFrame({"Missing": full.isnull().sum(), "ชนิดข้อมูล": full.dtypes.astype(str),
                               "ค่าไม่ซ้ำ": full.nunique()}), width="stretch")
    c2.metric("Missing ทั้งหมด", int(full.isnull().sum().sum()))
    c2.metric("แถวซ้ำที่ตัดออก", int(full.duplicated().sum()))
    c2.metric("เหลือข้อมูล", f"{len(raw):,} แถว")

    st.header("2.2 Feature Engineering")
    st.dataframe(pd.DataFrame({
        "Feature ใหม่": ["DescLength", "IsDog", "IsMale", "IsMixedBreed", "HasColor2", "IsFree", "BreedGroup",
                         "MaturitySize / FurLength / Health"],
        "สร้างจาก": ["Description", "Type", "Gender", "Breed1", "Color2", "Fee", "Breed1", "ข้อความเดิม"],
        "วิธีการ": ["นับจำนวนคำ (ข้อความเปลี่ยนเป็นตัวเลข)", "Dog = 1, Cat = 0", "Male = 1, Female = 0",
                    "พันธุ์ผสม = 1", "มีสีรอง = 1", "ค่าธรรมเนียม 0 = 1",
                    f"เก็บ {TOP_BREEDS} พันธุ์ที่พบบ่อย ที่เหลือเป็น Other (ลดจำนวนหมวดหมู่ {raw['Breed1'].nunique()} → {TOP_BREEDS + 1})",
                    "แปลงเป็นลำดับ เช่น Small=1, Medium=2, Large=3"],
    }), hide_index=True, width="stretch")
    st.markdown("**ตัวอย่างก่อนและหลัง Feature Engineering**")
    c1, c2 = st.columns(2)
    c1.dataframe(raw[["Type", "Age", "Breed1", "Gender", "Color2", "MaturitySize", "Fee"]].head(6), width="stretch")
    c2.dataframe(X[["IsDog", "Age", "BreedGroup", "IsMixedBreed", "IsMale", "HasColor2", "MaturitySize", "IsFree"]]
                 .head(6), width="stretch")

    st.header("2.3 จัดการ Outlier (Clipping)")
    _, X_noclip, _, _, _ = prepare(False)
    rows = []
    for c in CLIP_COLS:
        lo, hi = X_noclip[c].quantile(0.01), X_noclip[c].quantile(0.99)
        rows.append({"Feature": c, "Min เดิม": X_noclip[c].min(), "P1": lo, "P99": hi, "Max เดิม": X_noclip[c].max(),
                     "จำนวนที่ถูก clip": int(((X_noclip[c] < lo) | (X_noclip[c] > hi)).sum())})
    st.dataframe(pd.DataFrame(rows).set_index("Feature").round(2), width="stretch")
    col = st.selectbox("ดู Boxplot ก่อน / หลัง", CLIP_COLS)
    fig, axes = plt.subplots(1, 2, figsize=(10, 2.8))
    axes[0].boxplot(X_noclip[col], vert=False)
    axes[0].set_title(f"Before: {col}")
    lo, hi = X_noclip[col].quantile(0.01), X_noclip[col].quantile(0.99)
    axes[1].boxplot(X_noclip[col].clip(lo, hi), vert=False)
    axes[1].set_title(f"After clipping: {col}")
    fig.tight_layout()
    fig_show(fig)
    if not clip_outliers:
        st.warning("ตอนนี้ปิดการ Clip outlier อยู่ (เปิดได้ที่แถบด้านซ้าย)")

    st.header("2.4 Encoding และ Feature Scaling")
    st.latex(r"z = \frac{x - \mu}{\sigma}")
    prep = make_preprocessor().fit(X_train)
    names = prep.get_feature_names_out()
    Xt = pd.DataFrame(prep.transform(X_train), columns=names)
    c1, c2 = st.columns(2)
    c1.metric("จำนวน features ก่อน Encoding", X.shape[1])
    c2.metric("จำนวน features หลัง One-Hot", Xt.shape[1])
    st.write("ใช้ `ColumnTransformer` รวม StandardScaler (ตัวเลข) และ OneHotEncoder (หมวดหมู่) ไว้ใน Pipeline "
             "จึง fit กับ Train set เท่านั้น ป้องกัน Data leakage")
    st.dataframe(Xt.head(8).round(3), width="stretch")
    st.dataframe(Xt[[n for n in names if n.startswith("num__")]].agg(["mean", "std"]).T.round(3).T, width="stretch")

    st.header("2.5 แบ่ง Train / Test และสัดส่วนคลาส")
    c = st.columns(3)
    c[0].metric("Train", f"{len(X_train):,}")
    c[1].metric("Test", f"{len(X_test):,}")
    c[2].metric("สัดส่วนถูกรับเลี้ยง (Train / Test)", f"{y_train.mean():.1%} / {y_test.mean():.1%}")
    st.caption(f"ข้อมูลไม่สมดุล: ถูกรับเลี้ยง {y.mean():.1%} จึงใช้ stratify, วัดผลด้วย F1 และ ROC-AUC นอกจาก Accuracy "
               "และมีตัวเลือก class_weight = balanced")

    st.header("2.6 โค้ด Preprocessing")
    st.code('''df = df.drop_duplicates()
df["Description"] = df["Description"].fillna("")
y = (df["AdoptionSpeed"] < 4).astype(int)            # Target

X = engineer(df)                                     # Feature engineering
X[CLIP_COLS] = X[CLIP_COLS].clip(lower=P1, upper=P99, axis=1)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), NUM_COLS),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)
model = Pipeline([("prep", preprocessor), ("model", RandomForestClassifier())])
model.fit(X_train, y_train)''', language="python")


THEORY = [
    ("Logistic Regression", "Linear model · Probabilistic",
     "คำนวณผลรวมถ่วงน้ำหนักของ features แล้วแปลงเป็นความน่าจะเป็น 0-1 ด้วยฟังก์ชัน Sigmoid "
     "ถ้าความน่าจะเป็นเกิน 0.5 ทำนายว่าถูกรับเลี้ยง หาค่าน้ำหนักโดยลด Log loss",
     [r"P(y=1 \mid x) = \sigma(w^\top x + b) = \frac{1}{1 + e^{-(w^\top x + b)}}",
      r"\text{Loss} = -\frac{1}{n}\sum \left[y_i \log \hat{p}_i + (1-y_i)\log(1-\hat{p}_i)\right] + \frac{1}{2C}\lVert w \rVert^2"],
     "C (ความแรงของ Regularization)", "เร็ว ตีความน้ำหนักได้ ให้ความน่าจะเป็น", "จับความสัมพันธ์ไม่เชิงเส้นไม่ได้"),
    ("K-Nearest Neighbor (KNN)", "Instance-based · Lazy learning",
     "หาสัตว์ K ตัวใน Train set ที่ลักษณะใกล้เคียงที่สุด แล้วใช้เสียงข้างมากของ K ตัวนั้นเป็นคำตอบ",
     [r"d(x, x_i) = \sqrt{\sum_{j}(x_j - x_{ij})^2}", r"\hat{y} = \operatorname{mode}\{y_i : i \in N_K(x)\}"],
     "K, weights (uniform / distance)", "เข้าใจง่าย ไม่ต้องฝึก", "ทำนายช้า ไวต่อ scale และ feature จำนวนมาก"),
    ("Decision Tree", "Tree-based",
     "สร้างต้นไม้ของคำถาม ใช่/ไม่ใช่ เช่น PhotoAmt <= 2 ? โดยเลือกจุดแบ่งที่ทำให้กลุ่มย่อยบริสุทธิ์ที่สุด",
     [r"\text{Gini} = 1 - \sum_k p_k^2 \qquad \text{Entropy} = -\sum_k p_k \log_2 p_k",
      r"\text{Gain} = I(\text{parent}) - \sum_j \frac{N_j}{N} I(\text{child}_j)"],
     "max_depth, min_samples_leaf, criterion", "ตีความง่าย เห็นกฎการตัดสินใจ", "Overfit ง่ายถ้าต้นไม้ลึก"),
    ("Random Forest", "Ensemble · Bagging",
     "สร้าง Decision Tree หลายต้นจากข้อมูลที่สุ่มแบบ Bootstrap และสุ่ม features ในแต่ละการแบ่ง "
     "แล้วโหวตผล ทำให้ความแปรปรวนลดลงและแม่นยำกว่าต้นไม้ต้นเดียว",
     [r"\hat{y} = \operatorname{mode}\{h_1(x), \dots, h_T(x)\}"],
     "n_estimators, max_depth, min_samples_leaf", "แม่นยำ ทนต่อ overfitting บอก Feature importance ได้",
     "ตีความยากกว่าต้นไม้ต้นเดียว ใช้หน่วยความจำมาก"),
    ("Gradient Boosting", "Ensemble · Boosting",
     "สร้างต้นไม้ทีละต้นแบบต่อเนื่อง ต้นใหม่เรียนรู้จากความผิดพลาด (gradient ของ loss) ของต้นก่อนหน้า "
     "แล้วบวกผลรวมกันด้วย learning rate (ใช้ HistGradientBoosting ของ scikit-learn)",
     [r"F_m(x) = F_{m-1}(x) + \eta \, h_m(x)", r"h_m \approx -\frac{\partial L(y, F_{m-1}(x))}{\partial F_{m-1}(x)}"],
     "learning_rate, max_iter (จำนวนต้นไม้), max_depth", "มักแม่นยำที่สุดกับข้อมูลตาราง",
     "ต้องจูนหลายพารามิเตอร์ ถ้า learning rate สูงเกินจะ overfit"),
]


def page_theory():
    st.title("3. การสร้างโมเดล ML และทฤษฎี")
    st.write("**Workflow:** ข้อมูลดิบ → Preprocessing (Pipeline) → แบ่ง Train/Test → ฝึกโมเดล → ประเมินผล → ใช้งานบน Streamlit")
    st.dataframe(pd.DataFrame({
        "โมเดล": [t[0] for t in THEORY], "กลุ่ม": [t[1] for t in THEORY],
        "scikit-learn": ["LogisticRegression", "KNeighborsClassifier", "DecisionTreeClassifier",
                         "RandomForestClassifier", "HistGradientBoostingClassifier"],
    }, index=range(1, 6)), width="stretch")
    st.caption("ใช้ DummyClassifier (ทายคลาสที่พบบ่อยที่สุดเสมอ) เป็น Baseline เพื่อดูว่าโมเดลเรียนรู้ได้จริงหรือไม่")
    tabs = st.tabs([t[0].split(" (")[0] for t in THEORY])
    for tab, (name, group, idea, eqs, params, pros, cons) in zip(tabs, THEORY):
        with tab:
            st.subheader(name)
            st.caption(group)
            st.markdown(f"**แนวคิด:** {idea}")
            for e in eqs:
                st.latex(e)
            st.markdown(f"**Hyperparameters:** {params}")
            c1, c2 = st.columns(2)
            c1.success(f"**ข้อดี:** {pros}")
            c2.warning(f"**ข้อจำกัด:** {cons}")
    st.header("ตัวชี้วัด")
    c1, c2 = st.columns(2)
    with c1:
        st.latex(r"\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN}")
        st.latex(r"\text{Precision} = \frac{TP}{TP + FP} \quad \text{Recall} = \frac{TP}{TP + FN}")
    with c2:
        st.latex(r"F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}")
        st.latex(r"\text{Macro-}F_1 = \frac{1}{C}\sum_{c=1}^{C} F_{1,c}")
        st.markdown("**ROC-AUC:** พื้นที่ใต้กราฟ ROC (TPR เทียบ FPR ทุก threshold) 0.5 = เดาสุ่ม, 1.0 = สมบูรณ์")


@st.cache_data(show_spinner="กำลังฝึกและประเมินโมเดลทั้งหมด ...", max_entries=3)
def evaluate_all(clip, bal, ts, rs):
    _, Xa, ya, _, _ = prepare(clip)
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=ts, random_state=rs, stratify=ya)
    cv = StratifiedKFold(5, shuffle=True, random_state=rs)
    rows, rocs, cms = [], {}, {}
    for name in ["Baseline (Dummy)"] + MODEL_NAMES:
        m = make_model(name, balanced=bal, rs=rs)
        t0 = time.time()
        m.fit(Xtr, ytr)
        fit_t = time.time() - t0
        p = m.predict(Xte)
        prob = m.predict_proba(Xte)[:, 1]
        cvs = cross_val_score(make_model(name, balanced=bal, rs=rs), Xtr, ytr, cv=cv, scoring="roc_auc", n_jobs=1)
        rows.append({
            "โมเดล": name,
            "Train Accuracy": accuracy_score(ytr, m.predict(Xtr)),
            "Test Accuracy": accuracy_score(yte, p),
            "Precision (macro)": precision_score(yte, p, average="macro", zero_division=0),
            "Recall (macro)": recall_score(yte, p, average="macro", zero_division=0),
            "F1 (macro)": f1_score(yte, p, average="macro", zero_division=0),
            "ROC-AUC": roc_auc_score(yte, prob),
            "CV ROC-AUC (5-fold)": cvs.mean(),
            "CV Std": cvs.std(),
            "เวลาฝึก (วินาที)": fit_t,
        })
        fpr, tpr, _ = roc_curve(yte, prob)
        rocs[name] = (fpr, tpr)
        cms[name] = confusion_matrix(yte, p, labels=[0, 1])
        if name == "Random Forest":
            imp = pd.Series(m.named_steps["model"].feature_importances_,
                            index=m.named_steps["prep"].get_feature_names_out()).sort_values(ascending=False)
    return pd.DataFrame(rows).set_index("โมเดล"), rocs, cms, imp


def page_compare():
    st.title("4. การประเมินและเปรียบเทียบโมเดล")
    st.write(f"ฝึกทุกโมเดลด้วย Train set เดียวกัน ({len(X_train):,} ตัว) และวัดผลกับ Test set ({len(X_test):,} ตัว) "
             f"พร้อม 5-fold Cross-validation  ·  class_weight = {'balanced' if balanced else 'None'}")
    res, rocs, cms, imp = evaluate_all(bool(clip_outliers), bool(balanced), float(test_size), random_state)
    models_only = res.drop(index="Baseline (Dummy)")
    best = models_only["ROC-AUC"].idxmax()
    c = st.columns(4)
    c[0].metric("โมเดลที่ดีที่สุด (ROC-AUC)", best)
    c[1].metric("ROC-AUC", f"{res.loc[best, 'ROC-AUC']:.4f}",
                f"+{res.loc[best, 'ROC-AUC'] - res.loc['Baseline (Dummy)', 'ROC-AUC']:.3f} จาก Baseline")
    c[2].metric("Test Accuracy", f"{res.loc[best, 'Test Accuracy']:.4f}")
    c[3].metric("F1 (macro)", f"{res.loc[best, 'F1 (macro)']:.4f}")

    st.header("4.1 ตารางเปรียบเทียบ")
    st.caption("Precision / Recall / F1 เป็นค่าเฉลี่ยแบบ macro ของทั้ง 2 คลาส เหมาะกับข้อมูลไม่สมดุล "
               "(ไฮไลต์ค่าที่ดีที่สุดโดยไม่นับ Baseline)")
    st.dataframe(res.style.format("{:.4f}").highlight_max(
        subset=pd.IndexSlice[MODEL_NAMES, ["Test Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)",
                                           "ROC-AUC", "CV ROC-AUC (5-fold)"]],
        color="#c8f7c5"), width="stretch")

    st.header("4.2 กราฟเปรียบเทียบ")
    left, right = st.columns(2)
    with left:
        st.markdown("**ตัวชี้วัดบน Test set**")
        mets = ["Test Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)", "ROC-AUC"]
        fig, ax = plt.subplots(figsize=(7, 4.2))
        xs = np.arange(len(res))
        w = 0.16
        for i, mname in enumerate(mets):
            ax.bar(xs + (i - 2) * w, res[mname], w, label=mname)
        ax.set_xticks(xs, [n.replace(" (Dummy)", "").replace("Logistic Regression", "Logistic")
                           .replace("Gradient Boosting", "Grad. Boost") for n in res.index], fontsize=8, rotation=15)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=7, ncol=3, loc="lower right")
        fig.tight_layout()
        fig_show(fig)
    with right:
        st.markdown("**ROC Curve**")
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for name, (fpr, tpr) in rocs.items():
            ax.plot(fpr, tpr, lw=2 if name == best else 1.2, ls="--" if "Dummy" in name else "-",
                    label=f"{name} (AUC={res.loc[name, 'ROC-AUC']:.3f})")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig_show(fig)
    left, right = st.columns(2)
    with left:
        st.markdown("**Train vs Test Accuracy (ตรวจ Overfitting)**")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        xs = np.arange(len(models_only))
        ax.bar(xs - 0.2, models_only["Train Accuracy"], 0.4, label="Train", color="#9aa5b1")
        ax.bar(xs + 0.2, models_only["Test Accuracy"], 0.4, label="Test", color="#e07a5f")
        ax.set_xticks(xs, [n.replace("Logistic Regression", "Logistic") for n in models_only.index], fontsize=8)
        ax.set_ylim(0.5, 1.02)
        ax.legend()
        fig.tight_layout()
        fig_show(fig)
    with right:
        st.markdown("**5-fold Cross-validation ROC-AUC (mean ± std)**")
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.errorbar(range(len(models_only)), models_only["CV ROC-AUC (5-fold)"], yerr=models_only["CV Std"],
                    fmt="o", capsize=6, color="#3d405b")
        ax.set_xticks(range(len(models_only)),
                      [n.replace("Logistic Regression", "Logistic") for n in models_only.index], fontsize=8)
        ax.set_ylabel("ROC-AUC")
        fig.tight_layout()
        fig_show(fig)

    st.header("4.3 Confusion Matrix")
    cols = st.columns(len(MODEL_NAMES))
    for col, name in zip(cols, MODEL_NAMES):
        with col:
            fig, ax = plt.subplots(figsize=(3, 2.9))
            ConfusionMatrixDisplay(cms[name], display_labels=["No", "Yes"]).plot(ax=ax, cmap="Oranges", colorbar=False)
            ax.set_title(name.replace("Logistic Regression", "Logistic"), fontsize=9)
            fig.tight_layout()
            fig_show(fig)

    st.header("4.4 ปัจจัยที่มีผลต่อการรับเลี้ยง (Random Forest Feature Importance)")
    top = imp.head(12)[::-1]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.barh([n.replace("num__", "").replace("cat__", "") for n in top.index], top.values, color="#81b29a")
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig_show(fig)

    st.header("4.5 สรุปผล")
    gap = models_only["Train Accuracy"] - models_only["Test Accuracy"]
    top3 = ", ".join(n.replace("num__", "").replace("cat__", "") for n in imp.head(3).index)
    st.markdown(f"""
- **{best}** ทำได้ดีที่สุด ROC-AUC **{res.loc[best, 'ROC-AUC']:.4f}** (CV {res.loc[best, 'CV ROC-AUC (5-fold)']:.4f} ± {res.loc[best, 'CV Std']:.4f})
- ทุกโมเดลได้ ROC-AUC และ F1 (macro) สูงกว่า Baseline ที่ทายว่า "ถูกรับเลี้ยง" เสมอ \
(Baseline: Accuracy {res.loc['Baseline (Dummy)', 'Test Accuracy']:.4f} แต่ F1 (macro) เพียง {res.loc['Baseline (Dummy)', 'F1 (macro)']:.4f} และ ROC-AUC 0.5) \
แสดงว่าโมเดลเรียนรู้รูปแบบจากข้อมูลได้จริง
- Accuracy อย่างเดียววัดผลไม่พอ เพราะข้อมูลไม่สมดุล การทายคลาสเดียวตลอดก็ได้ Accuracy สูงแล้ว จึงต้องดู F1 (macro) และ ROC-AUC ประกอบ
- **{gap.idxmax()}** มีช่องว่าง Train-Test มากที่สุด ({gap.max():.3f}) แสดงถึงการ Overfit
- ปัจจัยสำคัญ 3 อันดับแรก: **{top3}**
""")


def page_app():
    st.title("5. Streamlit Application")
    tab1, tab2 = st.tabs(["ทำนายโอกาสรับเลี้ยง", "ทดลองปรับโมเดล"])
    with tab1:
        st.write("กรอกข้อมูลสัตว์เลี้ยง แล้วให้ทุกโมเดลทำนายว่าจะถูกรับเลี้ยงภายใน 100 วันหรือไม่")
        with st.form("predict"):
            c = st.columns(4)
            ptype = c[0].selectbox("ชนิด", ["Dog", "Cat"])
            age = c[1].number_input("อายุ (เดือน)", 0, 255, 3)
            gender = c[2].selectbox("เพศ", ["Male", "Female"])
            breed = c[3].selectbox("สายพันธุ์", sorted(raw["Breed1"].astype(str).unique().tolist()),
                                   index=sorted(raw["Breed1"].astype(str).unique().tolist()).index(top_breeds[0]))
            c = st.columns(4)
            color1 = c[0].selectbox("สีหลัก", sorted(raw["Color1"].astype(str).unique().tolist()))
            color2 = c[1].selectbox("สีรอง", sorted(raw["Color2"].astype(str).unique().tolist()))
            size = c[2].selectbox("ขนาดเมื่อโต", [s for s in ORDINAL_MAPS["MaturitySize"] if s in set(raw["MaturitySize"])])
            fur = c[3].selectbox("ความยาวขน", [s for s in ORDINAL_MAPS["FurLength"] if s in set(raw["FurLength"])])
            c = st.columns(4)
            vacc = c[0].selectbox("ฉีดวัคซีน", sorted(raw["Vaccinated"].astype(str).unique().tolist()))
            steril = c[1].selectbox("ทำหมัน", sorted(raw["Sterilized"].astype(str).unique().tolist()))
            health = c[2].selectbox("สุขภาพ", [s for s in ORDINAL_MAPS["Health"] if s in set(raw["Health"])])
            fee = c[3].number_input("ค่าธรรมเนียม (ริงกิต)", 0, 3000, 0)
            c = st.columns([1, 3])
            photos = c[0].number_input("จำนวนรูป", 0, 30, 3)
            desc = c[1].text_area("คำบรรยายประกาศ", "Very friendly and playful puppy, vaccinated and healthy. "
                                  "Looking for a loving home.", height=80)
            ok = st.form_submit_button("ทำนาย")
        if ok:
            new = pd.DataFrame([{
                "Type": ptype, "Age": age, "Breed1": breed, "Gender": gender, "Color1": color1, "Color2": color2,
                "MaturitySize": size, "FurLength": fur, "Vaccinated": vacc, "Sterilized": steril, "Health": health,
                "Fee": fee, "Description": desc, "PhotoAmt": photos,
            }])
            Xn = engineer(new, top_breeds)
            for col, (lo, hi) in clip_bounds.items():
                Xn[col] = Xn[col].clip(lo, hi)
            out = []
            with st.spinner("กำลังทำนาย ..."):
                for name in MODEL_NAMES:
                    m = train_one(name, (), bool(clip_outliers), bool(balanced), float(test_size), random_state)
                    pr = float(m.predict_proba(Xn)[0, 1])
                    out.append({"โมเดล": name, "โอกาสถูกรับเลี้ยง": pr, "ผลทำนาย": CLASS_NAMES[int(pr >= 0.5)]})
            out = pd.DataFrame(out).set_index("โมเดล")
            avg = out["โอกาสถูกรับเลี้ยง"].mean()
            votes = (out["โอกาสถูกรับเลี้ยง"] >= 0.5).sum()
            if avg >= 0.5:
                st.success(f"### มีโอกาสถูกรับเลี้ยงภายใน 100 วัน: {avg:.0%}  ({votes}/{len(out)} โมเดลเห็นด้วย)")
            else:
                st.error(f"### มีความเสี่ยงไม่ถูกรับเลี้ยงภายใน 100 วัน (โอกาสได้บ้าน {avg:.0%}, {votes}/{len(out)} โมเดล)")
            st.progress(min(max(avg, 0.0), 1.0))
            st.dataframe(out.style.format({"โอกาสถูกรับเลี้ยง": "{:.1%}"}), width="stretch")
            tips = []
            if photos < 3:
                tips.append("เพิ่มรูปในประกาศ (ประกาศที่มีรูปหลายรูปมักได้บ้านเร็วกว่า)")
            if len(desc.split()) < 20:
                tips.append("เขียนคำบรรยายให้ละเอียดขึ้น")
            if fee > 0:
                tips.append("ลองลดค่าธรรมเนียมหรือรับเลี้ยงฟรี")
            if vacc != "Yes":
                tips.append("ฉีดวัคซีนก่อนลงประกาศ")
            if tips:
                st.info("**คำแนะนำเพิ่มโอกาส:**  \n" + "  \n".join(f"- {t}" for t in tips))
    with tab2:
        name = st.selectbox("เลือกโมเดล", MODEL_NAMES, index=3)
        c = st.columns(3)
        if name == "Logistic Regression":
            params = {"C": c[0].select_slider("C", [0.01, 0.1, 0.5, 1.0, 5.0, 10.0], value=1.0)}
        elif name == "KNN":
            params = {"k": c[0].slider("K", 1, 50, 15),
                      "weights": c[1].selectbox("weights", ["uniform", "distance"])}
        elif name == "Decision Tree":
            params = {"max_depth": c[0].slider("max_depth", 1, 20, 6),
                      "min_leaf": c[1].slider("min_samples_leaf", 1, 100, 20),
                      "criterion": c[2].selectbox("criterion", ["gini", "entropy"])}
        elif name == "Random Forest":
            params = {"n_estimators": c[0].slider("n_estimators", 10, 400, 200, step=10),
                      "max_depth": c[1].slider("max_depth", 2, 30, 12),
                      "min_leaf": c[2].slider("min_samples_leaf", 1, 20, 2)}
        else:
            params = {"lr": c[0].select_slider("learning_rate", [0.01, 0.05, 0.1, 0.2, 0.5], value=0.1),
                      "n_iter": c[1].slider("max_iter (จำนวนต้นไม้)", 20, 500, 200, step=20),
                      "max_depth": c[2].slider("max_depth", 2, 15, 6)}
        m = train_one(name, tuple(sorted(params.items())), bool(clip_outliers), bool(balanced),
                      float(test_size), random_state)
        p = m.predict(X_test)
        prob = m.predict_proba(X_test)[:, 1]
        c = st.columns(5)
        c[0].metric("Accuracy", f"{accuracy_score(y_test, p):.4f}")
        c[1].metric("Precision (macro)", f"{precision_score(y_test, p, average='macro', zero_division=0):.4f}")
        c[2].metric("Recall (macro)", f"{recall_score(y_test, p, average='macro', zero_division=0):.4f}")
        c[3].metric("F1 (macro)", f"{f1_score(y_test, p, average='macro', zero_division=0):.4f}")
        c[4].metric("ROC-AUC", f"{roc_auc_score(y_test, prob):.4f}")
        left, right = st.columns(2)
        with left:
            fig, ax = plt.subplots(figsize=(4.5, 4))
            ConfusionMatrixDisplay(confusion_matrix(y_test, p, labels=[0, 1]),
                                   display_labels=CLASS_NAMES_EN).plot(ax=ax, cmap="Oranges", colorbar=False)
            fig.tight_layout()
            fig_show(fig)
        with right:
            if name == "Decision Tree":
                fig, ax = plt.subplots(figsize=(10, 5))
                plot_tree(m.named_steps["model"], max_depth=2, filled=True, rounded=True, fontsize=7,
                          feature_names=[n.replace("num__", "").replace("cat__", "")
                                         for n in m.named_steps["prep"].get_feature_names_out()],
                          class_names=CLASS_NAMES_EN, ax=ax)
                fig_show(fig)
            else:
                fpr, tpr, _ = roc_curve(y_test, prob)
                fig, ax = plt.subplots(figsize=(4.5, 4))
                ax.plot(fpr, tpr, color="#e07a5f", lw=2)
                ax.plot([0, 1], [0, 1], "k--", lw=1)
                ax.set_xlabel("False Positive Rate")
                ax.set_ylabel("True Positive Rate")
                ax.set_title("ROC Curve")
                fig.tight_layout()
                fig_show(fig)


@st.cache_resource(show_spinner="กำลังฝึกโมเดล ...", max_entries=8)
def train_one(name, params, clip, bal, ts, rs):
    _, Xa, ya, _, _ = prepare(clip)
    Xtr, _, ytr, _ = train_test_split(Xa, ya, test_size=ts, random_state=rs, stratify=ya)
    return make_model(name, dict(params), balanced=bal, rs=rs).fit(Xtr, ytr)



def youtube_url(url):
    """แปลงลิงก์ YouTube ทุกรูปแบบให้เหลือ watch?v=ID (ตัด fbclid ฯลฯ)"""
    import re
    m = re.search(r"(?:v=|youtu\.be/|shorts/|embed/)([A-Za-z0-9_-]{11})", url or "")
    return f"https://www.youtube.com/watch?v={m.group(1)}" if m else None


def list_files(*exts):
    found = []
    for f in sorted(BASE_DIR.rglob("*")):
        if f.suffix.lower() in exts and ".venv" not in f.parts and "_video_work" not in f.parts:
            found.append(f)
    return found


def notebook_preview(path, max_cells=40):
    import json
    try:
        nb = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        st.warning(f"อ่านไฟล์ notebook ไม่ได้: {e}")
        return
    cells = nb.get("cells", [])
    st.caption(f"ทั้งหมด {len(cells)} cells (แสดง {min(len(cells), max_cells)} cells แรก)")
    for cell in cells[:max_cells]:
        src = "".join(cell.get("source", []))
        if not src.strip():
            continue
        if cell.get("cell_type") == "markdown":
            st.markdown(src)
        elif cell.get("cell_type") == "code":
            st.code(src, language="python")
            for out in cell.get("outputs", [])[:2]:
                text = "".join(out.get("text", [])) or "".join(out.get("data", {}).get("text/plain", []))
                if text.strip():
                    st.text(text[:1500])


def page_files():
    st.title("ไฟล์ประกอบโปรเจกต์")
    st.write("รวมผลงานทั้งหมดของโปรเจกต์: วิดีโอนำเสนอ, โค้ดบน Google Colab, สไลด์ PowerPoint และ GitHub")

    # ---------- YouTube
    st.header("วิดีโอนำเสนอ (YouTube)")
    yt = youtube_url(PROJECT_LINKS.get("youtube"))
    if yt:
        st.video(yt)
        st.markdown(f"[เปิดใน YouTube]({yt})")
    else:
        local = list_files(".mp4")
        if local:
            st.info("ยังไม่ได้ใส่ลิงก์ YouTube จึงแสดงวิดีโอจากไฟล์ในโปรเจกต์แทน")
            st.video(str(local[0]))
        else:
            st.info("ยังไม่ได้ใส่ลิงก์ YouTube (แก้ที่ PROJECT_LINKS[\"youtube\"] ด้านบนของ app.py)")

    # ---------- Colab
    st.header("โค้ดบน Google Colab")
    if PROJECT_LINKS.get("colab"):
        st.link_button("เปิดใน Google Colab", PROJECT_LINKS["colab"], type="primary")
    nbs = list_files(".ipynb")
    for nb in nbs:
        st.download_button(f"ดาวน์โหลด {nb.name}", nb.read_bytes(), file_name=nb.name,
                           mime="application/x-ipynb+json", key=f"dl_{nb.name}")
        with st.expander(f"ดูเนื้อหา {nb.name}"):
            notebook_preview(nb)
    if not PROJECT_LINKS.get("colab") and not nbs:
        st.info("ยังไม่มีไฟล์ Colab: ใส่ลิงก์ที่ PROJECT_LINKS[\"colab\"] หรืออัปโหลดไฟล์ .ipynb ไว้ในโฟลเดอร์โปรเจกต์")

    # ---------- PowerPoint
    st.header("สไลด์นำเสนอ (PowerPoint)")
    slides = PROJECT_LINKS.get("slides", "")
    if slides:
        embed = slides
        if "docs.google.com/presentation" in slides:
            embed = slides.split("/edit")[0].split("/pub")[0] + "/embed?start=false&loop=false"
        if "docs.google.com/presentation" in slides or "canva.com" in slides:
            components.iframe(embed, height=520)
        st.link_button("เปิดสไลด์", slides)
    decks = list_files(".pptx", ".pdf")
    for f in decks:
        mime = "application/pdf" if f.suffix.lower() == ".pdf" else \
            "application/vnd.openxmlformats-officedocument.presentationml.presentation"
        st.download_button(f"ดาวน์โหลด {f.name}", f.read_bytes(), file_name=f.name, mime=mime, key=f"dl_{f.name}")
    if not slides and not decks:
        st.info("ยังไม่มีสไลด์: อัปโหลดไฟล์ .pptx ไว้ในโฟลเดอร์โปรเจกต์ หรือใส่ลิงก์ที่ PROJECT_LINKS[\"slides\"]")

    # ---------- GitHub
    st.header("Source code (GitHub)")
    st.link_button("เปิด GitHub Repository", PROJECT_LINKS["github"])
    st.caption(PROJECT_LINKS["github"])


PAGES = dict(zip(MENU, [page_home, page_problem, page_preprocess, page_theory, page_compare, page_app,
                        page_files]))
PAGES[page]()

st.divider()
st.caption(f"พัฒนาโดย {DEVELOPER['name']} | รหัส {DEVELOPER['student_id']} | {DEVELOPER['section']}")
