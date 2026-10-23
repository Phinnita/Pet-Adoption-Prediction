# Pet Adoption Prediction

เว็บแอป Streamlit ทำนายว่าสุนัขหรือแมวจะถูกรับเลี้ยงภายใน 100 วันหรือไม่ ด้วยชุดข้อมูลจริง **PetFinder.my** (ฉบับย่อของ TensorFlow)
และเปรียบเทียบโมเดล Machine Learning 5 แบบ

## ผู้พัฒนา
- ชื่อ-นามสกุล: นางสาวภิญนิตา ซิ่วมณี
- รหัสนักศึกษา: 664245042
- หมู่เรียน: 66/44

## หัวข้อในเว็บไซต์
1. การกำหนดปัญหาและ Dataset
2. Data Preprocessing
3. การสร้างโมเดล ML (ทฤษฎี)
4. การประเมินและเปรียบเทียบโมเดล
5. Streamlit Application

## โมเดล
Logistic Regression · K-Nearest Neighbor · Decision Tree · Random Forest · Gradient Boosting (เทียบกับ Baseline)

## วิธีรัน
```bash
pip install -r requirements.txt
python download_data.py     # ครั้งแรก (หรือแอปจะดาวน์โหลดเอง)
streamlit run app.py
```

## ไฟล์
- `app.py` โค้ดหลัก
- `petfinder-mini.csv` ข้อมูล
- `profile.jpg` รูปผู้พัฒนา
- `requirements.txt`
